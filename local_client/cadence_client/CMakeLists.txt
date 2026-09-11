# CMakeLists.txt — build cadence_runner_io for one of two targets:
#
#   -DCADENCE_TARGET=generic  (default)  host x86 binary; cadence:: ops run on the CPU via the
#                                        generic REFERENCE kernels. 36 op impls.
#   -DCADENCE_TARGET=hifi                static xtensa-elf for the xt-run ISS; cadence:: ops run
#                                        via the HiFi4 nnlib DSP microkernels. 79 op impls.
#
# Configure with (build_runner.sh wires all of this up):
#   cmake -S mobile/cadence_client -B <build> -G Ninja \
#     -DCADENCE_TARGET=generic|hifi \
#     -DEXECUTORCH_ROOT=<abs .../pytorch_ref/executorch> \
#     -DCMAKE_PREFIX_PATH="<ET>/cmake-out[-xtensa]/lib*/cmake/ExecuTorch" \
#     -DPYTHON_EXECUTABLE=<.venv/bin/python> -DCMAKE_BUILD_TYPE=Release
# The hifi target additionally needs -DCMAKE_TOOLCHAIN_FILE=<ET>/backends/cadence/cadence.cmake
# and the Tensilica env (see xtensa_env.sh).
cmake_minimum_required(VERSION 3.19)
project(cadence_runner_io CXX C)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "pass -DEXECUTORCH_ROOT=<path to pytorch_ref/executorch>")
endif()

set(CADENCE_TARGET "generic" CACHE STRING "Cadence kernel target: generic or hifi")
set_property(CACHE CADENCE_TARGET PROPERTY STRINGS generic hifi)
if(NOT CADENCE_TARGET MATCHES "^(generic|hifi)$")
  message(FATAL_ERROR "CADENCE_TARGET must be 'generic' or 'hifi', got '${CADENCE_TARGET}'")
endif()
message(STATUS "cadence_runner_io: CADENCE_TARGET=${CADENCE_TARGET}")

# Mirror backends/cadence/CMakeLists.txt preamble so the op subdirs build the same way they do
# in-tree.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)
# ROOT_DIR is what the cadence operators/kernels CMakeLists reference via ${ROOT_DIR}/..;
# in-tree it is inherited from backends/cadence. Point it there.
set(ROOT_DIR ${EXECUTORCH_ROOT}/backends/cadence)
set(_cadence_dir ${EXECUTORCH_ROOT}/backends/cadence)

find_package(executorch CONFIG REQUIRED)
executorch_target_link_options_shared_lib(executorch)
target_include_directories(executorch INTERFACE ${_common_include_directories})

add_executable(cadence_runner_io ${CMAKE_CURRENT_LIST_DIR}/cadence_runner_io.cpp)
target_include_directories(
  cadence_runner_io PUBLIC ${ROOT_DIR}/.. ${_common_include_directories}
)

if(CADENCE_TARGET STREQUAL "generic")
  # ---- generic: reference kernels on the host CPU ---------------------------------------
  # Adds `cadence_kernels` and `custom_ops` (the cadence:: reference impls). Also defines an
  # in-tree `cadence_ops_lib`, which we deliberately IGNORE — see the ATen coverage note below.
  add_subdirectory(
    ${_cadence_dir}/generic/kernels ${CMAKE_BINARY_DIR}/cadence_generic_kernels
  )
  add_subdirectory(
    ${_cadence_dir}/generic/operators ${CMAKE_BINARY_DIR}/cadence_generic_operators
  )

  # ---- ATen op coverage ------------------------------------------------------------------
  # The in-tree `cadence_ops_lib` is codegen'd from the whole backends/cadence/aot/functions.yaml,
  # whose ATen half is a hand-picked 36-op subset (`aten_ops_cadence`). A runner linked against
  # it cannot LOAD any graph touching an ATen op outside that list — it dies with
  # `Missing operator: aten::<op>.out`, which was 72.5% of the corpus_v5/cadence corpus.
  #
  # Fix: codegen a SLIM ops lib from the cadence:: CUSTOM ops only, and get the ATen kernels from
  # the prebuilt `portable_ops_lib` (all 200 portable ops + dim_order_ops). The two registries are
  # disjoint by construction (`cadence::` vs `aten::`/`dim_order_ops::`), so nothing is registered
  # twice — which is what made naively adding portable_ops_lib abort at init before.
  #
  # This wholesale swap is safe ONLY for generic, whose ATen entries all name plain
  # `torch::executor::` kernels that portable_ops_lib provides identically. It would be WRONG for
  # hifi, whose ATen entries name the impl::HiFi:: kernels under test — hence the split below.
  set(_cadence_custom_yaml ${CMAKE_BINARY_DIR}/cadence_custom_ops.yaml)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gen_custom_ops_yaml.py
            ${_cadence_dir}/aot/functions.yaml ${_cadence_custom_yaml}
    RESULT_VARIABLE _split_rc OUTPUT_VARIABLE _split_out
  )
  if(NOT _split_rc EQUAL 0)
    message(FATAL_ERROR "gen_custom_ops_yaml.py failed: ${_split_out}")
  endif()
  message(STATUS "cadence custom-op yaml: ${_split_out}")

  gen_selected_ops(LIB_NAME "cadence_custom_ops_lib" OPS_SCHEMA_YAML "${_cadence_custom_yaml}")
  generate_bindings_for_kernels(
    LIB_NAME "cadence_custom_ops_lib" FUNCTIONS_YAML "${_cadence_custom_yaml}"
  )
  # KERNEL_LIBS custom_ops = the cadence:: generic (reference) impls from the subdir above.
  gen_operators_lib(
    LIB_NAME "cadence_custom_ops_lib" KERNEL_LIBS custom_ops DEPS executorch
  )

  # Link both registries PLAINLY: each already stamps its own whole-archive link option
  # (gen_operators_lib does it for ours, the ExecuTorch install does it for portable_ops_lib), so
  # wrapping either in another -Wl,--whole-archive double-includes the objects and double-registers
  # every kernel, which aborts at init.
  target_link_libraries(
    cadence_runner_io executorch extension_data_loader portable_ops_lib cadence_custom_ops_lib
  )
else()
  # ---- hifi: HiFi4 nnlib microkernels under the xt-run ISS ------------------------------
  # nnlib is built by its OWN makefile (see hifi/third-party/nnlib/CMakeLists.txt), which needs
  # the Tensilica env — XTENSA_CORE et al. Clone it first with:
  #   backends/cadence/install_requirements.sh   (or git clone foss-xtensa/nnlib-hifi4)
  if(NOT EXISTS ${_cadence_dir}/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib)
    message(
      FATAL_ERROR
        "hifi target needs nnlib-hifi4. Clone it into "
        "${_cadence_dir}/hifi/third-party/nnlib/nnlib-hifi4 (see backends/cadence/install_requirements.sh)"
    )
  endif()
  add_subdirectory(
    ${_cadence_dir}/hifi/third-party/nnlib ${CMAKE_BINARY_DIR}/cadence_nnlib
  )
  add_subdirectory(${_cadence_dir}/hifi/kernels ${CMAKE_BINARY_DIR}/cadence_hifi_kernels)
  # Defines `custom_ops` (cadence:: HiFi impls), `aten_ops_cadence` (57 HiFi-optimised ATen
  # kernels) and codegens `cadence_ops_lib` from aot/functions_hifi.yaml over both. Unlike the
  # generic case we KEEP cadence_ops_lib: its ATen half is the impl::HiFi:: code under test.
  add_subdirectory(${_cadence_dir}/hifi/operators ${CMAKE_BINARY_DIR}/cadence_hifi_operators)

  # UPSTREAM BUILD-LIST DRIFT: hifi/operators/ holds 79 op_*.cpp but its CMakeLists.txt references
  # only 76. Two of the three strays implement kernels that functions_hifi.yaml *does* declare, so
  # the codegen'd registry references them and the link dies with undefined
  # impl::HiFi::native::quantized_depthwise_conv1d_{ncl,nlc}_per_tensor_out. (The third stray,
  # op_softmax_f32_f32.cpp, is not named by the yaml, so it stays out.) Compiled here rather than
  # patching executorch. Listed AFTER cadence_ops_lib below so the linker can resolve into it.
  add_library(
    cadence_hifi_missing_ops
    ${_cadence_dir}/hifi/operators/op_quantized_depthwise_conv1d_ncl.cpp
    ${_cadence_dir}/hifi/operators/op_quantized_depthwise_conv1d_nlc.cpp
  )
  target_include_directories(
    cadence_hifi_missing_ops PUBLIC ${ROOT_DIR}/.. ${CMAKE_BINARY_DIR}
                                    ${_common_include_directories}
  )
  target_link_libraries(cadence_hifi_missing_ops PUBLIC executorch PRIVATE cadence_kernels)

  # ...but functions_hifi.yaml only covers 57 of the 200 portable ATen ops, so on its own it
  # reproduces the `Missing operator: aten::<op>.out` coverage cliff. Supplement it with the
  # COMPLEMENT of the portable registry (the 143 ops it does not claim, plus dim_order_ops),
  # codegen'd over the installed `portable_kernels`. Disjoint from cadence_ops_lib by
  # construction, so no kernel is registered twice.
  set(_complement_yaml ${CMAKE_BINARY_DIR}/portable_complement_ops.yaml)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gen_portable_complement_yaml.py
            ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
            ${_cadence_dir}/aot/functions_hifi.yaml ${_complement_yaml}
    RESULT_VARIABLE _comp_rc OUTPUT_VARIABLE _comp_out
  )
  if(NOT _comp_rc EQUAL 0)
    message(FATAL_ERROR "gen_portable_complement_yaml.py failed: ${_comp_out}")
  endif()
  message(STATUS "portable complement yaml: ${_comp_out}")

  gen_selected_ops(
    LIB_NAME "portable_complement_ops_lib" OPS_SCHEMA_YAML "${_complement_yaml}"
  )
  generate_bindings_for_kernels(
    LIB_NAME "portable_complement_ops_lib" FUNCTIONS_YAML "${_complement_yaml}"
  )
  gen_operators_lib(
    LIB_NAME "portable_complement_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
  )

  target_link_libraries(
    cadence_runner_io executorch extension_data_loader cadence_ops_lib
    cadence_hifi_missing_ops portable_complement_ops_lib
  )
endif()
