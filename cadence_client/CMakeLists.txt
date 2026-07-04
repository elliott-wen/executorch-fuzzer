# CMakeLists.txt — build cadence_runner_io (host x86 file-I/O runner for the Cadence
# generic/reference CPU kernels). Links the installed ExecuTorch core (find_package) + the
# cadence generic op library (cadence_ops_lib, built here from the executorch source subdirs).
#
# Configure with:
#   cmake -S mobile/cadence_client -B <build> -G Ninja \
#     -DEXECUTORCH_ROOT=<abs .../pytorch_ref/executorch> \
#     -DCMAKE_PREFIX_PATH="<ET_ROOT>/cmake-out/lib64/cmake/ExecuTorch;<ET_ROOT>/cmake-out/third-party/gflags" \
#     -DPYTHON_EXECUTABLE=<.venv/bin/python> -DCMAKE_BUILD_TYPE=Release
# (build_runner.sh wires all of this up.)
cmake_minimum_required(VERSION 3.19)
project(cadence_runner_io CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "pass -DEXECUTORCH_ROOT=<path to pytorch_ref/executorch>")
endif()

# Mirror backends/cadence/CMakeLists.txt preamble so the generic op subdirs build the same
# way they do in-tree.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)
# ROOT_DIR is what the generic operators/kernels CMakeLists reference via ${ROOT_DIR}/..;
# in-tree it is inherited from backends/cadence. Point it there.
set(ROOT_DIR ${EXECUTORCH_ROOT}/backends/cadence)

find_package(executorch CONFIG REQUIRED)
executorch_target_link_options_shared_lib(executorch)
# NOTE: deliberately NOT force-linking portable_ops_lib — cadence_ops_lib brings its own aten
# op subset, and adding portable_ops_lib double-registers overlapping kernels (aten::_to_copy.out).
target_include_directories(executorch INTERFACE ${_common_include_directories})

find_package(gflags REQUIRED)

# Build the generic (host reference) cadence kernels + cadence_ops_lib. These add the
# `cadence_kernels` and `cadence_ops_lib` targets (codegen from backends/cadence/aot/functions.yaml).
add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cadence/generic/kernels
  ${CMAKE_BINARY_DIR}/cadence_generic_kernels
)
add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/cadence/generic/operators
  ${CMAKE_BINARY_DIR}/cadence_generic_operators
)

add_executable(cadence_runner_io ${CMAKE_CURRENT_LIST_DIR}/cadence_runner_io.cpp)
target_include_directories(
  cadence_runner_io PUBLIC ${ROOT_DIR}/.. ${_common_include_directories}
)
# Link ONLY cadence_ops_lib (NOT portable_ops_lib): it bundles the aten op subset it needs
# (aten_ops_cadence) + the cadence:: custom ops. Link it PLAINLY — gen_operators_lib already
# stamps the whole-archive link option onto the target, so wrapping it in another
# -Wl,--whole-archive double-includes the objects and double-registers kernels
# (aten::_to_copy.out), which aborts at init.
target_link_libraries(
  cadence_runner_io
  executorch
  gflags
  extension_data_loader
  cadence_ops_lib
)
