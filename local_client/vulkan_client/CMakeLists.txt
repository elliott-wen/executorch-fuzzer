# CMakeLists.txt — build vulkan_runner_io (host x86 file-I/O runner for the ExecuTorch
# Vulkan delegate). Links the installed ExecuTorch core (find_package) + the Vulkan runtime
# delegate (vulkan_backend) + the portable CPU kernels (portable_ops_lib for non-delegated
# ops). The core in ${CMAKE_PREFIX_PATH} MUST have been built with -DEXECUTORCH_BUILD_VULKAN=ON
# so vulkan_backend is exported (build_runner.sh installs such a core into cmake-out-vulkan).
#
# Simpler than the VGF client's: enabling the Vulkan delegate makes the ET build compile volk
# and the SPIR-V shader library itself, so there is no vendored-volk target here and no
# external blob library to put on the link line.
cmake_minimum_required(VERSION 3.19)
project(vulkan_runner_io C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT EXECUTORCH_ROOT)
  message(FATAL_ERROR "pass -DEXECUTORCH_ROOT=<path to pytorch_ref/executorch>")
endif()

set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)
add_compile_definitions(C10_USING_CUSTOM_GENERATED_MACROS)

# Defines executorch_target_link_options_shared_lib (whole-archive helper) used below.
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

find_package(executorch CONFIG REQUIRED)
executorch_target_link_options_shared_lib(executorch)
target_include_directories(executorch INTERFACE ${_common_include_directories})

find_package(gflags REQUIRED)

add_executable(vulkan_runner_io ${CMAKE_CURRENT_LIST_DIR}/vulkan_runner_io.cpp)
target_include_directories(vulkan_runner_io PUBLIC ${_common_include_directories})

# vulkan_backend and portable_ops_lib already carry the whole-archive link option (stamped by
# executorch_target_link_options_shared_lib / gen_operators_lib in the ET build), so their
# static initializers — the Vulkan backend registration and the aten kernel registrations —
# survive the link. quantized_ops_lib is here for the same reason as in the VGF client: a
# --quantize corpus leaves q/dq nodes outside the delegate, and without it the method fails to
# LOAD with "Missing operator: quantized_decomposed::quantize_per_tensor.out" (0x14), which is
# indistinguishable from a real backend coverage gap.
target_link_libraries(
  vulkan_runner_io
  executorch
  gflags
  extension_data_loader
  vulkan_backend
  portable_ops_lib
  quantized_ops_lib
  ${CMAKE_DL_LIBS}
)
