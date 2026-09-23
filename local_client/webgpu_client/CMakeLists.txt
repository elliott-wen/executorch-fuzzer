# CMakeLists.txt — build webgpu_runner_io (host x86 file-I/O runner for the ExecuTorch
# WebGPU delegate). Links the installed ExecuTorch core (find_package) + the WebGPU runtime
# delegate (webgpu_backend) + the portable CPU kernels (portable_ops_lib for non-delegated
# ops). The core in ${CMAKE_PREFIX_PATH} MUST have been built with -DEXECUTORCH_BUILD_WEBGPU=ON
# so webgpu_backend is exported (build_runner.sh installs such a core into cmake-out-webgpu).
#
# WHY A SEPARATE RUNNER AND NOT A FLAG ON THE VULKAN ONE: webgpu_backend registers itself as
# Backend{"VulkanBackend", ...} — the SAME delegate id vulkan_backend claims (see
# backends/webgpu/runtime/WebGPUBackend.cpp). Two backends cannot answer to one id in a single
# process, so the choice of runtime is a LINK-TIME one and needs its own binary. The .pte is
# byte-identical either way; only this link line differs.
#
# Dawn: upstream builds the WebGPU delegate against Google's Dawn (find_package Dawn, in
# backends/webgpu/CMakeLists.txt, links dawn::webgpu_dawn). There is no FetchContent for it,
# so build_runner.sh downloads Google's pinned prebuilt and passes Dawn_DIR.
cmake_minimum_required(VERSION 3.19)
project(webgpu_runner_io C CXX)

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

# Threads BEFORE Dawn: dawn::webgpu_dawn's link interface names Threads::Threads, and
# FindThreads additionally requires a language to be enabled — so both of these must sit
# AFTER project(), not in the header. backends/webgpu/CMakeLists.txt pairs them the same way.
find_package(Threads REQUIRED)

# Dawn BEFORE executorch: the exported webgpu_backend target carries `dawn::webgpu_dawn` in
# its INTERFACE_LINK_LIBRARIES, so find_package(executorch) fails while generating with
#   The link interface of target "webgpu_backend" contains: dawn::webgpu_dawn
#   but the target was not found.
# Passing -DDawn_DIR alone is not enough — the imported target only exists once Dawn's own
# config has been found here too.
find_package(Dawn REQUIRED)

find_package(executorch CONFIG REQUIRED)
executorch_target_link_options_shared_lib(executorch)
target_include_directories(executorch INTERFACE ${_common_include_directories})

find_package(gflags REQUIRED)

add_executable(webgpu_runner_io ${CMAKE_CURRENT_LIST_DIR}/webgpu_runner_io.cpp)
target_include_directories(webgpu_runner_io PUBLIC ${_common_include_directories})

# webgpu_backend and portable_ops_lib already carry the whole-archive link option (stamped by
# executorch_target_link_options_shared_lib / gen_operators_lib in the ET build), so their
# static initializers — the WebGPU backend registration and the aten kernel registrations —
# survive the link. quantized_ops_lib is here for the same reason as in the vulkan client: a
# --quantize corpus leaves q/dq nodes outside the delegate, and without it the method fails to
# LOAD with "Missing operator: quantized_decomposed::quantize_per_tensor.out" (0x14), which is
# indistinguishable from a real backend coverage gap.
target_link_libraries(
  webgpu_runner_io
  executorch
  gflags
  extension_data_loader
  webgpu_backend
  portable_ops_lib
  quantized_ops_lib
  ${CMAKE_DL_LIBS}
)
