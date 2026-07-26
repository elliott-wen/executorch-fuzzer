# CMakeLists.txt — build vgf_runner_io (host x86 file-I/O runner for the Arm VGF delegate).
# Links the installed ExecuTorch core (find_package) + the VGF runtime delegate (vgf_backend)
# + the portable CPU kernels (portable_ops_lib for non-delegated ops). The ExecuTorch core in
# ${CMAKE_PREFIX_PATH} MUST have been built with -DEXECUTORCH_BUILD_VGF=ON so vgf_backend is
# exported (build_runner.sh installs such a core into pytorch_ref/executorch/cmake-out-vgf).
#
# Configure with (build_runner.sh wires all of this up):
#   cmake -S mobile/vgf_client -B <build> -G Ninja \
#     -DEXECUTORCH_ROOT=<abs .../pytorch_ref/executorch> \
#     -DCMAKE_PREFIX_PATH="<ET_ROOT>/cmake-out-vgf/lib64/cmake/ExecuTorch;<ET_ROOT>/cmake-out-vgf/third-party/gflags" \
#     -DPYTHON_EXECUTABLE=<.venv/bin/python> -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.19)
project(vgf_runner_io C CXX)

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

# vgf_backend links libvgf.a (from the vgf_lib pip wheel) PRIVATE, but the imported `vgf`
# target is NOT in the ExecuTorch export set — so the exported interface emits a bare `-lvgf`.
# Add the wheel's lib dir to the linker search path so that `-lvgf` resolves to libvgf.a.
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
    "import vgf_lib,os;print(os.path.join(os.path.dirname(vgf_lib.__file__),'binaries','lib'))"
  OUTPUT_VARIABLE _libvgf_dir OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT EXISTS "${_libvgf_dir}/libvgf.a")
  message(FATAL_ERROR "libvgf.a not found under '${_libvgf_dir}' — pip install ai_ml_sdk_vgf_library")
endif()
link_directories(${_libvgf_dir})

# volk — the Vulkan function-pointer loader that vgf_backend was compiled against
# (-DUSE_VULKAN_VOLK). With EXECUTORCH_BUILD_VULKAN=OFF the ET core does not compile volk, so
# vgf_backend's vk* references are unresolved. Compile the vendored volk.c here and link it so
# those symbols are defined (and volkInitialize() dlopens libvulkan.so.1 at runtime). This
# avoids enabling the full Vulkan delegate, which would drag in the glslc shader toolchain.
set(_vulkan_tp ${EXECUTORCH_ROOT}/backends/vulkan/third-party)
add_library(vgf_volk STATIC ${_vulkan_tp}/volk/volk.c)
target_include_directories(
  vgf_volk PUBLIC ${_vulkan_tp}/volk ${_vulkan_tp}/Vulkan-Headers/include
)
target_compile_definitions(vgf_volk PUBLIC USE_VULKAN_WRAPPER USE_VULKAN_VOLK)

add_executable(vgf_runner_io ${CMAKE_CURRENT_LIST_DIR}/vgf_runner_io.cpp)
target_include_directories(
  vgf_runner_io PUBLIC ${_common_include_directories}
)
# Link the ExecuTorch core, the file data loader, the VGF delegate, and the portable kernels.
# vgf_backend and portable_ops_lib already carry the whole-archive link option (stamped by
# executorch_target_link_options_shared_lib / gen_operators_lib in the ET build), so their
# static initializers — the VGF Backend registration and the aten kernel registrations —
# survive the link. Both come from the find_package(executorch) export set.
target_link_libraries(
  vgf_runner_io
  executorch
  gflags
  extension_data_loader
  vgf_backend
  portable_ops_lib
  vgf_volk
  ${CMAKE_DL_LIBS}
)
