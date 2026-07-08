cmake_minimum_required(VERSION 3.26)

project(marlin_v100_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
message(STATUS "marlin_v100 build type: ${CMAKE_BUILD_TYPE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12" "3.13")

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR "Please set VLLM_PYTHON_EXECUTABLE before configuring marlin_v100.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

if (NOT HIP_FOUND)
  set(VLLM_GPU_LANG "CUDA")
else()
  message(FATAL_ERROR "marlin_v100 currently supports CUDA-only builds.")
endif()

# find_program(NVCC_EXECUTABLE nvcc)
# if (NOT NVCC_EXECUTABLE)
#   message(FATAL_ERROR "nvcc not found")
# endif()

function(marlin_generate_kernels SCRIPT_PATH LOG_FILENAME FAILURE_MESSAGE)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${SCRIPT_PATH} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE generation_result
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LOG_FILENAME}
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LOG_FILENAME}
  )
  if (NOT generation_result EQUAL 0)
    message(FATAL_ERROR "${FAILURE_MESSAGE}")
  endif()
endfunction()

function(marlin_configure_cuda_sources)
  set(options WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
  set(oneValueArgs CUDA_ARCHS)
  set(multiValueArgs SRCS)
  cmake_parse_arguments(MARLIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT MARLIN_SRCS)
    return()
  endif()

  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_SRCS}"
    CUDA_ARCHS "${MARLIN_CUDA_ARCHS}")
  if (MARLIN_WITH_STATIC_TEMPLATE_STUB_WORKAROUND AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_SRCS}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
endfunction()

# Hopper WGMMA/TMA PTX (wgmma.*, cp.async.bulk.tensor) requires sm_90a, not sm_90.
set(CUDA_ARCHS "9.0a")
set(VLLM_GPU_ARCHES "90a")
set(CUDA_ARCHS_STR "9.0a")
message(STATUS "marlin_v100 CUDA target architectures: ${CUDA_ARCHS} (sm_90a for WGMMA)")
message(STATUS "marlin_v100 CMake CUDA architectures: ${VLLM_GPU_ARCHES}")

get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(FILTER VLLM_GPU_FLAGS EXCLUDE REGEX "^-DNDEBUG$")
  list(APPEND VLLM_GPU_FLAGS "-lineinfo" "-O0")
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0")
  string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo -O0")
endif()
if(NVCC_THREADS)
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
endif()

set(MARLIN_V100_DENSE_SRC
  "csrc/torch_bindings_marlin.cpp")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_V100_DENSE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set(MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
marlin_generate_kernels(
  ${MARLIN_GEN_SCRIPT}
  marlin_generation.log
  "marlin_v100 dense kernel generation failed")

file(GLOB MARLIN_TEMPLATE_SM80_KERNEL_SRC CONFIGURE_DEPENDS
  "csrc/quantization/marlin/sm80_kernel_*.cu")
file(GLOB MARLIN_TEMPLATE_SM90_KERNEL_SRC CONFIGURE_DEPENDS
  "csrc/quantization/marlin/sm90_kernel_*.cu")
set(MARLIN_TEMPLATE_KERNEL_SRC
  ${MARLIN_TEMPLATE_SM80_KERNEL_SRC}
  ${MARLIN_TEMPLATE_SM90_KERNEL_SRC})
marlin_configure_cuda_sources(
  SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

set(MARLIN_SRCS
   "csrc/quantization/marlin/marlin.cu"
   "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
   "csrc/quantization/marlin/gptq_marlin_repack.cu"
   "csrc/quantization/marlin/awq_marlin_repack.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_SRCS}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_DENSE_SRC "${MARLIN_SRCS}")

set(MARLIN_V100_MOE_SRC
  "csrc/moe/torch_bindings_marlin.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu"
  "csrc/moe/grouped_topk_kernels.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_V100_MOE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set(MOE_MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
marlin_generate_kernels(
  ${MOE_MARLIN_GEN_SCRIPT}
  moe_marlin_generation.log
  "marlin_v100 moe kernel generation failed")

file(GLOB MARLIN_MOE_SM80_SRC CONFIGURE_DEPENDS
  "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
file(GLOB MARLIN_MOE_SM90_SRC CONFIGURE_DEPENDS
  "csrc/moe/marlin_moe_wna16/sm90_kernel_*.cu")
set(MARLIN_MOE_KERNEL_SRC
  ${MARLIN_MOE_SM80_SRC}
  ${MARLIN_MOE_SM90_SRC})
marlin_configure_cuda_sources(
  SRCS "${MARLIN_MOE_KERNEL_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_MOE_SRC ${MARLIN_MOE_KERNEL_SRC})

set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_MOE_OTHER_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_MOE_SRC "${MARLIN_MOE_OTHER_SRC}")

# Optional CUTLASS (example-69 grouped mixed-dtype GEMM experimental path).
if (DEFINED ENV{CUTLASS_DIR})
  set(MARLIN_CUTLASS_DIR "$ENV{CUTLASS_DIR}")
elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include/cutlass/cutlass.h")
  set(MARLIN_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
endif()

set(MARLIN_MOE_CUTLASS69_SRC
  "csrc/moe/marlin_moe_wna16/marlin_moe_cutlass69.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_MOE_CUTLASS69_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_MOE_SRC "${MARLIN_MOE_CUTLASS69_SRC}")

set(MOE_CUTLASS_INCLUDE_DIRS "")
set(MOE_CUTLASS_COMPILE_FLAGS "")
if (MARLIN_CUTLASS_DIR)
  message(STATUS "marlin_v100 CUTLASS_DIR: ${MARLIN_CUTLASS_DIR}")
  list(APPEND MOE_CUTLASS_INCLUDE_DIRS
    "${MARLIN_CUTLASS_DIR}/include"
    "${MARLIN_CUTLASS_DIR}/tools/util/include")
  list(APPEND MOE_CUTLASS_COMPILE_FLAGS "-DMARLIN_MOE_HAS_CUTLASS")
else()
  message(STATUS "marlin_v100 CUTLASS_DIR not set; CUTLASS69 path will stub out at runtime")
endif()

define_extension_target(
  _C
  DESTINATION marlin_v100
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${MARLIN_V100_DENSE_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

define_extension_target(
  _moe_C
  DESTINATION marlin_v100
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${MARLIN_V100_MOE_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS} ${MOE_CUTLASS_COMPILE_FLAGS}
  INCLUDE_DIRECTORIES ${MOE_CUTLASS_INCLUDE_DIRS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
