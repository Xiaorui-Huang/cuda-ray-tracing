cmake_minimum_required(VERSION 3.11)
project(raytracing LANGUAGES CUDA CXX)

# Verbose Makefile for debugging
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Add the include directory
include_directories("include")

# Source directory
set(SRC_DIR "src")
file(GLOB SRCFILES "${SRC_DIR}/*.cu")

list(APPEND SRCFILES main.cu)
list(APPEND SRCFILES ${SRC_CPP_FILES})

# Setting the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for Release and Debug build types
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# CUDA compiler flags for Ampere architecture
set(ARCH_CODE 86)
set(CMAKE_CUDA_ARCHITECTURES ${ARCH_CODE})

# enable link time optimization (LTO) for release builds
# https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/
set(CMAKE_CUDA_FLAGS_RELEASE "-gencode arch=compute_${ARCH_CODE},code=lto_${ARCH_CODE}") 

# Debug build flags
set(CMAKE_CUDA_FLAGS_DEBUG "--debug --device-debug") # '-G' for CUDA disables optimizations and '-g' adds debug information

# enable dynamic parallelism
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt")

# Create the executable
add_executable(${PROJECT_NAME} ${SRCFILES})

# Enable CUDA separate compilation
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Use FetchContent to add the nlohmann/json library
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Link the nlohmann_json library with your project
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

