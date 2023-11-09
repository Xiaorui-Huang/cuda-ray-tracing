cmake_minimum_required(VERSION 3.11)
project(raytracing LANGUAGES CUDA CXX)

# Verbose Makefile for debugging
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Add the include directory
include_directories("include/")
# include_directories("includeCpp/") # Header for Geometry Object processing

# Source directory
set(SRC_DIR "src/")
file(GLOB SRCFILES "${SRC_DIR}/*.cu")

# temporary measures
# set(SRC_CPP_DIR "srcCpp/") # New source directory for C++ files
# file(GLOB SRC_CPP_FILES "${SRC_CPP_DIR}/*.cpp")

list(APPEND SRCFILES main.cu)
list(APPEND SRCFILES ${SRC_CPP_FILES})

# Setting the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for Release and Debug build types
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3") # High level of optimization
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g") # No optimization, include debug information
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g") # '-G' for CUDA disables optimizations and '-g' adds debug information


# CUDA compiler flags for Ampere architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Use FetchContent to add the nlohmann/json library
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Create the executable
add_executable(${PROJECT_NAME} ${SRCFILES})

# Link the nlohmann_json library with your project
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

