cmake_minimum_required(VERSION 3.8)
project(raytracing LANGUAGES CXX CUDA)

# Verbose Makefile for debugging
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Add the include directory
include_directories("include/")
include_directories("includeCpp/") # Header for Geometry Object processing

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

# Create the executable
add_executable(${PROJECT_NAME} ${SRCFILES})

# Add JSON for Modern C++ include directory
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/json)




# cmake_minimum_required(VERSION 3.8)
# project(raytracing LANGUAGES CXX CUDA)  # Add CUDA here
# set(ROOT "${CMAKE_CURRENT_LIST_DIR}")

# if(UNIX)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
# endif()

# include_directories("include/")
# set(SRC_DIR "src/")

# file(GLOB SRCFILES "${SRC_DIR}/*.cu")  # Assuming you'll have .cu files
# set(HW2FILES 
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/Plane.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/Sphere.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/Triangle.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/TriangleSoup.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/first_hit.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/viewing_ray.cu"
#   "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_DIR}/write_ppm.cu")
# list(REMOVE_ITEM SRCFILES ${HW2FILES})
# list(APPEND SRCFILES main.cpp)

# # Compiler flags for C++
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(COMPILER_SUPPORTS_CXX0X)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# else()
#     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

# # Compiler flags for CUDA (modify based on your requirements)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# if(HW2LIB_DIR)
#   link_directories(${HW2LIB_DIR})
# endif()

# add_executable(${PROJECT_NAME} ${SRCFILES} ${LIBIGL_EXTRA_SOURCES})
# target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${ROOT}/eigen/ ${ROOT}/json)

# if(NOT HW2LIB_DIR)
#   add_library(hw2 ${HW2FILES})
#   target_include_directories(hw2 SYSTEM PUBLIC ${ROOT}/eigen ${ROOT}/json)
# endif()
# target_link_libraries(${PROJECT_NAME} hw2)





