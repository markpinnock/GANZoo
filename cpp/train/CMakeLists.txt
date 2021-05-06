# Top-level CMake project file

cmake_minimum_required (VERSION 3.10)

project ("train")

# Build options
option (BUILD_SHARED_LIBS "Build shared libraries" OFF)
option (BUILD_TESTING "Build unit tests" OFF)

# Include and library directories
set (TF_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third-party/tensorflow/include")
set (TF_LIB_DIR ${CMAKE_BINARY_DIR})
set (PB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third-party/protobuf/include")
set (PB_LIB_DIR "${CMAKE_SOURCE_DIR}/third-party/protobuf/lib")
include_directories (${TF_INCLUDE_DIR} ${PB_INCLUDE_DIR})

# Include sub-projects
add_subdirectory ("src")

# Add testing if necessary
if (BUILD_TESTING)
	enable_testing ()
	add_subdirectory ("tests")
endif()