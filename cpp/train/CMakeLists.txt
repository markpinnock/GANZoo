# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("train")

# Build options
option (BUILD_SHARED_LIBS "Build shared libraries" OFF)
option (BUILD_TESTING "Build unit tests" OFF)

# Include directories
set (TF_INCLUDE_DIR "include")
set (TF_LIB_DIR ${CMAKE_BINARY_DIR})
set (PB_INCLUDE_DIR "protobuf-3.9.2/src")
set (PB_LIB "protobuf-3.9.2/cmake/build/solution/Release")
include_directories (${TF_INCLUDE_DIR} ${PB_INCLUDE_DIR})

# Include sub-projects.
add_subdirectory ("src")

# Add testing if necessary
if (BUILD_TESTING)
	add_subdirectory ("tests")
endif()