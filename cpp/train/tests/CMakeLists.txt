# Testing CMake file

# Include and library directories
set (GTEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third-party/gtest/include")
set (GTEST_LIB_DIR "${CMAKE_SOURCE_DIR}/third-party/gtest/lib")
include_directories (${GTEST_INCLUDE_DIR})

# Add executable for unit tests
add_executable (tests "test_dataloader.cpp" "test_DCGAN.cpp")

# Add libraries for tests
find_library (GTEST NAMES gtest PATHS ${GTEST_LIB_DIR})
target_link_libraries (tests utils ${GTEST} ${TF} ${PROTOBUF})

# Add GoogleTest so that Visual Studio can add tests to Test Explorer
include (GoogleTest)
gtest_discover_tests (tests)