#find_package (GTest REQUIRED)
#enable_testing ()
set (GTEST_INCLUDE_DIR "external/googletest/googletest/include/")
set (GTEST_LIB_DIR "external/googletest/build/lib/RelWithDebInfo/")

include_directories (${GTEST_INCLUDE_DIR})

# Add executable for unit tests
add_executable (test_loaders test_loaders.cpp test_dataloader.cpp)
#set_property (TARGET test_loaders PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")

# Add libraries for tests
find_library (GTEST NAMES gtest gtest_maind PATHS ${GTEST_LIB_DIR})
target_link_libraries (test_loaders utils ${GTEST} ${TF} ${PROTOBUF})

#add_test (test_loaders test_loaders)