﻿# Source CMake file

# Add source to this project's executable
add_executable (train "train.cpp")
add_executable(inference "inference_easy.cpp")
add_library (json "json.cpp")
add_library (utils "dataloader.cpp" "DCGAN.cpp" "utils.cpp" "loaders.cpp")

# Add libraries
find_library (TF NAMES tensorflow_cc PATHS "${TF_LIB_DIR}/src")
find_library (PROTOBUF NAMES libprotobuf PATHS ${PB_LIB_DIR})

target_link_libraries (train json utils ${TF} ${PROTOBUF})
target_link_libraries (inference json utils ${TF} ${PROTOBUF})

# if (BUILD_SHARED_LIBS)
