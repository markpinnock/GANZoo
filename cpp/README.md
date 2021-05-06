Install Python +/- Anaconda/Miniconda/venv
- add to PATH or use Anaconda cmd

Install Bazel
- (tested with v3.7.2)
- https://docs.bazel.build/versions/master/install.html

Install MSYS2
- https://www.msys2.org/
- Can add to PATH variable

May need to install MS Build Tools if not packaged with Visual Studio already

CUDA Toolkit (tested with v10.1)
- https://developer.nvidia.com/cuda-toolkit

cuDNN (tested with v7.6.5)
- https://developer.nvidia.com/cudnn


Do this
- pacman -S git patch unzip

Config
- python configure.py (adds configuration to .tf_configure.bazelrc)
- Will ask for Python exe and lib location
- Default
- E.g.: C:/Users/<username>/Anaconda3
- E.g.: C:/Users/<username>/Anaconda3/lib/site-packages
- ROCm support ?AMD
- CUDA support Y
- CUDA 10, cuDNN 7
- CUDA/cuDNN location (simplify by copying cuDNN headers and libs into CUDA)
- C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/,C:/Users/roybo/Cpp/cudnn-10.1-windows10-x64-v7.6.5.32/cuda/
- If more than one copy CUDA, check CUDA_PATH system variable - configure.py may search here first causing version conflict
- Compute capability (e.g. 6.1 for P5000/GTX1080) https://developer.nvidia.com/cuda-gpus
- Opt flags: default
- Override eigen strong line: Y
- Interactive workspace for Android dev: N

Compiling
- Build DLLs using Bazel in top-level Tensorflow directory
- --config uses flags specified in opt flags config, remove --config=cuda if not GPU
- --define=no_tensorflow_py_deps=true used if not building Python version
- //tensorflow:tensorflow_cc provides c++ library target

bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:tensorflow_cc

- Build DLL import library

bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:tensorflow_cc_dll_import_lib

- Generate Tensorflow headers

bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:install_headers

- Libs will be in tensorflow/bazel-bin/tensorflow
- tensorflow_cc.dll, tensorflow_cc.lib
- Headers will be in tensorflow/bazel-bin/tensorflow/include

Potential problems
- Error regarding missing io_bazel_rules_docker at start of build

Add to top of WORKSPACE:
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
    strip_prefix = "rules_docker-0.7.0",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"],
)
or do git apply WORKSPACE.patch in top-level Tensorflow directory

- If numpy not found, may need to run initial phase of build in virtual environment containing numpy

- If problems with MSVC recognising flags, Visual Studio may not contain MS Build Tools, so download

Install Protobuf (tested with v3.9.2)
- https://github.com/protocolbuffers/protobuf
- Can be build using CMake and Visual Studio: https://github.com/protocolbuffers/protobuf/blob/master/cmake/README.md
- -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
- May find that CMake needs running from Visual Studio command prompt: run vcvars64.bat in cmd)


My CMake set up assumes dependencies copied to:

Linker errors
- Not all symbols exported due to maximum number of symbols allowed in DLL
- Several ways of marking them for export
- The file def_file_filter.py.tpl marks symbols for export
- Either manually add them in or git apply def_file_filter.patch in top-level Tensorflow directory
- Add TF_EXPORT macro (effectively __declspec(dllexport)) to required symbols in the header files
- Other
- Whichever technique used, re-compiling is needed (both //tensorflow:tensorflow_cc and //tensorflow:tensorflow_cc_dll_import_lib) and new versions compied to binary directory

bazel-out/x64_windows-opt/bin/external/com_google_protobuf/src: warning: directory does not exist

GoogleTest
-Dgtest_force_shared_crt=ON