# Installing Tensorflow C++ on Windows 10
To do this, you'll need to build Tensorflow from source. Instructions are available here (<https://www.tensorflow.org/install/source_windows>) but these assume you are building a _pip_ package for use with Python so we'll need to skip some parts.
This guide is tested with:
- Tensorflow 2.3
- MS Visual Studio 2019
- Python 3.7.x
- Anaconda
- Bazel 3.7.2
- MSYS2
- MSVC 2019
- CUDA Toolkit 10.1
- cuDNN 7.6.5

There are some potential problems and pitfalls described at the bottom.

## Install Python and Conda
1. First, you will need Python (tested with 3.7.9) and a virtual environment (e.g. Anaconda, Miniconda or venv/virtualenv)
2. Either add Python to the PATH variable or use Anaconda cmd

## Install Bazel
Bazel is Google's build tool. The following instructions have been tested with v3.7.2, which can be installed from: <https://docs.bazel.build/versions/master/install.html>.

## Install MSYS2
Install MSYS2: this provides tools for building software available through the _pacman_ package manager
1. Download and install from <https://www.msys2.org>
2. Can add to PATH variable
3. Run `pacman -S git patch unzip`

# MS Build Tools
If MS Build Tools aren't packaged with Visual Studio already, these will need installing, otherwise you'll get MSVC compiler errors later on.

# CUDA
CUDA provides the necessary DLLs to run models on the GPU. Compatible versions can be found at https://www.tensorflow.org/install/source_windows#gpu. If running models on CPU only, you can skip this step. Otherwise:
1. Download CUDA Toolkit (tested with v10.1) from <https://developer.nvidia.com/cuda-toolkit>
2. cuDNN (7.6.5) provides CUDA DLLs specific for deep learning, downloadable from <https://developer.nvidia.com/cudnn>

# Clone the Tensorflow repository
Git is available with MSYS2 if you don't already have it. Tensorflow can be cloned from <https://github.com/tensorflow/tensorflow>. Choose the version needed with `git checkout r2.3`.

# Build configuration
This step sets the options for building Tensorflow and saves the options to `.\tensorflow\.tf_configure.bazelrc` in your cloned Tensorflow repo.
1. `python configure.py`
2. Will ask for Python exe and lib location (default options should work, e.g. `C:\Users\<username>\Anaconda3`, `C:\Users\<username>\Anaconda3\lib\site-packages`)
3. Say no (N) to ROCm support as this is for AMD GPUs
4. Say yes (Y) to CUDA support and state CUDA version 10 and cuDNN version 7 if using GPU
5. Give CUDA/cuDNN location: e.g. `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1,C:/Users/roybo/Cpp/cudnn-10.1-windows10-x64-v7.6.5.32/cuda` (note forward slashes). You can omit the second (cuDNN) directory if you've copied the headers and libs into the CUDA toolkit directory.
6. If you have more than one version of CUDA, check CUDA_PATH system variable as configure.py may search here first causing a version conflict
7. Enter GPU compute capability (e.g. 6.1 for P5000/GTX1080), found at <https://developer.nvidia.com/cuda-gpus>
8. Opt flags: choose default
9. Override eigen strong line: Y (probably)
10. Interactive workspace for Android dev: N

# Building
Build DLLs using Bazel in the top-level Tensorflow directory (`.\tensorflow` hereafter).
- --config uses flags specified in opt flags config
- --define=no_tensorflow_py_deps=true used as we are not building Python package
- `//tensorflow:tensorflow_cc` is the C++ library target
- Remove `--config=cuda` if building for CPU only

1. Build the C++ libraries: `bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:tensorflow_cc`
2. Build the DLL import library: `bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:tensorflow_cc_dll_import_lib`
3. Generate the Tensorflow header files: `bazel build --config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:install_headers`

The DLLs will be in `.\tensorflow/bazel-bin/tensorflow`: look for tensorflow_cc.dll and tensorflow_cc.lib. Headers will be in `.\tensorflow\bazel-bin\tensorflow\include`.

# Potential Problems
### Error regarding missing _io_bazel_rules_docker_ at start of build
Solution: Add the following to top of `.\tensorflow\WORKSPACE`:
`http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
    strip_prefix = "rules_docker-0.7.0",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"],
)`

Or copy [WORKSPACE.patch](./WORKSPACE.patch) into top-level Tensorflow directory `.\tensorflow` and do `git apply WORKSPACE.patch`.

### Numpy not found
If numpy not found during build, may need to run initial phase of build in the virtual environment containing numpy

### MSVC not recognising compiler flags
If there are problems with MSVC recognising compiler flags, Visual Studio may not contain MS Build Tools, so this needs to be installed.