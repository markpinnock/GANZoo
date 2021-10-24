# Tensorflow C++ usage instructions:
First, build Tensorflow C++ libraires [using the guide here](./TF_INSTALL_GUIDE.md)

# Install Protobuf
You'll need to install Protobuf (this guide is tested with v3.9.2).
1. Clone from <https://github.com/protocolbuffers/protobuf>
2. You can build using CMake and Visual Studio using this guide: <https://github.com/protocolbuffers/protobuf/blob/master/cmake/README.md>
3. To avoid clashes with the C++ runtime libraries, set the compiler flag `-Dprotobuf_MSVC_STATIC_RUNTIME=OFF`
4. You may find that CMake needs running from the Visual Studio command prompt: (find and run `vcvars64.bat` in cmd)

You may get this error `bazel-out/x64_windows-opt/bin/external/com_google_protobuf/src: warning: directory does not exist`, but I can't remember why.

# Linker errors
There is a high chance that you will get linker errors after compiling your Tensorflow project. Not all symbols are exported due to there being a maximum number of symbols allowed in DLLs in Windows. The file `.\tensorflow\tensorflow\tools\def_file_filter\def_file_filter.py.tpl` specifies which symbols to export. There are several ways of marking them for export:
- Manually add them in `def_file_filter.py.tpl` if there aren't many (make sure to use 4 spaces rather than tab)
- Copy [def_file_filter.patch](./def_file_filter.patch) into top-level Tensorflow directory `.\tensorflow` and do `git apply def_file_filter.patch`, although many of these might not be necessary for your application
- Find the missing symbols in the header files and add TF_EXPORT macro (this is analogous to __declspec(dllexport))
Whichever technique used, re-compiling is needed (both `//tensorflow:tensorflow_cc` and `//tensorflow:tensorflow_cc_dll_import_lib`)

# GoogleTest errors
If using GoogleTest, use the compiler flag `-Dgtest_force_shared_crt=ON` to ensure no clashes with runtime libraries

# Using the SavedModel CLI
- `bazel build tensorflow/python/tools:saved_model_cli`
- tensorflow/bazel-bin/tensorflow/python/tools
- https://www.tensorflow.org/guide/saved_model
- or, found in Scripts in a Python installation of Tensorflow 