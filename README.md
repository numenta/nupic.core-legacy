<img src="http://numenta.org/87b23beb8a4b7dea7d88099bfb28d182.svg" alt="NuPIC Logo" width=100/>

# htm.core

[![Linux/OSX Build Status](https://travis-ci.org/htm-community/htm.core.svg?branch=master)](https://travis-ci.org/htm-community/htm.core)
[![OSX CircleCI](https://circleci.com/gh/htm-community/htm.core/tree/master.svg?style=svg)](https://circleci.com/gh/htm-community/htm.core/tree/master)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/59f87and1x0ugss9/branch/master?svg=true)](https://ci.appveyor.com/project/htm-community/nupic-cpp/branch/master)

This is a Community Fork of the [nupic.core](https://github.com/numenta/nupic.core) C++ repository, with Python bindings.

## Project Goals

- Actively developed C++ core library (Numenta's NuPIC repos are in maintenance mode only)
- Clean, lean, optimized, and modern codebase
- Stable and well tested code
- Open and easier involvement of new ideas across HTM community (it's fun to contribute, we make master run stable, but are more open to experiments and larger revamps of the code if it proves useful).
- Interfaces to other programming languages, currently C++ and Python

## Features

 * Implemented in C++11 through C++17
    + Static and shared lib files for use with C++ applications.
 * Interfaces to Python 3 and Python 2.7 (Only Python 3 under Windows)
 * Cross Platform Support for Windows, Linux, OSX and ARM64
 * Easy installation.  Many fewer dependencies than nupic.core, all are handled by CMake
 * Significant speed optimizations
 * Simplified codebase
    + Removed CapnProto serialization.  It was pervasive and complicated the
code considerably. It was replaced  with simple binary streaming serialization
in C++ library.
    + Removed sparse matrix libraries, use the optimized Connections class instead
 * New and Improved Algorithms
    + Revamped all algorithms APIs, making it easier for developers & researchers to use our codebase
    + Sparse Distributed Representation class, integration, and tools for working with them
 * API-compatibility with Numenta's code.
   An objective is to stay close to the [Nupic API Docs](http://nupic.docs.numenta.org/stable/api/index.html).
   This is a priority for the `NetworkAPI`.
   The algorithms APIs on the other hand have deviated from their original API (but their logic is the same as Numenta's).
   If you are porting your code to this codebase, please review the [API Changelog](API_CHANGELOG.md).

# Installation

## Building from Source

Fork or download the HTM-Community htm.core repository from https://github.com/htm-community/htm.core

## Prerequisites

- [CMake](http://www.cmake.org/)  Version 3.8  (3.14 for Visual Studio 2019)
- [Python](https://python.org/downloads/)
    - Version 3.4+
    - Version 2.7
      + We recommend the latest version of 2.7 where possible, but the system version should be fine.
      + Python 2 is Not Supported on Windows, use Python 3 instead.

  Be sure that your Python executable is in the Path environment variable.
  The Python that is in your default path is the one that will determine which
  version of Python the extension library will be built for.
  - NOTE: People have reported success with `Anaconda` python.
  - Other implementations of Python may not work.
  - Only the standard python from python.org have been tested.
- Python tools: In a command prompt execute the following.
```
cd to-repository-root
python -m pip install --user --upgrade pip setuptools setuptools-scm wheel
python -m pip install --no-cache-dir --user -r bindings/py/packaging/requirements.txt
```

Be sure you are running the right version of python. Check it with the following command:
```
python --version
```

## Simple Build for Python users (any platform)

1) At a command prompt, go to the root directory of this repository.

2) Run: `python setup.py install --user --force`

   This will build and install everything.

   * Option `--user` will install the library in into your home directory so
   that you don't need administrator/superuser permissions.

   * Option `--force` will install the library even if the same version of it is
   already installed, which is useful when developing the library.

   * If you run into problems due to caching of arguments in CMake, delete the
   folder `Repository/build` and try again.  This is only an issue when
   developing C++ code.

3) After that completes you are ready to import the library:
```python
python.exe
>>> import htm           # Python Library
>>> import htm.bindings  # C++ Extensions
>>> help( htm )          # Documentation
```

## Simple Build On Linux or OSX for C++ apps

After downloading the repository, do the following:
```
cd path-to-repository
mkdir -p build/scripts
cd build/scripts
cmake ../..
make -j install
```

| Build Artifact | File Location |
| :------------- | :------------ |
| Static Library         | `build/Release/lib/libhtm-core.a`    |
| Shared Library         | `build/Release/lib/libhtm-core.so`   |
| Header Files           | `build/Release/include/`             |
| Unit Tests             | `build/Release/bin/unit_tests`       |
| Hotgym Dataset Example | `build/Release/bin/benchmark_hotgym` |
| MNIST Dataset Example  | `build/Release/bin/mnist_sp`         |

 * A debug library can be created by adding `-DCMAKE_BUILD_TYPE=Debug` to the cmake command above.
   + The debug library will be put in `build/Debug`.
     Use the cmake option `-DCMAKE_INSTALL_PREFIX=../Release` to correct this.

 * The -j option can be used with the `make install` command to compile with multiple threads.

 * This will not build the Python interface.

## Simple Build On Windows (MS Visual Studio 2017 or 2019)

After downloading the repository, do the following:

 * NOTE: Visual Studio 2019 requires CMake version 3.14 or higher.
 * CD to the top of repository.
 * Double click on startupMSVC.bat
    - This will setup the build, create the solution file (build/scripts/htm.cpp.sln), and start MS Visual Studio.
 * Select `Release` or `Debug` as the Solution Configuration. Solution Platform must remain at x64.
 * Build everything.  This will build the C++ library.
 * In the solution explorer window, right Click on 'unit_tests' and select `Set as StartUp Project` so debugger will run unit tests.
 * If you also want the Python extension library; in a command prompt, cd to root of repository and run `python setup.py install --user --prefix=`.

## Docker Builds

### Build for Docker x86_64

If you are on `x86_64` and would like to build a Docker image:

```sh
docker build --build-arg arch=x86_64 .
```

### Docker build for ARM64

If you are on `ARM64` and would like to build a Docker image, run the command
below. The CircleCI automated ARM64 build (detailed below) uses this
specifically.

```sh
docker build --build-arg arch=arm64 .
```

## Automated Builds

### Linux auto build @ TravisCI

 * [Build](https://travis-ci.org/htm-community/htm.core)
 * [Config](./.travis.yml)

### Mac OS/X auto build @ CircleCI

 * [Build](https://circleci.com/gh/htm-community/htm.core/tree/master)
 * [Config](./.circleci/config.yml)
 * Local Test Build: `circleci local execute --job build-and-test`

### Windows auto build @ AppVeyor

 * [Build](https://ci.appveyor.com/project/htm-community/nupic-cpp/branch/master)
 * [Config](./appveyor.yml)

### ARM64 auto build @ CircleCI

This uses Docker and QEMU to achieve an ARM64 build on CircleCI's x86 hardware.

 * [Build](https://circleci.com/gh/htm-community/htm.core/tree/master)
 * [Config](./.circleci/config.yml)
 * Local Test Build: `circleci local execute --job arm64-build-test`

# Using a Graphical Integrated Development Environment

## Generate the IDE solution  (Netbeans, XCode, Eclipse, KDevelop, etc)

 * Choose the IDE that interest you (remember that IDE choice is limited to your OS).
 * Open CMake executable in the IDE.
 * Specify the source folder (`$HTM_CORE`) which is the location of the root CMakeList.exe.
 * Specify the build system folder (`$HTM_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Generate`.

## [For MS Visual Studio 2017 as the IDE](#simple-build-on-windows-ms-visual-studio-2017)

## For Eclipse as the IDE
 * File - new C/C++Project - Empty or Existing CMake Project
 * Location: (`$HTM_CORE`) - Finish
 * Project properties - C/C++ Build - build command set "make -C build/scripts VERBOSE=1 install -j 6"
 * There can be issue with indexer and boost library, which can cause OS memory to overflow -> add exclude filter to
   your project properties - Resource Filters - Exclude all folders that matches boost, recursively
 * (Eclipse IDE for C/C++ Developers, 2019-03)

For all new work, tab settings are at 2 characters.
The clang-format is LLVM style.

# Third Party Dependencies

The installation scripts will automatically download and build the dependencies it needs.

 * Boost   (Not needed by C++17 compilers that support the filesystem module)
 * Yaml-cpp
 * Eigen
 * PyBind11
 * gtest
 * cereal
 * mnist test data
 * numpy
 * pytest

Once these third party components have been downloaded and built they will not be
re-visited again on subsequent builds.  So to refresh the third party components
or rebuild them, delete the folder `build/ThirdParty` and then re-build.

If you are installing on an air-gap computer (one without Internet) then you can
manually download the dependencies.  On another computer, download the
distribution packages as listed and rename them as indicated. Copy these to
`${REPOSITORY_DIR}/build/ThirdParty/share` on the target machine.

| Name to give it        | Where to obtain it |
| :--------------------- | :----------------- |
| yaml-cpp.zip  (*note1) | https://github.com/jbeder/yaml-cpp/archive/master.zip |
| boost.tar.gz  (*note2) | https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz |
| eigen.tar.bz2          | http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2 |
| googletest.tar.gz      | https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz |
| mnist.zip     (*note3) | https://github.com/wichtounet/mnist/archive/master.zip |
| pybind11.tar.gz        | https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz |
| cereal.tar.gz          | https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz |

 * note1: Version 0.6.2 of yaml-cpp is broken so use the master from the repository.
 * note2: Boost is not required for Windows (MSVC 2017) or any compiler that supports C++17 with std::filesystem.
 * note3: Data used for demo. Not required.

# Testing

### There are two sets of Unit Tests:

 * C++ Unit tests -- to run: `./build/Release/bin/unit_tests`
 * Python Unit tests -- to run: `python setup.py test`

# Examples

## Python Examples

There are a number of python examples, which are runnable from the command line.
They are located in the module `htm.examples`.

Example Command Line Invocation: `$ python -m htm.examples.sp.hello_sp`

## Hot Gym

This is a simple example application that calls the SpatialPooler and
TemporalMemory algorithms directly.  This attempts to predict the electrical
power consumption for a gymnasium over the course of several months.

To run python version:
```
python -m htm.examples.hotgym
```

To run C++ version: (assuming current directory is top of repository)
```
./build/Release/bin/benchmark_hotgym
```

There is also a dynamically linked version of Hot Gym (not available on MSVC). 
You will need specify the location of the shared library with LD_LIBRARY_PATH.

To run: (assuming current directory is top of repository)
```
LD_LIBRARY_PATH=build/Release/lib ./build/Release/bin/dynamic_hotgym
```

## MNIST benchmark

The task is to recognize images of hand written numbers 0-9.
This is often used as a benchmark.  This should score at least 95%.

To run: (assuming current directory is top of repository)
```
  ./build/Release/bin/mnist_sp
```
