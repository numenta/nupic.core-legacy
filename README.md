<img src="http://numenta.org/87b23beb8a4b7dea7d88099bfb28d182.svg" alt="NuPIC Logo" width=100/>

# NuPIC C++ Core Library

[![Linux/OSX Build Status](https://travis-ci.org/htm-community/htm.cpp.svg?branch=master)](https://travis-ci.org/htm-community/htm.cpp)
[![OSX CircleCI](https://circleci.com/gh/htm-community/htm.cpp/tree/master.svg?style=svg)](https://circleci.com/gh/htm-community/htm.cpp/tree/master)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/59f87and1x0ugss9/branch/master?svg=true)](https://ci.appveyor.com/project/htm-community/htm-cpp/branch/master)


## Community NuPIC.cpp (former htm.core) repository

This fork is a community version of the [htm.core](https://github.com/numenta/htm.core) C++ repository with Python bindings.
Our aim is to provide an actively developed successor to the htm.core and htm repositories by Numenta,
which are not actively developed anymore.

### Our goals

- [Goals for the next release](https://github.com/htm-community/htm.cpp/blob/master/RELEASE_V1_GOALS.md)
- Actively developed C++ core library for HTM/htm.core (Numenta's repos are in maintanance mode only)
- Clean & lean, fast, modern codebase (dependency removal, c++11/17, modernized code, faster)
- Stable and well tested code
- API-compatibility with Numenta's code *)
- Open and easier involvement of new ideas across HTM community (it's fun to contribute, we make master run stable, but are more open to experiments and larger revamps of the code if it proves useful).
- Cross Platform Support
- [Modularity](bindings/py/README.md) through bindings to the core library
  - Currently only python has bindings, located in `bindings/py`


This repository contains the C++ source code for the Numenta Platform for
Intelligent Computing ([NuPIC](http://numenta.org/htm.html)).
It will eventually contain all algorithms for NuPIC, but is currently in a transition period.

\*) Nupic API compatability: The objective is to stay as close as possible to the [Nupic API Docs](http://htm.docs.numenta.org/stable/api/index.html)
with the aim that we don't break `.py` code written against the numenta's htm.core extension library if they were to be
ran against this extention library. If you are porting your code to this codebase, please review [API Changelog](API_CHANGELOG.md).

### New Features

Some of the major differences between this library and Numenta's extension library are the following:

 * Support for Python 3 and Python 2.7 (Only Python 3 under windows)
 * Support for Linux, OSx, and Windows MS Visual Studio 2017
 * Support for C++11 through C++17
 * Replaced SWIG with PyBind11 for Python interface.
 * Removed CapnProto serialization.  It was prevasive and complicated the code considerably. It was replaced
 with simple binary streaming serialization in C++ library.
 * Many code optimizations, modernization (Spatial Pooler shares optimized Connections backend with Temporal memory)
 * Modular structure
 * Interfaces & API stabilization, making it easier for developers & researchers to use our codebase
 * Much easier installation (reduced dependencies, all are handeled by CMake)
 * Static and shared lib files for use with C++ applications.
 * New and Improved Algorithms:
   - Sparse Distributed Representations
   - Anomaly Likelihood
   - Backtracking Temporal Memory
   - Significantly faster Spatial Pooler and Connections

## Installation

### Prerequisites

- [CMake](http://www.cmake.org/)
- [Python](https://python.org/downloads/)
    - Version 2.7  We recommend you use the latest 2.7 version where possible. But the system version should be fine. (The extension library for Python 2.7 not supported on Windows.)
    - Version 3.4+  The Nupic Python repository will need to be upgraded as well before this will be useful.
  Be sure that your Python executable is in the Path environment variable. The Python that is in your default path is the one
  that will determine which version of Python the extension library will be built for.
  NOTE: Anaconda Python not supported.
  Other implementations of Python may not work.
  Only the standard python from python.org have been tested.
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

### Building from Source

Fork or download the HTM-Community Nupic.cpp repository from https://github.com/htm-community/htm.cpp

#### Simple Build for Python users (any platform)

The easiest way to build from source is as follows.
```
    cd to-repository-root
    python setup.py install --user --force
```
Note that `--force` option will overwrite any existing files even if they are
the same version, which is useful when developing the library & bindings.

Note that `--user` option will install the extension libaries in ~/.local so
that you don't need superuser permissions.

This will build everything including the htm.core static library and Python extension libraries and then install them.

After that completes you are all set to run your .py programs which import the extensions:
 * htm.bindings.algorithms
 * htm.bindings.engine_internal
 * htm.bindings.math
 * htm.bindings.encoders
 * htm.bindings.sdr

The installation scripts will automatically download and build the dependancies it needs.
 * Boost   (Not needed by C++17 compilers that support the filesystem module)
 * Yaml-cpp
 * Eigen
 * PyBind11
 * gtest
 * cereal
 * mnist test data
 * numpy
 * pytest

 If you are installing on an air-gap computer (one without internet) then you can
 manually download the dependancies.  On another computer, download the distribution
 packages as listed and rename them as indicated. Copy these to
  `${REPOSITORY_DIR}/build/ThirdParty/share` on the target machine.

 | Name to give it | Where to obtain it |
 | :-------------- | :----------------- |
 | yaml-cpp.zip  *note1 | https://github.com/jbeder/yaml-cpp/archive/master.zip |
 | boost.tar.gz  *note2 | https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz |
 | eigen.tar.bz2        | http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2 |
 | googletest.tar.gz    | https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz |
 | mnist.zip     *note3 | https://github.com/wichtounet/mnist/archive/master.zip |
 | pybind11.tar.gz      | https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz |
 | cereal.tar.gz        | https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz |

 *note1: Version 0.6.2 of yaml-cpp is broken so use the master from the repository.
 *note2: Boost is not required for Windows (MSVC 2017) or any compiler that supports C++17 with std::filesystem.
 *note3: Data used for demo. Not required.

#### Simple Build On Linux or OSX for C++ apps

After downloading the repository, do the following:
```
	cd path-to-repository
	mkdir -p build/scripts
	cd build/scripts
	cmake ../..
	make install
```
This will build the Nupic.core library without the Python interface.
 * build/Release/lib/libhtm-core.a   static library
 * build/Release/lib/libhtm-core.so  shared library
 * The headers will be in `build/Release/include`.

A debug library can be created by adding `-DCMAKE_BUILD_TYPE=Debug` to the cmake command above.  The -j3 could be used
with the `make install` command to compile with multiple threads.

#### Simple Build On Windows (MS Visual Studio 2017)

After downloading the repository, do the following:
 * CD to top of repository.
 * Double click startupMSVC.bat  -- This will setup the build and create the solution file (.sln).
 * Double click build/scripts/htm.cpp.sln -- This starts up Visual Studio
 * Select `Release` or `Debug` as the Solution Configuration. Solution Platform must remain at x64.
 * Build everything.  -- This will build the C++ library.
 * In the solution explorer window, right Click on 'unit_tests' and select `Set as StartUp Project` so debugger will run unit tests.
 * If you also want the Python extension library; in a command prompt, cd to root of repository and run `python setup.py install --user --prefix=`.


### Docker Builds

#### Build for Docker x86_64

If you are on `x86_64` and would like to build a Docker image:

```sh
docker build --build-arg arch=x86_64 .
```

#### Docker build for ARM64

If you are on `ARM64` and would like to build a Docker image, run the command
below. The CircleCI automated ARM64 build (detailed below) uses this
specifically.

```sh
docker build --build-arg arch=arm64 .
```

### Automated Builds

#### Linux auto build @ TravisCI

 * [Build](https://travis-ci.org/htm-community/htm.cpp)
 * [Config](./.travis.yml)

#### Mac OS/X auto build @ CircleCI

 * [Build](https://circleci.com/gh/htm-community/htm.cpp/tree/master)
 * [Config](./.circleci/config.yml)
 * Local Test Build: `circleci local execute --job build-and-test`

#### Windows auto build @ AppVeyor

 * [Build](https://ci.appveyor.com/project/htm-community/htm-cpp/branch/master)
 * [Config](./appveyor.yml)

#### ARM64 auto build @ CircleCI

This uses Docker and QEMU to achieve an ARM64 build on CircleCI's x86 hardware.

 * **TODO!** [Build]()
 * [Config](./.circleci/config.yml)
 * Local Test Build: `circleci local execute --job arm64-build-test`


### Testing

#### Unit tests for the library

There are two sets of unit tests.
 * C++ Unit tests -- to run: `cd build/Release/bin; ./unit_tests`
 * Python Unit tests -- to run: `python setup.py test`

### Using graphical interface

#### Generate the IDE solution  (Netbeans, XCode, Eclipse, KDevelop, etc)

 * Choose the IDE that interest you (remember that IDE choice is limited to your OS).
 * Open CMake executable in the IDE.
 * Specify the source folder (`$HTM_CORE`) which is the location of the root CMakeList.exe.
 * Specify the build system folder (`$HTM_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Generate`.

#### For MS Visual Studio 2017 as the IDE
 * Double click startupMSVC.bat  -- This will setup the build and create the solution file (.sln).
 * Double click build/scripts/htm.cpp.sln -- This starts up Visual Studio
 * In the solution explorer window, right Click on 'unit_tests' and select `Set as StartUp Project` so debugger will run unit tests.
 * Start a debug session.

#### For Eclipse as the IDE
 * File - new C/C++Project - Empty or Existing CMake Project
 * Location: (`$HTM_CORE`) - Finish
 * Project properties - C/C++ Build - build command set "make -C build/scripts VERBOSE=1 install -j 6"
 * There can be issue with indexer and boost library, which can cause OS memory to overflow -> add exclude filter to
   your project properties - Resource Filters - Exclude all folders that matches boost, recursively
 * (Eclipse IDE for C/C++ Developers, 2019-03)

For all new work, tab settings are at 2 characters.
The clang-format is LLVM style.

### Examples

#### Hot Gym
A simple example of an app that calls SP and TM algorithms directly is 
the hot gym app.  This tries to predict the power consuption for a gym
which has fluctuations in temporature.  This is often used as a benchmark.

To run:  (assuming current directory is top of repository)
```
  ./build/Release/bin/benchmark_hotgym
```

There is also a dynamically linked version of Hot Gym (not available on MSVC). 
You will need specify the location
of the shared library with LD_LIBRARY_PATH.

To run: (assuming current directory is top of repository)
```
LD_LIBRARY_PATH=build/Release/lib ./build/Release/bin/dynamic_hotgym
```

#### NMIST benchmark

The task is to recognise images of hand written numbers 0-9.
This should score at least 95%.
This is often used as a benchmark.  Adjust the number of iterations in 
MNST_SP.cpp line 56 before compiling.

to run: (assuming current directory is top of repository)
```
  ./build/Release/bin/mnist_sp
```
