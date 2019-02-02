<img src="http://numenta.org/87b23beb8a4b7dea7d88099bfb28d182.svg" alt="NuPIC Logo" width=100/>

# NuPIC C++ Core Library
[![Linux/OSX Build Status](https://travis-ci.org/htm-community/nupic.cpp.svg?branch=master)](https://travis-ci.org/htm-community/nupic.cpp)  
[![OSX CircleCI](https://circleci.com/gh/htm-community/nupic.cpp/tree/master.svg?style=svg)](https://circleci.com/gh/htm-community/nupic.cpp/tree/master) 
[![Windows Build status](https://ci.appveyor.com/api/projects/status/59f87and1x0ugss9/branch/master?svg=true)](https://ci.appveyor.com/project/htm-community/nupic-cpp/branch/master)

## Community NuPIC.cpp (former nupic.core) repository

This fork is a community version of the [nupic.core](https://github.com/numenta/nupic.core) C++ repository with Python bindings. 
Our aim is to provide an actively developed successor to the nupic.core and nupic repositories by Numenta, 
which are not actively developed anymore. 


### Our goals

- actively developed C++ core library for HTM/nupic.core (Numenta's repos are in maintanance mode only)
- clean & lean, fast, modern codebase (dependency removal, c++11/17, modernized code, faster)
- API-compatibility with Numenta's code *)
- open and easier involvement of new ideas across HTM community (it's fun to contribute, we make master run stable, but are more open to experiments and larger revamps of the code if it proves useful), new features include: 
  - Anomaly Likelihood
  - BacktrackingTM
  - much faster Spatial pooler implementation (runs on Connections)
- stable and well tested code
- easier portability to new platforms (due to removal of custom code (ASM,..) and reliance of C++ standardized features) 
- [modularity](bindings/py/README.md) through bindings to the core library
  - ie. python bindings in bindings/py 


This repository contains the C++ source code for the Numenta Platform for 
Intelligent Computing ([NuPIC](http://numenta.org/nupic.html)). 
It will eventually contain all algorithms for NuPIC, but is currently in a transition period. 

\*) Nupic API compatability: The objective is to stay as close as possible to the [Nupic API Docs](http://nupic.docs.numenta.org/stable/api/index.html) 
with the aim that we don't break `.py` code written against the numenta's nupic.core extension library if they were to be 
ran against this extention library. If you are porting your code to this codebase, please review [API Changelog](API_CHANGELOG.md).

### New Features 

Some of the major differences between this library and Numenta's extension library are the following:

 * Support for Python 2.7 and Python 3.x (Only Python 3.x under windows)
 * Support for Linux, OSx, and Windows MS Visual Studio 2017
 * Support for C++11, C++17 
 * Replaced SWIG with PyBind11 for Python interface.
 * Removed CapnProto serialization.  It was prevasive and complicated the code considerably. It was replaced 
 with simple binary streaming serialization in C++ library.
 * Many code optimizations, modernization (Spatial Pooler shares optimized Connections backend with Temporal memory) 
 * Modular structure
 * Interfaces & API stabilization, making it easier for developers & researchers to use our codebase
 * Much easier installation (reduced dependencies, all are handeled by CMake) 


## Installation 

### Prerequisites

- [CMake](http://www.cmake.org/)
- [Python](https://python.org/downloads/)
    - Version 2.7  We recommend you use the latest 2.7 version where possible. But the system version should be fine. (The extension library for Python 2.7 not supported on Windows.)
    - Version 3.4+  The Nupic Python repository will need to be upgraded as well before this will be useful.
  Be sure that your Python executable is in the Path environment variable. The Python that is in your default path is the one
  that will determine which version of Python the extension library will be built for.
- Python tools: In a command prompt execute the following.
```
  cd to-repository-root
  python -m pip install --user --upgrade pip setuptools setuptools-scm
  python -m pip install --no-cache-dir --user -r bindings/py/packaging/requirements.txt
```

  Be sure you are running the right version of python. Check it with the following command:
```
  python --version
```

### Building from Source

Fork or download the HTM-Community Nupic.cpp repository from https://github.com/htm-community/nupic.cpp

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

This will build everything including the nupic.cpp static library and Python extension libraries and then install them.

After that completes you are all set to run your .py programs which import the extensions:
 * nupic.bindings.algorithms
 * nupic.bindings.engine_internal
 * nupic.bindings.math
 
The installation scripts will automatically download and build the dependancies it needs.
 * Boost   (Not needed by C++17 compilers that support the filesystem module)
 * Yaml-cpp
 * Eigen
 * PyBind11
 * gtest
 * numpy
 * pytest
 
#### Simple Build On Linux or OSX for C++ apps
 
After downloading the repository, do the following:
```
	cd path-to-repository
	mkdir -p build/scripts
	cd build/scripts
	cmake ../..
	make install
```	
This will build the Nupic.core library without the Python interface. You will find the
library in `build/Release/lib`. The headers will be in `build/Release/include`.

A debug library can be created by adding `-DCMAKE_BUILD_TYPE=Debug` to the cmake command above.  The -j3 could be used 
with the `make install` command to use multiple threads.

#### Simple Build On Windows (MS Visual Studio 2017) 

After downloading the repository, do the following:
 * CD to top of repository.
 * Double click startupMSVC.bat  -- This will setup the build and create the solution file (.sln).
 * Double click build/scripts/nupic.cpp.sln -- This starts up Visual Studio
 * Select `Release` or `Debug` as the Solution Configuration. Solution Platform must remain at x64.
 * Build everything.  -- This will build the C++ library.
 * In the solution explorer window, right Click on 'unit_tests' and select `Set as StartUp Project` so debugger will run unit tests.
 * If you also want the Python extension library; in a command prompt, cd to root of repository and run `python setup.py install --user --prefix=`.

### Testing

#### Unit tests for the library

There are two sets of unit tests.
 * C++ Unit tests -- to run: `cd build/Release/bin; ./unit_tests`
 * Python Unit tests -- to run: `python setup.py test`
 
### Using graphical interface

#### Generate the IDE solution  (Netbeans, XCode, Eclipse, KDevelop, etc)

 * Choose the IDE that interest you (remember that IDE choice is limited to your OS).
 * Open CMake executable in the IDE.
 * Specify the source folder (`$NUPIC_CORE`) which is the location of the root CMakeList.exe.
 * Specify the build system folder (`$NUPIC_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Generate`.
 
#### For MS Visual Studio 2017 as the IDE
 * Double click startupMSVC.bat  -- This will setup the build and create the solution file (.sln).
 * Double click build/scripts/nupic.cpp.sln -- This starts up Visual Studio
 * In the solution explorer window, right Click on 'unit_tests' and select `Set as StartUp Project` so debugger will run unit tests.
 * Start a debug session.

For all new work, tab settings are at 2 characters.
The clang-format is LLVM style.
