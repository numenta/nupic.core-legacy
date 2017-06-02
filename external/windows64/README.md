# Introduction

The NuPIC Core C++ library can be built with MinGWPy GCC compilers or Microsoft Visual C++ compilers. The Python nupic.bindings can **ONLY** be built using the MinGWPy GCC compiler.

> **Note:** If you intend to _only_ use the C++ library it can be easier to use the MS Visual Studio 2015 Integrated Development Environment (IDE) as described below.

> **Note:** To use the C++ library with [NuPIC Python](https://github.com/numenta/nupic) you must build the library with the MinGWPy GCC tools. See the `external\windows64-gcc\README.md` text file for more details.

Refer to nupic.core/external/README.md for an overview of NuPIC Core library build and dependencies.

[CMake](http://www.cmake.org) based build files are used to define the entire build process. The [CMake-GUI](http://www.cmake.org/) application _can_ be used to generate _Visual Studio 14 2015 Win64_ solution and project files.

## Support applications

The following applications are required when rebuilding the core C++ library, and if required the external support libraries;

- [CMake](http://www.cmake.org/) - version 3.1+
- [Python 2.7.9+](https://www.python.org/downloads/windows/) - x86-64 version

For re-building the nupic.core C++ library the following are required;

- [Microsoft Visual Studio 2015](https://www.visualstudio.com/en-us/downloads/visual-studio-2015-downloads-vs) - Community Free edition (or Enterprise edition)
- [NumPy C++ headers](https://pypi.python.org/pypi/numpy/) - pip install numpy==1.12.1

## Rebuilding with CMake

The following table shows example CMake common settings;

<center>

| Name | Value |
|:---- |:----- |
| Source code | `%NUPIC_CORE%` |
| Binaries | `%NUPIC_CORE%/build/scripts` |
| CMAKE_INSTALL_PREFIX | `%NUPIC_CORE%/build/release` |
| PY_EXTENSIONS_DIR | `%NUPIC_CORE%/bindings/py/src/nupic/bindings` |

</center>

### NuPIC C++ Core library only (via MSVC)

To produce a solution and project files for use with Microsoft Visual Studio, you must use the `Visual Studio 14 2015 Win64` cmake generator.

```
rem Clone the repo
git clone https://github.com/numenta/nupic.core.git

rem Setup nupic.core and a place to store build files
cd nupic.core
set NUPIC_CORE=%CD%
mkdir build\scripts
cd build\scripts

rem Run cmake to generator MSVC 2015 solution and project files
cmake -G "Visual Studio 14 2015 Win64"
	-DCMAKE_INSTALL_PREFIX=..\release
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\src\nupic\bindings
	..\..
```

The resulting .sln file can then be loaded into MSVC 2015. This will _only_ allow building of the C++ libraries. It will _never_ be able to build the Python nupic.bindings, for that you must use the MinGWPy GCC compilers. See the `external\windows64-gcc\README.md` text file for details.

Make sure to select the `Release` `x64` configuration when building the `ALL_BUILD` project.

The `INSTALL` project has to be built after building the main library. External support libraries (x64 release) are stored in the Git repository. The `INSTALL` project copies the binaries, headers, and library files into the `CMAKE_INSTALL_PREFIX` directory.

After building the C++ library a manual install can be performed using the following command line;

```
cd %NUPIC_CORE%\build\scripts
cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake
```

## Run C++ tests

After building and installing, the following can run C++ related tests;

```
cd %NUPIC_CORE%\build\release\bin
cpp_region_test
unit_tests
```

## Build notes

The `%NUPIC_CORE%\.gitignore` file has a rule that ignores any directory called `build/` from Git. Making that directory a handy place to store build dependencies.

* The `ZERO_CHECK` project can be ignored. It is only used by CMake during the build process.
* Any changes made to project configurations from within Visual Studio must be carried across to the `src\CMakeLists.txt` file to become permanent changes.
* Any changes made to `src\CMakeLists.txt` requires a _(re)_-run of CMake / CMake-GUI to configure and generate new Visual Studio project files with those changes. Make sure that Visual Studio is not running when this step occurs.
* The solution file (v140 platform toolset) has an `ALL_BUILD` project that can rebuild the x64 Release version of the core library and test programs.
* The `INSTALL` project has to be built after building the main library. External libraries (x64 release) are stored in the Git repository, and packaged into the deployed version. The `INSTALL` project copies the binaries, headers, and library files into the `CMAKE_INSTALL_PREFIX` directory.
* The `PACKAGE` project implements the `src\CmakeLists.txt` file CPack packaging instructions. It takes the post-INSTALL release files and makes a tape archive file (tar format), that is then compressed (gzip format) into a single file.

##### NumPy

The C++ headers for NumPy can be installed via a pre-built NumPy Python package. For example; from http://www.lfd.uci.edu/~gohlke/pythonlibs/ Make sure to use a `cp27` `win_amd64` version.

##### Apache Portable Runtime (APR)

apr.dsw and aprutil.dsw workspace files can be imported into Visual Studio 2015. Directory naming allows these solutions to find the apr-iconv projects. With APR we only require the 'apr' project to be built. The 'libapr' project is for a DLL version. And with APR-UTIL we just need to build the 'apr-util' project.

##### Cap'n Proto

Download version **0.5.3** from https://capnproto.org/capnproto-c++-win32-0.5.3.zip and extract into %NUPIC_CORE%\build directory.

Install instructions can be found at https://capnproto.org/install.html Below is an example Visual Studio Command Prompt instructions to invoke cmake and generator a solution and project files for Cap'n Proto.

```
cd %NUPIC_CORE%\build\capnproto-c++-win32-0.5.3\capnproto-c++-0.5.3
vcvarsall.bat
cmake -G "Visual Studio 14 2015 Win64"
	-DCAPNP_LITE=1
	-DEXTERNAL_CAPNP=1
	-DCAPNP_INCLUDE_DIRS=.\src
	-DCAPNP_LIB_KJ=.\lib
	-DCAPNP_LIB_CAPNP=.\lib
	-DCAPNP_EXECUTABLE="..\capnproto-tools-win32-0.5.3\capnp.exe"
	-DCAPNPC_CXX_EXECUTABLE="..\capnproto-tools-win32-0.5.3\capnpc-c++.exe"
```

Building the test programs may halt a full build. But enough will be built for an Install, and finally copy of the new capnp.lib and kj.lib libraries.

##### Yaml

A valid libyaml.sln solution file can be found in directory `yaml-0.1.5\win32\vs2008` A new x64 platform solution can be added to it once imported into Visual Studio 2015. We only need to build the yaml project from this solution.

##### Yaml-cpp

In CMake-GUI, `%NUPIC_CORE%/build/yaml-cpp/` can be used for Source and Build directories. Make sure that MSVC_SHARED_RT is **ticked**, and BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not ticked**. When building the solution you may need to `#include <algorithm>` in `src\ostream_wrapper.cpp`

##### Z Lib

In `zlib-1.2.8\contrib\vstudio` there are solutions for Visual Studio 9, 10, and 11. The vc11 solution can be used with Visual Studio 2015. A x64 platform solution needs to be added to this imported solution. The zlibstat is the library we need to to build, and copy the rebuilt library over the z.lib file in directory `%NUPIC_CORE%/external/windows64/lib`
