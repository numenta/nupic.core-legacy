# Introduction

The NuPIC Core C++ library can be built with MinGWPy GCC compilers or Microsoft Visual C++ compilers. The Python nupic.bindings can **ONLY** be built using the MinGWPy GCC compiler. 

> **Note:** If you intend to _only_ use the C++ library it can be easier to use the MS Visual Studio 2015 Integrated Development Environment (IDE) as described below.

> **Note:** To use the C++ library with [NuPIC Python](https://github.com/numenta/nupic) you must build the library with the MinGWPy GCC tools. See the `external\windows64-gcc\README.md` text file for more details.

NuPIC's dependency on the Core library can be found here - https://github.com/numenta/nupic/wiki/NuPIC's-Dependency-on-nupic.core

It describes how the C++ library is linked to other languages using SWIG, such as Python x64. This nupic.core repository consist of two parts;

- Main C++ core library, and
- Python SWIG based bindings. 

The C++ library file is built into two files;

- `%NUPIC_CORE%/release/lib/nupic_core_solo.lib` contains _only_ the nupic.core C++ library
- `%NUPIC_CORE%/release/lib/nupic_core.lib` contains the C++ nupic.core and external support libraries

Where `NUPIC_CORE` is an environment variable that points to the git cloned directory.

[CMake](http://www.cmake.org) based build files are used to define the entire build process. The [CMake-GUI](http://www.cmake.org/) application _can_ be used to generate _Visual Studio 14 2015 Win64_ solution and project files.

## Support applications

The following applications are required when rebuilding the core C++ library, and if required the external support libraries;

- [CMake](http://www.cmake.org/) - version 3.1+
- [Python 2.7.9+](https://www.python.org/downloads/windows/) - x86-64 version

For re-building the nupic.core C++ library the following are required;

- [Microsoft Visual Studio 2015](https://www.visualstudio.com/en-us/downloads/visual-studio-2015-downloads-vs) - Community Free edition (or Enterprise edition)
- [NumPy C++ headers](http://www.lfd.uci.edu/~gohlke/pythonlibs/) - pip install `numpy-1.9.3+vanilla-cp27-none-win_amd64.whl`

## Rebuilding with CMake

The following table shows example CMake common settings;

<center>

| Name | Value |
|:---- |:----- |
| Source code | `%NUPIC_CORE%` |
| Binaries | `%NUPIC_CORE%/build/scripts` |
| CMAKE_INSTALL_PREFIX | `%NUPIC_CORE%/build/release` |
| PY_EXTENSIONS_DIR | `%NUPIC_CORE%/bindings/py/nupic/bindings` |

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
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\nupic\bindings
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

## External libraries

The C++ Core library depends on a handful of support libraries that are distributed within the GitHub repository. The following sub-sections can be used as a guide if you require changes to these pre-built libraries, such as creating MSVC Debug versions.

The Microsoft Visual Studio _x64 Release_ support libraries can be found in `%NUPIC_CORE%\external\windows64\lib`  

### Obtaining the library sources

The following libraries are embedded into the `%NUPIC_CORE%/lib/nupic_core` library.

| Library | Version | Filename |  Website |
|:------- |:------- |:-------- | :------- |
| APR | **1.5.1** | apr-**1.5.1**-win32-src.zip | https://apr.apache.org/ |
| Apr Iconv | **1.2.1** | apr-iconv-**1.2.1**-win32-src-r2.zip | https://apr.apache.org/ |
| Apr Util | **1.5.4** | apr-util-**1.5.4**-win32-src.zip | https://apr.apache.org/ |
| CapnProto | **0.5.3** | capnproto-c++-win32-**0.5.3**.zip | https://capnproto.org |
| Yaml | **0.1.5** | yaml-**0.1.5**.tar.gz | http://pyyaml.org/wiki/LibYAML |
| Yaml Cpp | **0.3.0** | yaml-cpp-**0.3.0**.tar.gz | https://code.google.com/p/yaml-cpp/ |
| Z Lib | **1.2.8** | zlib-**1.2.8**.tar.gz | http://www.zlib.net/ |

Extract them all into `%NUPIC_CORE%/build` directory. You may need to un-nest some of the directories. 

Note that APR expects certain directory names. Rename the following directories;  

`apr-1.5.1` to `apr`  
`apr-iconv-1.2.1` to `apr-iconv`  
`apr-util-1.5.4` to `apr-util`  

### Building the external libraries

Most of the libraries have a CMakeList.txt file, and possibly older workspace (DSW) or solution (DSP) files to convert using Visual Studio 2015. A CMakeList.txt file allows the use of the **CMake-GUI** application. Which can make it easy to tweak build variables, e.g. the required update of the `CMAKE_INSTALL_PREFIX` for each library; and advanced options such as compiler settings. 

Remember to set the configuration to Debug or Release, and setup a clone configuration of Win32 for **x64**. All support libraries are build for x64. 

If a solution contains an INSTALL project, the install scripts are placed inside a file called `cmake_install.cmake` An INSTALL project tries to execute the following command;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your `PATH` environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

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
