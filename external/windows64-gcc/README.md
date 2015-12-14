# Introduction

> The NuPIC Core C++ library can be built with MinGW GCC and Microsoft (MS) Visual C compilers. The associated Python SWIG nupic.bindings can **ONLY** be built with the MinGW GCC compiler. If you intend to only use the C++ library it is easier to use the MS Visual Studio 2015 Integrated Development Environment (IDE).  

> **BUT** to use the C++ library with the [NuPIC Python](https://github.com/numenta/nupic) project you must build the library with the MinGW-Py GCC tools.

NuPIC's dependency on the Core library can be found here - https://github.com/numenta/nupic/wiki/NuPIC's-Dependency-on-nupic.core

It describes how the C++ library is linked to other languages using SWIG, such as Python x64. The NuPIC.Core repository consist of two parts; the main C++ core library, and the SWIG based bindings. The NuPIC Core C++ library file is split into two files;

- `%NUPIC_CORE%/release/lib/nupic_core_solo.lib` contains _only_ the core library
- `%NUPIC_CORE%/release/lib/nupic_core.lib` contains the core and external support libraries

Where `%NUPIC_CORE%` is an optional environment variable that points to the git cloned directory.

CMake based build files are used to define the entire build process. The nupic.bindings SWIG side uses Python distutil and setuptools. The [CMake-GUI](http://www.cmake.org/) application _can_ be used to generate MinGW Makefiles, and/or Visual Studio 2015 Win64 solution and project files. The `%NUPIC_CORE%\appveyor.yml` script file shows how the AppVeyor automated build system progresses through building, packaging, and deployment.

## Python wheel installation

The easiest way to use nupic.core with nupic is to install via Python pip. This can be done using the following command;

<pre>
set DOWNLOAD_LOCATION="https://s3-us-west-2.amazonaws.com/artifacts.numenta.org/numenta/nupic.core/releases"
pip install %DOWNLOAD_LOCATION%/nupic.bindings/nupic.bindings-0.2.2-py2-none-win_amd64.whl
</pre>

Just make sure that the latest nupic.bindings version is used. To find the latest version, point your browser to https://s3-us-west-2.amazonaws.com/artifacts.numenta.org/ and search the web page for `win_amd64.whl`. In the above example it is version `0.2.2`

Once it is installed, you can import NuPIC bindings library to your python script using:
<pre>
import nupic.bindings
</pre>

If you need to make changes to the nupic.core code follow the remaining sections to see how to rebuild using CMake.

## Support applications

The following applications are required when rebuilding the core C++ library, Python SWIG bindings, and if required the external support libraries;

- [CMake](http://www.cmake.org/) - version 3.1+
- [Python 2.7.9+](https://www.python.org/downloads/windows/) - x86-64 version

For re-building the core C++ library and Python SWIG bindings the following are required;

- [7-Zip](http://www.7-zip.org/) - Or any archive manager that can extract `.7z` files
- [MinGW GCC for Python](https://bitbucket.org/carlkl/mingw-w64-for-python/downloads) - `mingwpy_amd64_vc90.7z` and `libpython-cp27-none-win_amd64.7z`
- [NumPy C++ headers](http://www.lfd.uci.edu/~gohlke/pythonlibs/) - pip install `numpy-1.9.3+vanilla-cp27-none-win_amd64.whl` _or via_ MSys2 command `pacman -S mingw64/mingw-w64-x86_64-python2-numpy`
- [MSYS2](http://sourceforge.net/p/msys2/wiki/MSYS2%20introduction/) - 64 bit version. Only needed when _rebuilding_ external support libraries with MinGWPy GCC or installing alternative NumPy packages

If you __only__ require the C++ core library -

- [Microsoft Visual Studio 2015](https://www.visualstudio.com/en-us/downloads/visual-studio-2015-downloads-vs) - Community Free edition (or Enterprise edition)

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

### NuPIC C++ Core and Python SWIG bindings

For rebuilding ALL of NuPIC.core you need to use the `MinGW Makefiles` generator. For example;

<pre>
rem Clone the repo
git clone https://github.com/numenta/nupic.core.git
cd nupic.core

rem Setup Python x64 to use MinGWPy compilers
copy external\windows64-gcc\bin\distutils.cfg C:\Python27-x64\Lib\distutils

rem Run cmake to generator MinGW Makefiles
mkdir build\scripts
cd build\scripts
cmake -G "MinGW Makefiles"
	-DCMAKE_BUILD_TYPE=Debug
	-DCMAKE_INSTALL_PREFIX=..\release
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\nupic\bindings
	..\..

rem Build and install NuPIC.core, and build SWIG binding libraries
mingw32-make -f Makefile install

rem Install Python SWIG nupic.bindings
cd ..\..
python setup.py install
</pre>

You can run the nupic.bindings tests with py.test:
<pre>
cd %NUPIC_CORE%
py.test --pyargs nupic.bindings
</pre>

### NuPIC C++ Core only (via MSVC)

To produce a solution and project files for use with Microsoft Visual Studio, you must use the `Visual Studio 14 2015 Win64` cmake generator.
  
<pre>
rem Clone the repo
git clone https://github.com/numenta/nupic.core.git

rem Run cmake to generator MSVC 2015 solution and project files
cd nupic.core
mkdir build\scripts
cd build\scripts
cmake -G "Visual Studio 14 2015 Win64"
	-DCMAKE_INSTALL_PREFIX=..\release
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\nupic\bindings
	..\..
</pre>

The resulting .sln file can then be loaded into MSVC 2015. This will only build the C++ libraries. It will never be able to build the Python SWIG binding libraries, for that you must use the MinGWPy GCC compilers.

## Run C++ tests

After building and installing, the following can run C++ related tests;

<pre>
cd %NUPIC_CORE%\build\release\bin
cpp_region_test
unit_tests
</pre>

## Build notes

The `%NUPIC_CORE%\.gitignore` file has a rule that ignores any directory called `build/` from Git. Making that directory a handy place to store build dependencies.

### MinGWPy GCC

* You must use the `mingwpy_amd64_vc90.7z` toolchain. Other MinGW x64 GCC toolchains do not work. It __must__ be the vc90 version to match up with Python 2.7.9+
* The `libpython-cp27-none-win_amd64.7z` has to be extracted into your Python 2.7 x64 directory _before_ running cmake (two `.a` library files placed into `C:\Python27-x64\libs` directory).
* It is convenient to extract the `mingwpy_amd64_vc90.7z` into a `C:\mingwpy` directory.
* After extracting the MinGW GCC 7z archive your `PATH` environment variable will need `C:\mingwpy\bin;` adding to it.
* `mingw32-make` is used with the CMake generated Makefiles.

### Visual Studio 2015

* The `ZERO_CHECK` project can be ignored. It is only used by CMake during the build process. 
* Any changes made to project configurations from within Visual Studio must be carried across to the `src\CMakeLists.txt` file to become permanent changes.  
* Any changes made to `src\CMakeLists.txt` requires a _(re)_-run of CMake / CMake-GUI to configure and generate new Visual Studio project files with those changes. Make sure that Visual Studio is not running when this step occurs.  
* The solution file (v140 platform toolset) has an `ALL_BUILD` project that can rebuild the x64 Release version of the core library and test programs.
* The `INSTALL` project has to be built after building the main library. External libraries (x64 release) are stored in the Git repository, and packaged into the deployed version. The `INSTALL` project copies the binaries, header, and library files into the `%CMAKE_INSTALL_PREFIX%` directory.
* The `PACKAGE` project implements the `src\CmakeLists.txt` file CPack packaging instructions. It takes the post-INSTALL release files and makes a tape archive file (tar format), that is then compressed (gzip format) into a single file. 

## External libraries

The C++ Core library depends on a handful of support libraries that are distributed within the GitHub repository. The following sub-sections can be used as a guide if you require changes to these pre-built libraries, such as creating MSVC Debug versions.

The Visual Studio x64 Release versions can be found in `%NUPIC_CORE%\external\windows64\lib`  
The MinGWPy GCC versions can be found in `%NUPIC_CORE%\external\windows64-gcc\lib`.

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

Most of the libraries have a CMakeList.txt file, and possibly older workspace (DSW) or solution (DSP) files to convert. A CMakeList.txt file allows the use of the **CMake-GUI** application. Which can make it easy to tweak build variables, e.g. the required update of 'CMAKE_INSTALL_PREFIX' for each library; and advanced options such as compiler settings. 

Building the Apr-iconv library is _not_ required and can be skipped. 

For MSVC 2015 solutions - remember to set the configuration to Debug/Release, and setup a clone configuration of Win32 for **x64**. 

All support libraries are build for x64. 

If a solution contains an Install project, the install scripts are placed inside a file called `cmake_install.cmake` A MSVC INSTALL project in a solution tries to execute the following command;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your `PATH` environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

#### NumPy

The C++ headers from NumPy can be installed via a pre-built NumPy Python package. For example; from http://www.lfd.uci.edu/~gohlke/pythonlibs/ or via MSys2 `pacman -S mingw64/mingw-w64-x86_64-python2-numpy` command.

#### MinGWPy GCC only

##### Apache Portable Runtime (APR)

Install the 64 bit version of MSys2 - http://sourceforge.net/p/msys2/wiki/MSYS2%20installation/

From within the MSys2 environment, run the `update-core` script followed by `pacman -Su`. Using `pacman -Ss apr` searches for APR packages. Use the following commands to install the required APR libraries; `pacman -S mingw64/mingw-w64-x86_64-apr --noconfirm` and `pacman -S mingw64/mingw-w64-x86_64-apr-util --noconfirm` 

This installs (assuming a default installation directory for MSys2) the libraries into `C:/msys64/mingw64/lib`. The two APR libraries (`libapr-1.a` and `libaprutil-1.a`) can then be copied into the `%NUPIC_CORE%\external\windows64-gcc-\lib` directory. And used with the CMake build command as described above to rebuild the core library.

##### Cap'n Proto

This is part of the main nupic.core CMake build process. The appropriate version is git cloned, as well as the associated Cap'n Proto compiler tools. It is built using the `MinGW Makefile` and ends up in the `%NUPIC_CORE%\build\scripts\ThirdParty\Install` directories.

##### Yaml, Yaml-cpp, and Z libbraries

A CMake command, or the CMake-GUI program, can be used to generate MinGW Makefiles. These can then be used with the `mingw32-make -f Makefile install` command to rebuild these libraries. Make sure that the `CMAKE_BUILD_TYPE` and `CMAKE_INSTALL_PREFIX` are set appropriately.

#### Microsoft Visual Compilers only

##### Apache Portable Runtime (APR)

apr.dsw and aprutil.dsw workspace files can be imported into Visual Studio 2015. Directory naming allows these solutions to find the apr-iconv projects. With APR we only require the 'apr' project to be built. The 'libapr' project is for a DLL version. And with Apr-util we just need to build the 'apr-util' project.

##### Cap'n Proto

Download version **0.5.3** from https://capnproto.org/capnproto-c++-win32-0.5.3.zip and extract into %NUPIC_CORE%\build directory.

Install instructions can be found at https://capnproto.org/install.html Below is an example Visual Studio Command Prompt instructions to invoke cmake and generator a solution and project files for Cap'n Proto.

<pre>
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
</pre>

Building the test programs may halt a full build. But enough will be built for an Install, and finally copy of the new capnp.lib and kj.lib libraries.

##### Yaml

A valid libyaml.sln solution file can be found in directory `yaml-0.1.5\win32\vs2008` A new x64 platform solution can be added to it once imported into Visual Studio 2015. We only need to build the yaml project from this solution.

##### Yaml-cpp

In CMake-GUI, `%NUPIC_CORE%/build/yaml-cpp/` can be used for Source and Build directories. Make sure that MSVC_SHARED_RT is **ticked**, and BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not ticked**. When building the solution you may need to `#include <algorithm>` in `src\ostream_wrapper.cpp` 

##### Z Lib

In `zlib-1.2.8\contrib\vstudio` there are solutions for Visual Studio 9, 10, and 11. The vc11 solution can be used with Visual Studio 2015. A x64 platform solution needs to be added to this imported solution. The zlibstat is the library we need to to build, and copy the rebuilt library over the z.lib file in directory `%NUPIC_CORE%/external/windows64/lib`

