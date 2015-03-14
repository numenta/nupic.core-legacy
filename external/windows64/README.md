# NuPIC Core

The core library of NuPIC uses a CMake based system to define build characteristics. The [CMake-GUI](http://www.cmake.org/) application can be used to generator a Visual Studio 2015 (Win64) solution and project files. 

Two environment variables must be setup before building the core; 

- `NUPIC_CORE_SOURCE` points to the root of the core directories, and
- `NUPIC_CORE` points to an installation directory.

The following applications are required to rebuilt the core and external libraries;

- [Microsoft Visual Studio Ultimate 2015 Preview](http://www.visualstudio.com/en-us/downloads/visual-studio-2015-downloads-vs) ([installer webpage](http://go.microsoft.com/?linkid=9863611&clcid=0x409))
- [CMake](http://www.cmake.org/) (version 3.1 onwards to get the Visual Studio 2015 generator).

The following table shows the CMake-GUI settings for the core library. Using these table settings the `%NUPIC_CORE%` directory could equal the CMAKE_INSTALL_PREFIX directory.

| Name | Value |
|:---- |:----- |
| Source code | `%NUPIC_CORE_SOURCE%/src` |
| Binaries | `%NUPIC_CORE_SOURCE%/build/scripts` |
| CMAKE_INSTALL_PREFIX | `%NUPIC_CORE_SOURCE%/build/release` |

The `%NUPIC_CORE_SOURCE%/build/scripts/nupic_core.sln` solution (v140 platform toolset) has an ALL_BUILD project that can rebuild the x64 Release version of the core library and test programs. As well as running the cpp_region test and unit tests.

**Notes:**
* The `%NUPIC_CORE_SOURCE%/.gitignore` file has a rule that ignores any directory called `build/` from Git. A convenient place to store build dependencies, such as debug libraries, as branches change and CMake-GUI is rerun.
* The `ZERO_CHECK` project is used by CMake during the build process. 
* With this being a CMake based cross-platform project, any changes made to project configurations from within Visual Studio must be carried across to the `src\CMakeLists.txt` file to become permanent changes.  
* Consequently, any changes made to `src\CMakeLists.txt` requires a _(re)_-run of CMake-GUI to configure and generate new Visual Studio project files with those changes. Make sure that Visual Studio is not running when this step occurs.  

The INSTALL project has to be built separately. External libraries (x64 release) are stored in the Git repository, and packaged into the deployed version. When the ALL_BUILD project completes, a separate run of the INSTALL project copies the binaries, header, and library files into the `%NUPIC_CORE%` directory.

The PACKAGE project implements the `src\CmakeLists.txt` file CPack packaging instructions. It takes the post-INSTALL release files and makes a tape archive file (tar format) that is then compressed (gzip format) into a single file. The `%NUPIC_CORE_SOURCE%\appveyor.yml` file shows an alternative method for library deployment using the 7z application.

The NuPIC core library file is split into two files;

- `%NUPIC_CORE%/lib/nupic_core_solo.lib` contains _only_ the core library
- `%NUPIC_CORE%/lib/nupic_core.lib` contains the core and support libraries

## External libraries

The Core library depends on a handful of external libraries that are distributed within the download package. The Windows x64 version of these libraries can be found in `%NUPIC_CORE_SOURCE%\external\windows64\lib` The following can be used as a guide if you require changes to these pre-built libraries, such as creating Debug versions.

### Obtaining the library sources

The following libraries are embedded into the `%NUPIC_CORE%/lib/nupic_core` library.

| Library | Version | Filename |  Website |
|:------- |:------- |:-------- | :------- |
| APR | **1.5.1** | apr-**1.5.1**-win32-src.zip | https://apr.apache.org/ |
| Apr Util | **1.5.4** | apr-util-**1.5.4**-win32-src.zip | https://apr.apache.org/ |
| Apr Iconv | **1.2.1** | apr-iconv-**1.2.1**-win32-src-r2.zip | https://apr.apache.org/ |
| Cap'n Proto | **0.5.0** | capnproto-c++-win32-**0.5.0**.zip | https://capnproto.org |
| Yaml | **0.1.5** | yaml-**0.1.5**.tar.gz | http://pyyaml.org/wiki/LibYAML |
| Yaml Cpp | **0.3.0** | yaml-cpp-**0.3.0**.tar.gz | https://code.google.com/p/yaml-cpp/ |
| Z Lib | **1.2.8** | zlib-**1.2.8**.tar.gz | http://www.zlib.net/ |

Extract them all into `%NUPIC_CORE_SOURCE%/build` directory. You may need to un-nest some of the directories, e.g. `apr-1.5.1/apr-1.5.1/...` to `apr-1.5.1/...` APR expects certain directory names, rename the following directories;  

`apr-1.5.1` to `apr`  
`apr-iconv-1.2.1` to `apr-iconv`  
`apr-util-1.5.4` to `apr-util`  

### Building the external libraries

Most of the libraries have a CMakeList.txt file, and possibly older workspace or solution files to convert. A CMakeList.txt file allows the use of **CMake-GUI** application. Which can make it easy to tweak build variables, e.g. the required update of 'CMAKE_INSTALL_PREFIX' for each library; and advanced options such as compiler settings. Building the Apr-iconv library is _not_ required and can be skipped. **Remember** to set the solution configuration in the Configuration Manager to Release. And if required setup a clone configuration of Win32 for **x64**.

If a solution contains an Install project, the install scripts are placed inside a file called cmake_install.cmake An INSTALL project in a solution tries to do the following;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your `%PATH%` environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

### Apache Portable Runtime (APR)

apr.dsw and aprutil.dsw workspace files can be imported into Visual Studio 2015. Directory naming allows these solutions to find the apr-iconv projects. With APR we only require the 'apr' project to be built. The 'libapr' project is for a DLL version. And with Apr-util we just need to build the 'apr-util' project.

### Cap'n Proto

Download version **0.5.0** from https://capnproto.org/capnproto-c++-win32-0.5.0.zip 

The three executable files found in `%NUPIC_CORE_SOURCE%\build\capnproto-tools-win32-0.5.0` should match the corresponding `%NUPIC_CORE_SOURCE%\external\windows64\bin` executable files, and tie in with the external capnp and kj common include directories. 

Install instructions can be found at https://capnproto.org/install.html This is an example Visual Studio Command Prompt line to invoke cmake and generator a solution and project files for Cap'n Proto.

> cd %NUPIC_CORE_SOURCE%\build\capnproto-c++-0.5.0  
> vcvarsall.bat  
> cmake -G "Visual Studio 14 2015 Win64" -DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1 -DCAPNP_INCLUDE_DIRS=..\\..\external\common\include -DCAPNP_LIB_KJ=.\ -DCAPNP_LIB_CAPNP=.\ -DCAPNP_EXECUTABLE="..\capnproto-tools-win32-0.5.0\capnpc-c++.exe"  
 
Building the test programs may halt a full build. But enough will be built for an Install, and finally copy out the new capnp.lib and kj.lib libraries.

### Yaml

A valid libyaml.sln solution file can be found in directory `yaml-0.1.5\win32\vs2008` A new x64 platform solution can be added to it once imported into Visual Studio 2015. We only need to build the yaml project from this solution.

### Yaml-cpp  

In CMake-GUI `%NUPIC_CORE_SOURCE%/build/yaml-cpp/` can be used for Source and Build directories. Make sure that MSVC_SHARED_RT is **ticked**, and BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not ticked**. When building the solution you may need to `#include <algorithm>` in `src\ostream_wrapper.cpp` 

### Z Lib  

In `zlib-1.2.8\contrib\vstudio` there are solutions for Visual Studio 9, 10, and 11. The vc11 solution can be used with Visual Studio 2015. A x64 platform solution can be added to this imported solution. The zlibstat is the library we need to overwite z.lib in directory `%NUPIC_CORE_SOURCE%/external/windows64/lib`
