# NuPIC Core

The core library of NuPIC uses a CMake based system to define build characteristics. The [CMake-GUI](http://www.cmake.org/) application can be used to generator Visual Studio 2015 (Win64) solution and project files. The following table shows the CMake-GUI settings for the core library -

| Name | Value |
|:---- |:----- |
| Source code | %NUPIC_CORE_SOURCE%/src |
| Binaries | %NUPIC_CORE_SOURCE%/build/scripts |
| CMAKE_INSTALL_PREFIX | %NUPIC_CORE_SOURCE%/build/release |

The %NUPIC_CORE_SOURCE%/build/scripts/nupic_core.sln solution (v140 platform toolset) has an ALL_BUILD project that can rebuild the x64 Release version of the core library and test programs. As well as running the cpp_region test and unit tests. The INSTALL project has to be built seperately. External libraries (x64 release) are stored in the Git repository, and packaged into the deployed version.

## External libraries

The Core library depends on a handful of external libraries that are distributed within the download package. The Windows x64 version of these libraries can be found in %NUPIC_CORE_SOURCE%\external\windows64\lib

The following can be used as a guide if you require changes to these pre-built libraries. To use any rebuilt external library, you need to link against the %NUPIC_CORE%/lib/nupic_core_solo library.

### Obtaining the library sources

These libraries are embedded into the %NUPIC_CORE%/lib/nupic_core library.

| Library | Version | Website | Filename |
|:------- |:------- |:------- |:-------- |
| APR | 1.5.1 | https://apr.apache.org/ | apr-1.5.1-win32-src.zip |
| Apr Util | 1.5.4 | https://apr.apache.org/ | apr-util-1.5.4-win32-src.zip |
| Apr Iconv | 1.2.1 | https://apr.apache.org/ | apr-iconv-1.2.1-win32-src-r2.zip |
| Cap'n Proto | 0.5.0 | https://capnproto.org | capnproto-c++-win32-0.5.0.zip |
| Yaml | 0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| Yaml Cpp | 0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| Z Lib | 1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |

Extract them all into %NUPIC_CORE_SOURCE%/build directory. The %NUPIC_CORE_SOURCE%/.gitignore file has a rule that ignores any directory called build/ You may need to un-nest some of the directories, e.g. apr-1.5.1/apr-1.5.1/... to apr-1.5.1/...

Perform some housekeeping tasks on the Apr directories by renaming them -  

apr-1.5.1 to apr  
apr-iconv-1.2.1 to apr-iconv  
apr-util-1.5.4 to apr-util  

### Building the external libraries

Most of the libraries have a CMakeList.txt file, and possibly older workspace or solution files to convert. A CMakeList.txt file allows the use of **CMake-GUI** application. Which can make it easy to tweak build variables, e.g. the required update of 'CMAKE_INSTALL_PREFIX' for each library; and advanced options such as compiler settings. Building the Apr-iconv library is _not_ required and can be skipped. **Remember** to set the solution configuration in the Configuration Manager to Release. And if required setup a clone configuration of Win32 for **x64**.

If a solution contains an Install project, the install scripts are placed inside a file called cmake_install.cmake An INSTALL project in a solution tries to do the following;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your %PATH% environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

### Apache Portable Runtime (APR)

apr.dsw and aprutil.dsw workspace files can be imported into Visual Studio 2015. Directory naming allows these solutions to find the apr-iconv projects. With APR we only require the 'apr' project to be built. The 'libapr' project is for a DLL version. And with Apr-util we just need to build the 'apr-util' project.

### Cap'n Proto

Download version 0.5.0 from https://capnproto.org/capnproto-c++-win32-0.5.0.zip 

The three executables found in %NUPIC_CORE_SOURCE%\build\capnproto-tools-win32-0.5.0 should match the %NUPIC_CORE_SOURCE%\external\windows64\bin executables, and tie in with the external capnp and kj common include directories. 

Install instructions can be found at https://capnproto.org/install.html This is an example Visual Studio Command Prompt line to invoke cmake and generator a solution and project files for Cap'n Proto.

> cd %NUPIC_CORE_SOURCE%\build\capnproto-c++-0.5.0  
> vcvarsall.bat  
> cmake -G "Visual Studio 14 2015 Win64" -DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1 -DCAPNP_INCLUDE_DIRS=..\\..\external\common\include -DCAPNP_LIB_KJ=.\ -DCAPNP_LIB_CAPNP=.\ -DCAPNP_EXECUTABLE="..\capnproto-tools-win32-0.5.0\capnpc-c++.exe"  
 
### Yaml

A valid libyaml.sln solution file can be found in directory yaml-0.1.5\win32\vs2008 A new x64 platform solution can be added to it once imported into Visual Studio 2015.

### Yaml-cpp  

In CMake-GUI %NUPIC_CORE_SOURCE%/build/yaml-cpp/ can be used for Source and Build directoreis. Make sure that MSVC_SHARED_RT is **ticked**, and BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not ticked**. When building the solution you may need to '#include <algorithm>' in src\ostream_wrapper.cpp 

### Z Lib  

In zlib-1.2.8\contrib\vstudio there are solutions for Visual Studio 9, 10, and 11. The vc11 solution can be used with Visual Studio 2015. A x64 platform solution can be added to this imported solution. The zlibstat is the library we need to overwite z.lib in directory %NUPIC_CORE_SOURCE%/external/windows64/lib
