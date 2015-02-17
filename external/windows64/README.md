# NuPIC Core ExternaL Libraries

NuPIC Core depends on a handful of external libraries that are distributed within the download package. The following can be used as a guide if you require changes to these pre-built libraries. To use any rebuilt external library, you need to link against the %NUPIC_CORE%/lib/nupic_core_solo library.

## Obtaining the library sources

These libraries are embedded into the %NUPIC_CORE%/lib/nupic_core library.

| Library | Version | Website | Filename |
|:------- |:------- |:------- |:-------- |
| Apache Portable Runtime (Apr) | 1.5.1 | https://apr.apache.org/ | apr-1.5.1-win32-src.zip |
| Apr util | 1.5.4 | https://apr.apache.org/ | apr-util-1.5.4-win32-src.zip |
| Apr iconv | 1.2.1 | https://apr.apache.org/ | apr-iconv-1.2.1-win32-src-r2.zip |
| Cap'n Proto | 0.5.0 | https://capnproto.org | capnproto-c++-win32-0.5.0.zip |
| Yaml | 0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| Yaml Cpp | 0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| Z lib | 1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |

Extract them all into %NUPIC_CORE_SOURCE%/build (the %NUPIC_CORE_SOURCE%/.gitignore file has a rule that ignores any directory called build/).

Perform some housekeeping tasks on the Apr directories by renaming them -
apr-1.5.1 to apr  
apr-iconv-1.2.1 to apr-iconv  
apr-util-1.5.4 to apr-util  

## Building the external libraries

Most of the libraries have a CMakeList.txt file, and possibly older workspace or solution files to convert. A CMakeList.txt file allows the use of **CMake-GUI** application, which can make it easy to tweak build variables, e.g. the required update of 'CMAKE_INSTALL_PREFIX' for each library; and advanced options such as compiler settings. Building the Apr-iconv library is _not_ required and can be skipped. 

**Remember** to set the solution configuration in the Configuration Manager to Release. And if required setup a clone configuration of Win32 for **x64**.

If a solution contains an Install project, the install scripts are placed inside a file called cmake_install.cmake An INSTALL project in a solution tries to do the following;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your %PATH% environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

## Cap'n Proto

Download version 0.5.0 from https://capnproto.org/capnproto-c++-win32-0.5.0.zip The three executables found in  %NUPIC_CORE_SOURCE%\build\capnproto-tools-win32-0.5.0 should match the %NUPIC_CORE_SOURCE%\external\windows64\bin executables. 
Install instructions can be found at https://capnproto.org/install.html Here is an example Visual Studio Command Prompt line to invoke cmake and generator a solution and project files for Cap'n Proto.

> %NUPIC_CORE_SOURCE%\build\capnproto-c++-0.5.0>cmake -G "Visual Studio 14 2015 Win64" -DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1 -DCAPNP_INCLUDE_DIRS=..\..\external\common\include -DCAPNP_LIB_KJ=.\ -DCAPNP_LIB_CAPNP=.\ -DCAPNP_EXECUTABLE="..\capnproto-tools-win32-0.5.0\capnpc-c++.exe"
 
## Yaml

This has a valid libyaml.sln solution file for importing. Found in directory yaml-0.1.5\win32\vs2008 A new x64 platform solution can be added to it, once loaded into Visual Studio.

## Yaml-cpp  

Using %NUPIC_CORE_SOURCE%/build/yaml-cpp/ for Source and Build directoreis, make sure MSVC_SHARED_RT **is** ticked. And BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not** ticked. When building the solution your may need to '#include <algorithm>' in src\ostream_wrapper.cpp 

## Z Lib  

In zlib-1.2.8\contrib\vstudio there are solutions and projects for Visual Studio 9, 10, and 11. The vc11 can be used with Visual Studio 2015.  A <New...> x64 platform solution can be added to the solution, once loaded into Visual Studio. The zlibstat is the library we need to copy to %NUPIC_CORE_SOURCE%/external/windows64/lib and overwrite z.lib.

## CMake-GUI settings for NuPIC Core

| Name | Value |
|:---- |:----- |
| Source code | %NUPIC_CORE_SOURCE%/src |
| Binaries | %NUPIC_CORE_SOURCE%/build/scripts |
| CMAKE_INSTALL_PREFIX | %NUPIC_CORE_SOURCE%/build/release |
