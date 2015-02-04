# NuPIC Core ExternaL Libraries

NuPIC Core depends on a number of pre-built external libraries which are normally distributed with the source. The following can be used as a guide if you require changes to these pre-built libraries.

## Obtaining the library sources

These libraries will be statically linked to the NuPIC core library.

| Library | Website | Filename |
|:------- |:------- |:-------- |
| apr-1.5.1 | https://apr.apache.org/ | apr-1.5.1-win32-src.zip |
| apr-util-1.5.4 | https://apr.apache.org/ | apr-util-1.5.4-win32-src.zip |
| apr-iconv-1.2.1 | https://apr.apache.org/ | apr-iconv-1.2.1-win32-src-r2.zip |
| yaml-0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| yaml-cpp-0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| zlib-1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |

Extract them into %NUPIC_CORE%/external/win32/build The %NUPIC_CORE%/.gitignore file has a rule that makes any directory called build/ to be ignored by Git.

Perform some housekeeping tasks on the Apr directories by renaming them. This is an expected step for the Apr build system. 
apr-1.5.1 to apr 
apr-iconv-1.2.1 to apr-iconv 
apr-util-1.5.4 to apr-util 

## CMake building the external libraries

Most of these libraries have a CMakeList.txt file. That allows for the use of the Windows version of CMake. The CMake-GUI application makes it easy to tweak build variables, e.g. the required update of 'CMAKE_INSTALL_PREFIX' for each library. Building the Apr-iconv library is not required and can be skipped. 

**Remember** to set the solution configuration in the Configuration Manager to Release. And if required setup a clone configuration of Win32 for **x64**.

If a solution contains an Install project, the install scripts are placed inside a file called cmake_install.cmake An INSTALL project in a solution tries to do the following;  

cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your %PATH% environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin"). Restart Visual Studio _and_ VS Command Prompt when this changes.

### Example CMake build - Apr Apache Portable Runtime

- Run CMake-GUI application
- Setup the following options -
  * Where is the source code:    $NUPIC_CORE/external/win64/build/apr  
  * Where to build the binaries: $NUPIC_CORE/external/win64/build/apr  
- Press the 'Configure' button
- Change CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64  
- Press 'Configure' button
  * Red backgrounds on Name/Value pairs should go back to white.
  * Any Name/Value pairs that are still have a red background **must** be resolved
- Press 'Configure' button, until all pairs resolved
- Press 'Generate'

Open APR.sln with your Visual Studio IDE and 'Rebuild All' on the solution. Then 'Build Only' the INSTALL project. In the apr/ directory there will be a x64/ sub-directory. Copy the apr/x64/LibR/apr-1.lib to $NUPIC_CORE/external/win64/lib/apr-1.lib 

Repeat the above steps for the APR-Util solution. Apr-util requires a couple of other changes in CMake-GUI - 

  * Point APR_INCLUDE to $NUPIC_CORE/external/win64/build/apr
  * Point APR_LIBRARIES to $NUPIC_CORE/external/win64/build/apr
  * Turn off APR_HAS_LDAP and APU_HAVE_ODBC (for now..)
  * And point CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64

Time to tidy the $NUPIC_CORE/external/win64/include directory. Make a new directory called apr-1 and move all the .h files into it.

You may need to edit apr_arch_utf8.h Changing three #include files; 
'#include "apr.h" 
'#include "apr_lib.h" 
'#include "apr_errno.h" 
to  
'#include "apr-1/apr.h" 
'#include "apr-1/apr_lib.h" 
'#include "apr-1/apr_errno.h" 

#### Yaml

This has a valid libyaml.sln solution file for importing. Found in directory yaml-0.1.5\win32\vs2008 A <New...> x64 platform solution can be added to it, once loaded into Visual Studio.

#### Yaml-cpp  

Using $NUPIC_CORE/build/yaml-cpp/ for Source and Build directoreis, make sure MSVC_SHARED_RT **is** ticked. And BUILD_SHARED_LIBS and MSVC_STHREADED_RT are both **not** ticked. When building the solution your may need to '#include <algorithm>' in src\ostream_wrapper.cpp 

#### Z Lib  

In zlib-1.2.8\contrib\vstudio there are solutions and projects for Visual Studio 9, 10, and 11. The vc11 can be used with Visual Studio 2015.  A <New...> x64 platform solution can be added to the solution, once loaded into Visual Studio. The zlibstat is the library we need to copy to %NUPIC_CORE%/external/windows64/lib and overwrite z.lib (or zd.lib for debug version).

#### Misc

For CMake assistance in copying the relevant binaries, headers, and library files 
'nupic.core\external\win32\apr>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\include -P cmake_install.cmake'  

To rebuild NuPIC Core itself using the CMake-GUI application, the following settings can be used.

| Name | Value |
|:---- |:----- |
| Source code | %NUPIC_CORE%/src |
| Binaries | %NUPIC_CORE%/build/scripts |
| CMAKE_INSTALL_PREFIX | %NUPIC_CORE%/build/release |
