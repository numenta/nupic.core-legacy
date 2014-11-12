# NuPIC Core external libraries

NuPIC Core depends on a number of pre-built external libraries which are
normally distributed with the source.  

The following can be used as a guide if you require changes to the pre-built libraries.

## Obtaining the library sources

Obtain the source for the following libraries. These will be statically linked to the NuPIC core libraries.

| Library | Website | Filename |
|:------- |:------- |:-------- |
| apr-1.5.1 | https://apr.apache.org/ | apr-1.5.1-win32-src.zip |
| apr-util-1.5.4 | https://apr.apache.org/ | apr-util-1.5.4-win32-src.zip |
| apr-iconv-1.2.1 | https://apr.apache.org/ | apr-iconv-1.2.1-win32-src-r2.zip |
| yaml-0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| yaml-cpp-0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| zlib-1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |

Extract them into $NUPIC_CORE/external/win32/build

.gitignore contains a rule to make any directory called build/ be ignored by Git.

Next, perform some housekeeping tasks on the Apr directories by renaming them. This is an expected step for the Apr build system.

apr-1.5.1 to apr  
apr-iconv-1.2.1 to apr-iconv  
apr-util-1.5.4 to apr-util  


## CMake building the external libraries

Most of these libraries have a CMakeList.txt file. That allows for the use of the Windows version of CMake, the only versions that support Visual Studio generators. The CMake-GUI application also makes it easy to make tweak the build environment, e.g. changing 'CMAKE_INSTALL_PREFIX'.

Open each solution file (.sln) into your Visual Studio IDE in alphabetical order of each external library. Building Apr-iconv library is not required and can be skipped. **Remember** to set the solution configuration in the Configuration Manager to Release. And if required setup a clone configuration for **x64**.

Install scripts are placed inside a file called cmake_install.cmake The INSTALL project in each solution tries to do the following;  

cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

### Example CMake build - Apache Portable Runtime (APR)

- Run CMake-GUI application
- Setup the following options -
  * Where is the source code:    $NUPIC_CORE/external/win64/build/apr  
  * Where to build the binaries: $NUPIC_CORE/external/win64/build/apr  
- Press the 'Configure' button
- Change CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64  
- Press 'Configure' button again.
  * Red backgrounds on Name/Value pairs should go back to white.
  * Any Name/Value pairs that are still have a red background **must** be resolved, before pressing the 'Configure' button again.
- When all Name/Value pairs are valid press 'Generate'.

Open APR.sln with your Visual Studio IDE and 'Rebuild All' on the solution. Once rebuilt 'Build Only' the INSTALL project to get the library files into the right places.

In the apr/ directory there will be a x64/ sub-directory. Copy the win64/build/apr/x64/LibR/apr-1.lib to win64/lib

Now repeat all the above steps for the APR-Util solution. Apr-util requires a couple of other changes. 

  * Point APR_INCLUDE to $NUPIC_CORE/external/win64/build/apr
  * Point APR_LIBRARIES to $NUPIC_CORE/external/win64/build/apr
  * Turn off APR_HAS_LDAP and APU_HAVE_ODBC
  * And point CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64

Time to tidy the $NUPIC_CORE/external/win64/include directory. Make a new directory called apr-1 and move all the .h files into it.

That is APR built and ready to be statically linked to the main NuPIC core library.

### Possible build issues  

#### APR

Edit apr_arch_utf8.h and change the three #include from  
'#include "apr.h"  
'#include "apr_lib.h"  
'#include "apr_errno.h"  

to  

'#include "apr-1/apr.h"  
'#include "apr-1/apr_lib.h"  
'#include "apr-1/apr_errno.h"  

#### Yaml

Has a valid libyaml.sln solution file for importing. Found in directory yaml-0.1.5\win32\vs2008 A <New...> x64 platform solution can be added to it, once loaded into Visual Studio.

#### Yaml-cpp  

'#include <algorithm>' in src\ostream_wrapper.cpp  

#### Z Lib  

Look in zlib-1.2.8\contrib\vstudio for contributed solutions and projects for Visual Studio.

#### Installation

For CMake assistance in copying the relevant binaries, headers, and library files 
'nupic.core\external\win32\apr>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\include -P cmake_install.cmake'  
'nupic.core\external\win32\pcre-8.35\cmake>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\..\include\pcre -P cmake_install.cmake'  


To rebuild NuPIC Core itself using CMake-GUI application, the following settings can be used.

| Name | Value |
|:---- |:----- |
| Source code | $NUPIC_CORE/src |
| Binaries | $NUPIC_CORE/build/scripts |
| CMAKE_INSTALL_PREFIX | $NUPIC_CORE/build/release |


