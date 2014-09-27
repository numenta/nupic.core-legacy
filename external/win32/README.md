# NuPIC Core external libraries

NuPIC Core depends on a number of pre-built external libraries which are
normally distributed with the source.  Use the following commands as a guide.

## Obtaining the library sources

Obtain the source for the following libraries. These will be statically linked to the NuPIC core libraries.

| Library | Website | Filename |
|:------- |:------- |:-------- |
| apr-1.5.1 | https://apr.apache.org/ | apr-1.5.1-win32-src.zip |
| apr-util-1.5.4 | https://apr.apache.org/ | apr-util-1.5.4-win32-src.zip |
| apr-iconv-1.2.1 | https://apr.apache.org/ | apr-iconv-1.2.1-win32-src-r2.zip |
| pcre-8.35 | http://www.pcre.org/ | pcre-8.35.zip |
| yaml-0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| yaml-cpp-0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| zlib-1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |

Extract them into $NUPIC_CORE/external/win32/build The root .gitignore contains a rule to make any directory called build/ be ignored by Github.

Next, perform some housekeeping tasks on the Apr directories by renaming them  

apr-1.5.1 to apr  
apr-iconv-1.2.1 to apr-iconv  
apr-util-1.5.4 to apr-util  

Make sure that all these sub-directories are correct. Depending on how you extract the gzip/zip files into the build/directory, you may end up with, for example, pcre-8.35/pcre-8.35/..  

## CMake building the external libraries

All but one of these libraries has a CMakeList.txt, that allows for the use of the Windows version of CMake. The CMake-GUI application **must** be used for all these external projects. The Windows version of CMake is the only versions that supports Visual Studio generators. Plus it makes it easy to make tweaks to the build environments, e.g. changing 'CMAKE_INSTALL_PREFIX'

Configure each one with the generator that matches your installed IDE.

Open each solution file (.sln) into your Visual Studio IDE, in alphabetical order of each external library. Using Rebuild Solution. ***Remember** to set the solution configuration in the Configuration Manager to Release.

Install scripts are placed inside a file called cmake_install.cmake The INSTALL project in each solution doesn't work, but tries to do the following;  

cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

**Note**: The apr-iconv library has two project files to import into Visual Studio. We just need the static version called apriconv.dsp

### Example CMake build - Apache Portable Runtime (APR)

For example, the Apr (1.5.1) project;  
  - Run CMake-GUI (3.0.1)
  - Setup the following options;
    * Where is the source code:    $NUPIC_CORE/external/win64/build/apr  
    * Where to build the binaries: $NUPIC_CORE/external/win64/build/apr  
  - Press 'Configure' button
  - Change CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64  
  - Press 'Configure' button again.
    * Red backgrounds on Name/Value pairs should go back to white.
    * Correct any Name/Value pairs that are still have a red background and press 'Configure'.
  - When all Name/Value pairs are valid press 'Generate'.

Open APR.sln with your Visual Studio IDE and Rebuild All on the the solution. Once rebuilt, 'Build Only' the INSTALL project to get the library files into the right places. It makes an initial mess of the $NUPIC_CORE/include directory, but we need it like that for now.

In the apr/ directory there will be a x64/ sub-directory. Copy the win64/build/apr/x64/LibR/apr-1.lib to win64/lib

Now repeat all the above steps for the APR-Util solution. Apr-util requires a couple of other changes. 

  * Point APR_INCLUDE to $NUPIC_CORE/external/win64/build/apr
  * Point APR_LIBRARIES to $NUPIC_CORE/external/win64/build/apr
  * Turn off APR_HAS_LDAP and APU_HAVE_ODBC
  * And point CMAKE_INSTALL_PREFIX to $NUPIC_CORE/external/win64

For the apr-iconv library, import apriconv.dsp into Visual Studio and Rebuild all. Copy the build\apriconv\x64\LibR\apriconv.lib into the win64/lib

Time to tidy the $NUPIC_CORE/external/win64/include directory.  
Make a new directory called apr-1 Move all the .h files into it.

And that is APR built and ready to be statically linked to the NuPIC core library.

### Possible build issues  

#### PCRE

'PCRE_SUPPORT_UTF is off by default.

#### Yaml

Has a valid libyaml.sln solution file for importing. Found in directory yaml-0.1.5\win32\vs2008 A <New...> x64 platform solution can be added to it, once loaded into Visual Studio.

#### Yaml-cpp  

'#include <algorithm>' in src\ostream_wrapper.cpp  

#### Z Lib  

Look in zlib-1.2.8\contrib\vstudio for contributed solutions and projects for Visual Studio.


For CMake assistance in copying the relevant binaries, headers, and library files 
'nupic.core\external\win32\apr>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\include -P cmake_install.cmake  
'nupic.core\external\win32\pcre-8.35\cmake>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\..\include\pcre -P cmake_install.cmake  


And, of course to build NuPIC Core itself:
