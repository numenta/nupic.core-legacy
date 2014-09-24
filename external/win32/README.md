NuPIC Core external libraries
=============================

NuPIC Core depends on a number of pre-built external libraries which are
normally distributed with the source.  Use the following commands as a guide.

Obtain the source for the following libraries. These will be statically linked to the NuPIC core libraries.

| Library | Website | Filename |
|:------- |:------- |:-------- |
| apr-1.5.1 | http://www.apache.org/dist/apr/ | apr-1.5.1-win32-src.zip |
| apr-iconv-1.2.1 | http://www.apache.org/dist/apr/ | apr-iconv-1.2.1-win32-src-r2.zip |
| apr-util-1.5.3 | https://apr.apache.org/ | apr-util-1.5.3-win32-src.zip |
| boost-1.52.0 | http://www.boost.org/users/history/version_1_52_0.html | boost_1_52_0.zip |
| yaml-0.1.5 | http://pyyaml.org/wiki/LibYAML | yaml-0.1.5.tar.gz |
| yaml-cpp-0.3.0 | https://code.google.com/p/yaml-cpp/ | yaml-cpp-0.3.0.tar.gz |
| zlib-1.2.8 | http://www.zlib.net/ | zlib-1.2.8.tar.gz |
| pcre-8.35 | http://www.pcre.org/ | pcre-8.35.zip |

Extract them into $NUPIC_CORE/external/win32

And perform some housekeeping tasks on the apr directories by renaming them

apr-1.5.1 to apr  
apr-util-1.5.3 to apr-util  
apr-iconv-1.2.1 to apr-iconv  

Each one has a CMakeList.txt that allows for the use of the Windows version of CMake. The CMake-GUI version **must** be used to build Visual Studio project and solution files for all these external projects. The Windows version of CMake is the only versions that supports Visual Studio generators. Plus it is easy to make needed tweaks to the build environments, e.g. 'CMAKE_INSTALL_PREFIX'

Configure, and then Generate, each one with the generator that matches your installed IDE.

Open each solution file (.sln) in your VS IDE and build the Release configuration.

Possible  
libapr-1 Link errors in apr_atomic with VS2013  
http://mail-archives.apache.org/mod_mbox/apr-dev/201311.mbox/%3C1383702562.18420.YahooMailNeo@web122303.mail.ne1.yahoo.com%3E
Remove all "(apr_atomic_win32_ptr_fn)" unresolved external symbol Interlocked from apr_atomic.c in both apr-1 and libapr-1 libraries

Copy apr\include\arch to win32\include\apr-1\arch

Edit apr_arch_utf8.h and change the three #include from
...C++
#include "apr.h"
#include "apr_lib.h"
#include "apr_errno.h"
...
to
...C++
#include "apr-1/apr.h"
#include "apr-1/apr_lib.h"
#include "apr-1/apr_errno.h"
...

yaml-cpp
#include <algorithm> in src\ostream_wrapper.cpp

Z Lib
contrib\vstudio\vc11\x86\ZlibStatRelease\zlibstat.lib  
zlib-1.2.8\contrib\vstudio\vc11 if needed for comparison to CMake version of ZLib projects  

For CMake assistance in copying the relevant binaries, headers, and library files  
nupic.core\external\win32\apr>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\include -P cmake_install.cmake  
nupic.core\external\win32\pcre-8.35\cmake>cmake -DBUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..\..\include\pcre -P   cmake_install.cmake  


And, of course to build NuPIC Core itself:
