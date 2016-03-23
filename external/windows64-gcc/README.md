# Introduction

The NuPIC Core C++ library can be built with MinGWPy GCC or Microsoft (MS) Visual C compilers. The Python nupic.bindings can **ONLY** be built with the MinGWPy GCC compiler. To use the C++ library with the [NuPIC Python](https://github.com/numenta/nupic) project you must build the library with the MinGW-Py GCC tools, as described below.

NuPIC's dependency on the Core library can be found here - https://github.com/numenta/nupic/wiki/NuPIC's-Dependency-on-nupic.core

It describes how the C++ library is linked to other languages using SWIG, such as Python x64. The NuPIC.Core repository consist of two parts;

- Main C++ core library, and 
- Python SWIG based bindings.

The NuPIC Core C++ library file is split into two files;

- `%NUPIC_CORE%/release/lib/nupic_core_solo.lib` contains _only_ the core library
- `%NUPIC_CORE%/release/lib/nupic_core.lib` contains the C++ core and external support libraries

Where `NUPIC_CORE` is an environment variable that points to the git cloned directory.

CMake based build files are used to define the entire build process. The nupic.bindings SWIG side uses Python distutil and setuptools. The [CMake-GUI](http://www.cmake.org/) application _can_ be used to generate MinGW Makefiles files.

The `%NUPIC_CORE%\appveyor.yml` script file shows how the AppVeyor automated build system progresses through building, packaging, and deployment using MinGWPy GCC.

If you need to make changes to the nupic.core C++ code and/or Python SWIG bindings, follow the remaining sections to see how to rebuild it.

## Support applications

The following applications are required when rebuilding the core C++ library, Python SWIG bindings, and if required the external support libraries;

- [CMake](http://www.cmake.org/) - version 3.1+
- [Python 2.7.9+](https://www.python.org/downloads/windows/) - x86-64 version
- [7-Zip](http://www.7-zip.org/) - Or any archive manager that can extract `.7z` files
- [MinGW GCC for Python](https://bitbucket.org/carlkl/mingw-w64-for-python/downloads) - `mingwpy_amd64_vc90.7z` and `libpython-cp27-none-win_amd64.7z`
- [NumPy C++ headers](http://www.lfd.uci.edu/~gohlke/pythonlibs/) - pip install `numpy-1.9.3+vanilla-cp27-none-win_amd64.whl` _or via_ MSys2 command `pacman -S mingw64/mingw-w64-x86_64-python2-numpy`
- [MSYS2](http://sourceforge.net/p/msys2/wiki/MSYS2%20introduction/) - 64 bit version. Only needed when _rebuilding_ external support libraries with MinGWPy GCC or installing alternative support libraries

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

For rebuilding nupic.core and nupic.bindings you need to use the `MinGW Makefiles` generator. For example;

```
rem Clone the repo
git clone https://github.com/numenta/nupic.core.git
cd nupic.core

rem Setup Python x64 to use MinGWPy GCC compilers
copy nupic.core\external\windows64-gcc\bin\distutils.cfg C:\Python27-x64\Lib\distutils

rem Setup nupic.core and a place to store build files
cd nupic.core
set NUPIC_CORE=%CD%
mkdir build\scripts
cd build\scripts

rem Run cmake to generator MinGW Makefiles
cmake -G "MinGW Makefiles"
	-DCMAKE_BUILD_TYPE=Debug
	-DCMAKE_INSTALL_PREFIX=..\release
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\nupic\bindings
	..\..

rem Build and install NuPIC.core, and build SWIG binding libraries
mingw32-make -f Makefile install

rem Install Python SWIG nupic.bindings
cd %NUPIC_CORE%
python setup.py install
```

## Running tests

After building and installing, the following can run C++ related tests;

```
cd %NUPIC_CORE%\build\release\bin
cpp_region_test
unit_tests
```

You can run the nupic.bindings tests with py.test;

```
cd %NUPIC_CORE%
py.test --pyargs nupic.bindings
```

## Build notes

The `%NUPIC_CORE%\.gitignore` file has a rule that ignores any directory called `build/` from Git. Making that directory a handy place to store build dependencies.

* You must use the `mingwpy_amd64_vc90.7z` toolchain. Other MinGW x64 GCC toolchains do not work. It also __must__ be the amd64 vc90 version to match up with the Python 2.7.9+ x64 version.
* The `libpython-cp27-none-win_amd64.7z` has to be extracted into your Python 2.7 x64 directory _before_ running cmake (two `.a` library files placed into `C:\Python27-x64\libs` directory).
* It is convenient to extract the `mingwpy_amd64_vc90.7z` into a `C:\mingwpy` directory.
* After extracting the MinGW GCC 7z archive your `PATH` environment variable will need `C:\mingwpy\bin;` adding to it.
* Make sure to copy `%NUPIC_CORE%\external\windows64-gcc\bin\distutils.cfg` into the `C:\Python27-x64\Lib\distutils` directory.
* `mingw32-make` is used with the CMake generated Makefiles.

## External libraries

The C++ Core library depends on a handful of support libraries that are distributed within the GitHub repository. The following sub-sections can be used as a guide if you require changes to these pre-built libraries.

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

Most of the libraries have a CMakeList.txt file. A CMakeList.txt file allows the use of the **CMake-GUI** application. Which can make it easy to tweak build variables, e.g. the required update of `CMAKE_INSTALL_PREFIX` for each library; and advanced options such as compiler settings. 

Install the 64 bit version of MSys2 - http://sourceforge.net/p/msys2/wiki/MSYS2%20installation/

From the MSys2 command line run the `update-core` script followed by `pacman -Su`, to make sure MSys2 is up to date.

Explicit building of the Apr-iconv library is _not_ required and can be skipped. It's referenced and should be built as part of the apr/apr-util build process. 

All support libraries are need to be built for x64. 

If a solution contains an INSTALL project, the install scripts are placed inside a file called `cmake_install.cmake` A manual install can be performed by execute the following command;  

> cmake.exe -DBUILD_TYPE=Release -P cmake_install.cmake

This implies that your `PATH` environment variable has a directory to the cmake.exe (typically "C:\Program Files (x86)\CMake\bin").

##### NumPy Python package

The C++ headers from NumPy can be installed via a pre-built NumPy Python package. For example; from http://www.lfd.uci.edu/~gohlke/pythonlibs/ or via MSys2 `pacman -S mingw64/mingw-w64-x86_64-python2-numpy` command.

##### Apache Portable Runtime (APR) libraries

Using `pacman -Ss apr` searches for APR packages. Use the following commands to install the required APR libraries; `pacman -S mingw64/mingw-w64-x86_64-apr --noconfirm` and `pacman -S mingw64/mingw-w64-x86_64-apr-util --noconfirm` 

This installs (assuming a default installation directory for MSys2) the libraries into `C:/msys64/mingw64/lib`. The two APR libraries (`libapr-1.a` and `libaprutil-1.a`) can then be copied into the `%NUPIC_CORE%\external\windows64-gcc-\lib` directory. And used with the CMake build command as described above to rebuild the core library.

##### Cap'n Proto library

This is part of the main nupic.core CMake build process. The appropriate version is git cloned, as well as the associated Cap'n Proto compiler tools. It is built using a `MinGW Makefile` and ends up in the `%NUPIC_CORE%\build\scripts\ThirdParty\Install` directories.

##### Yaml, Yaml-cpp, and Z libraries

A CMake command, or the CMake-GUI program, can be used to generate MinGW Makefiles. These can then be used with the `mingw32-make -f Makefile install` command to rebuild these libraries. Make sure that the `CMAKE_BUILD_TYPE` and `CMAKE_INSTALL_PREFIX` are set appropriately.
