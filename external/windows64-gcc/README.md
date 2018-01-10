# Introduction

The NuPIC Core C++ library can be built with MinGWPy GCC or Microsoft (MS) Visual C compilers. The Python nupic.bindings can **ONLY** be built with the MinGWPy GCC compiler. To use the C++ library with the [NuPIC Python](https://github.com/numenta/nupic) project you must build the library with the MinGW-Py GCC tools, as described below.

Refer to nupic.core/external/README.md for an overview of NuPIC Core library build and dependencies.

CMake based build files are used to define the entire build process. The nupic.bindings SWIG side uses Python distutil and setuptools. The [CMake-GUI](http://www.cmake.org/) application _can_ be used to generate MinGW Makefiles.

The `%NUPIC_CORE%\appveyor.yml` script file shows how the AppVeyor automated build system progresses through building, packaging, and deployment using MinGWPy GCC.

If you need to make changes to the nupic.core C++ code and/or Python SWIG bindings, follow the remaining sections to see how to rebuild it.

## Support applications

The following applications are required when rebuilding the core C++ library, Python SWIG bindings, and if required the external support libraries;

- [CMake](http://www.cmake.org/) - version 3.1+
- [Python 2.7.9+](https://www.python.org/downloads/windows/) - x86-64 version
- [MinGW GCC for Python](`%PYTHONHOME%\\Scripts\\pip.exe install -i https://pypi.anaconda.org/carlkl/simple mingwpy`)
- [NumPy C++ headers](`%PYTHONHOME%\\Scripts\\pip.exe install numpy==1.11.2`)

## Rebuilding with CMake

The following table shows example CMake common settings;

<center>

| Name | Value |
|:---- |:----- |
| Source code | `%NUPIC_CORE%` |
| Binaries | `%NUPIC_CORE%/build/scripts` |
| CMAKE_INSTALL_PREFIX | `%NUPIC_CORE%/build/release` |
| PY_EXTENSIONS_DIR | `%NUPIC_CORE%/bindings/py/src/nupic/bindings` |

</center>

For rebuilding nupic.core and nupic.bindings you need to use the cmake `MinGW Makefiles` generator. For example;

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
	-DPY_EXTENSIONS_DIR=..\..\bindings\py\src\nupic\bindings
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

* The toolchain __must__ be the amd64 vc90 version to match up with the Python 2.7.9+ x64 version. Install the mingwpy python toolchain via
 `%PYTHONHOME%\\Scripts\\pip.exe install -i https://pypi.anaconda.org/carlkl/simple mingwpy`. Other MinGW x64 GCC toolchains do not work.
* Make sure to copy `%NUPIC_CORE%\external\windows64-gcc\bin\distutils.cfg` into the `C:\Python27-x64\Lib\distutils` directory.
* `mingw32-make` is used with the CMake generated Makefiles.

Your `PATH` environment variable must include a directory of the cmake.exe (typically "C:\Program Files (x86)\CMake\bin") tool.


##### NumPy Python package

The C++ headers from NumPy can be installed via a pre-built NumPy Python package. For example:
> `%PYTHONHOME%\\Scripts\\pip.exe install numpy==1.11.2`.
