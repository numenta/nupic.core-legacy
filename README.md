# NuPIC Core [![Build Status](https://travis-ci.org/numenta/nupic.core.png?branch=master)](https://travis-ci.org/numenta/nupic.core) [![Coverage Status](https://coveralls.io/repos/numenta/nupic.core/badge.png?branch=master)](https://coveralls.io/r/numenta/nupic.core?branch=master)

This repository contains the C++ source code for the Numenta Platform for Intelligent Computing ([NuPIC](http://numenta.org/nupic.html)). It will eventually contain all algorithms for NuPIC, but is currently in a transition period. For details on building NuPIC within the python environment, please see http://github.com/numenta/nupic.

## Build and test NuPIC Core:

Important notes:
 * For developers (contributing to NuPIC Core) please follow the [C++ Development Workflow steps](https://github.com/numenta/nupic.core/wiki/C++-Development-Workflow).
 * `$NUPIC_CORE` is the current location of the repository that you downloaded from GitHub.

### Using command line

#### Configure and generate build files:

The commands below assume that `CXX` is the C++ compiler. It should be the name of the C++ compiler you wish to use to compile NuPIC Core, like `g++` or `clang++`.

    mkdir -p $NUPIC_CORE/build/scripts
    cd $NUPIC_CORE/build/scripts
    cmake $NUPIC_CORE/src -DCMAKE_CXX_COMPILER="${CXX}" [-DCMAKE_INSTALL_PREFIX=../release]

> **Note**: The `-DCMAKE_INSTALL_PREFIX=../release` option shown above is optional, and specifies the location where `nupic.core` should be installed. If omitted, `nupic.core` will be installed in a system location. Using this option is useful when testing versions of `nupic.core` with `nupic` (see [NuPIC's Dependency on nupic.core](https://github.com/numenta/nupic/wiki/NuPIC's-Dependency-on-nupic.core)).

#### Build:

    cd $NUPIC_CORE/build/scripts
    # optionally start a fresh build
    make clean
    make -j3
    
> **Note**: The `-j3` option specifies '3' as the maximum number of parallel jobs/threads that Make will use during the build in order to gain speed. However, you can increase this number depending your CPU.

#### Install:

    make install

#### Run the tests:

    cd $NUPIC_CORE/build/scripts
    make tests_cpp_region
    make tests_unit

### Using graphical interface

#### Generate the IDE solution:

 * Open CMake executable.
 * Specify the source folder (`$NUPIC_CORE/src`).
 * Specify the build system folder (`$NUPIC_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Generate`.
 * Choose the IDE that interest you (remember that IDE choice is limited to your OS, i.e. Visual Studio is available only on CMake for Windows).

#### Build:

 * Open `nupic_core.*proj` solution file generated on `$NUPIC_CORE/build/scripts`.
 * Run `ALL_BUILD` project from your IDE.

#### Run the tests:

 * Run any `tests_*` project from your IDE (check `output` panel to see the results).

