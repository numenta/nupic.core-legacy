# NuPIC Core [![Build Status](https://travis-ci.org/rcrowder/nupic.core.png?branch=103-windows-build)](https://travis-ci.org/rcrowder/nupic.core) [![Build status](https://ci.appveyor.com/api/projects/status/g2vdotgyeh8nnpnn)](https://ci.appveyor.com/project/rcrowder/nupic-core) [![Coverage Status](https://coveralls.io/repos/numenta/nupic.core/badge.png?branch=103-windows-build)](https://coveralls.io/r/numenta/nupic.core?branch=103-windows-build)

This repository contains the C++ source code for the Numenta Platform for Intelligent Computing ([NuPIC](http://numenta.org/nupic.html)). It will eventually contain all algorithms for NuPIC, but is currently in a transition period. For details on building NuPIC within the python environment, please see http://github.com/numenta/nupic.

## Build and test NuPIC Core:

Important notes:
 * `$NUPIC_CORE` is the current location of the repository that you downloaded from GitHub.

### Using command line

#### Configure and generate build files:

    mkdir -p $NUPIC_CORE/build/scripts
    cd $NUPIC_CORE/build/scripts
    cmake $NUPIC_CORE/src

#### Build:

    cd $NUPIC_CORE/build/scripts
    # optionally start a fresh build
    make clean # or 'make distclean' for a complete clean
    make -j3
    
> **Note**: The `-j3` option specifies '3' as the maximum number of parallel jobs/threads that Make will use during the build in order to gain speed. However, you can increase this number depending your CPU.

#### Run the tests:

    cd $NUPIC_CORE/build/scripts
    make tests_htm 
    make tests_unit

### Using graphical interface

#### Generate the IDE solution:

 * Open CMake-GUI executable.
 * Specify the source folder (`$NUPIC_CORE/src`).
 * Specify the build system folder (`$NUPIC_CORE/build/scripts`), i.e. where IDE solution will be created.
 * Click `Configure`.
 * Choose the IDE that interest you (remember that IDE choice is limited to your OS).
 * Specify the CMake install prefix (`$NUPIC_CORE/build/release`) 
 * Click `Configure`.
 * Click `Generate`.

Visual Studio is available only via CMake for Windows. Express versions of Visual Studio will need Windows SDK.

#### Build:

 * Open `nupic_core.sln' solution file found in `$NUPIC_CORE/build/scripts`.
 * Rebuild `ALL_BUILD` project from your IDE.

This will build a static Release version of the NuPIC core library. This is then used in the test programs that are built and run. Watch the Output Window for build issues and test results.

The library also contains all external libraries that the core depends upon. The ALL_BUILD project will skip building the distclean and INSTALL projects. The library can be found in '$CMAKE_INSTALL_PREFIX/lib', with the header files in the usual install and external sub-directories.

