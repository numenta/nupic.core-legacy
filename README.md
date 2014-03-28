# NuPIC Core [![Build Status](https://travis-ci.org/numenta/nupic.core.png?branch=master)](https://travis-ci.org/numenta/nupic.core)

> This project is still in the process of being extracted from the [NuPIC](http://github.com/numenta/nupic) codebase as a part of our [core extraction plan](https://github.com/numenta/nupic/wiki/nupic.core-Extraction-Plan). 

**This README is incomplete!**

This repository contains the C++ source code for the Numenta Platform for Intelligent Computing ([NuPIC](http://numenta.org/nupic.html)). It will eventually contain all algorithms for NuPIC, but is currently in a transition period. For details on building NuPIC within the python environment, please see http://github.com/numenta/nupic.

## Build and test NuPIC Core:

Important notes:
 * $REPOSITORY is the current location of the repository that you downloaded from GitHub.

### Using command line

#### Configure and generate build files:

    mkdir -p $REPOSITORY/build/scripts
    cd $REPOSITORY/build/scripts
    cmake $REPOSITORY/src

#### Build:

    cd $REPOSITORY/build/scripts
    make -j3
    
> **Note**: -j3 option specify '3' as the maximum number of parallel jobs/threads that Make will use during the build in order to gain speed. However, you can increase this number depending your CPU.

#### Run the C++ tests:

    cd $REPOSITORY/build/release/bin
    htmtest
    testeverything

### Using graphical interface

#### Generate the IDE solution:

 * Open CMake executable.
 * Specify the source folder ($REPOSITORY/src).
 * Specify the build system folder ($REPOSITORY/build/scripts), ie where IDE solution will be created.
 * Click 'Generate'.
 * Choose the IDE that interest you (remember that IDE choice is limited to your OS, ie Visual Studio is available only on CMake for Windows).

#### Build:

 * Open 'nupic.core.*proj' solution file generated on $REPOSITORY/build/scripts.
 * Run 'ALL_BUILD' project from your IDE.

#### Run the C++ tests:

 * Run 'HtmTest' and 'TestEverything' projects from your IDE (check 'output' panel to see the results).
