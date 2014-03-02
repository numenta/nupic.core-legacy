<div align="center">
    <img title="Numenta Logo" src="http://numenta.org/images/250x250numentaicon.gif"/>
</div>

# Core Algorithms of the Numenta Platform for Intelligent Computing (NuPIC.Core)

[![Build Status](https://travis-ci.org/numenta/nupic.core.png?branch=master)](https://travis-ci.org/numenta/nupic.core)

NuPIC.Core is a library that contains the Cortical Learning Algorithm (CLA) which is the core of the Numenta Platform for Intelligent Computing (NuPIC).

For more information, see [numenta.org](http://numenta.org) or the [Github wiki](https://github.com/numenta/nupic/wiki).

Issue tracker at [issues.numenta.org](https://issues.numenta.org/browse/NPC).

## Installation

For all installation options, see the [Getting Started](https://github.com/numenta/nupic/wiki/Getting-Started) wiki page.

Currently supported platforms:
 * Linux (32/64bit)
 * Mac OSX
 * Raspberry Pi (ARMv6)
 * Chromebook (Ubuntu ARM, Crouton) (ARMv7)

Dependencies:
 * GCC (4.6-4.8), or Clang
 * Make or any IDE supported by CMake (Visual Studio, Eclipse, XCode, KDevelop, etc)

The dependencies are included in platform-specific repositories for convenience:

* [nupic-linux64](https://github.com/numenta/nupic-linux64) for 64-bit Linux systems
* [nupic-darwin64](https://github.com/numenta/nupic-darwin64) for 64-bit OS X systems

## Build and test NuPIC.Core:

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
    testeverything

### Using graphical interface

#### Generate the IDE solution:

 * Open CMake executable.
 * Specify the source folder ($REPOSITORY/src).
 * Specify the build system folder ($REPOSITORY/build/scripts), ie where IDE solution will be created.
 * Click 'Generate'.
 * Choose the IDE that interest you (remember that IDE choice is limited to your OS, ie Visual Studio is available only on CMake for Windows).

#### Build:

 * Open 'Nupic.*proj' solution file generated on $REPOSITORY/build/scripts.
 * Run 'ALL_BUILD' project from your IDE.

#### Run the tests:

 * Run 'TestEverything' project from your IDE (check 'output' panel to see the results).