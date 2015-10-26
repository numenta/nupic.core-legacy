# NuPIC Core [![Unix-like Build Status](https://travis-ci.org/numenta/nupic.core.png?branch=master)](https://travis-ci.org/numenta/nupic.core) [![Windows Build Status](https://ci.appveyor.com/api/projects/status/px32pcil23vunml0/branch/master?svg=true)](https://ci.appveyor.com/project/numenta-ci/nupic-core/branch/master) [![Coverage Status](https://coveralls.io/repos/numenta/nupic.core/badge.png?branch=master)](https://coveralls.io/r/numenta/nupic.core?branch=master)

This repository contains the C++ source code for the Numenta Platform for Intelligent Computing ([NuPIC](http://numenta.org/nupic.html)). It will eventually contain all algorithms for NuPIC, but is currently in a transition period. For details on building NuPIC within the python environment, please see http://github.com/numenta/nupic.

## Build and test NuPIC Core:

Important notes:

 * For developers (contributing to NuPIC Core) please follow the [Development Workflow](https://github.com/numenta/nupic.core/wiki/Development-Workflow) steps.
 * `$NUPIC_CORE` is the current location of the repository that you downloaded from GitHub.
 * Platform specific Readme.md text files exist in some `external/` subdirectories
 * See the main [wiki](https://github.com/numenta/nupic.core/wiki) for more build notes

### Using command line

#### Install Dependencies

- Python - We recommend you use the system version where possibly.
    - Version 2.7
- [NumPy](http://www.numpy.org/) - Can be installed through some system package managers or via [pip](https://pip.pypa.io/)
    - Version 1.9.2
- [pycapnp](http://jparyani.github.io/pycapnp/)
    - Version 0.5.5
- [CMake](http://www.cmake.org/)

The Python depedencies (NumPy and pycapnp) can be installed with `pip`:

    pip install -r bindings/py/requirements.txt

#### Configure and generate C++ build files:

    mkdir -p $NUPIC_CORE/build/scripts
    cd $NUPIC_CORE/build/scripts
    cmake $NUPIC_CORE -DCMAKE_INSTALL_PREFIX=../release

> **Note**: If you have dependencies precompiled but not in standard system locations then you can specify where to find them with `-DCMAKE_PREFIX_PATH` (for bin/lib) and `-DCMAKE_INCLUDE_PATH` (for header files).

> **Note**: The `-DCMAKE_INSTALL_PREFIX=../release` option shown above is optional, and specifies the location where `nupic.core` should be installed. If omitted, `nupic.core` will be installed in a system location. Using this option is useful when testing versions of `nupic.core` with `nupic` (see [NuPIC's Dependency on nupic.core](https://github.com/numenta/nupic/wiki/NuPIC's-Dependency-on-nupic.core)).

#### Build:

    # While still in $NUPIC_CORE/build/scripts
    make -j3

> **Note**: The `-j3` option specifies '3' as the maximum number of parallel jobs/threads that Make will use during the build in order to gain speed. However, you can increase this number depending your CPU.

#### Install:

    # While still in $NUPIC_CORE/build/scripts
    make install

#### Run the tests:

    cd $NUPIC_CORE/build/release/bin
    ./cpp_region_test
    ./unit_tests

#### Install nupic.bindings for nupic:

    cd $NUPIC_CORE
    python setup.py install --nupic-core-dir=$NUPIC_CORE/build/release

> **Note**: set `--nupic-core-dir` to the location where `nupic.core` was installed.

If you get a gcc exit code 1, you may consider running this instead:

     CC=clang CXX=clang++ python setup.py install --user

If you are installing on Mac OS X, you must add the instruction `ARCHFLAGS="-arch x86_64"` before the python call:

    ARCHFLAGS="-arch x86_64" python setup.py install

Alternatively, you can use the `develop` command to link to Python source code in place. This is useful if you are changing Python code because you don't need to recompile between changes.

    python setup.py develop

> _Note_: If you get a "permission denied" error when using the setup commands above, you may add the `--user` flag to install to a location in your home directory, which should resolve any permissions issues. Doing this, you may need to add this location to your PATH and PYTHONPATH.

Once it is installed, you can import NuPIC bindings library to your python script using:

    import nupic.bindings

You can run the nupic.bindings tests with `py.test`:

    py.test --pyargs nupic.bindings

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

