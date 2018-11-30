NuPIC Core external libraries
=============================

NuPIC Core depends on a number of external libraries that are distributed with
the source. The build of these libraries is integrated into the cmake-based
build of nupic.core.

These libraries are found in the nupic.core sources under external/common/share.

The C++ library is linked to other languages using SWIG, such as Python x64.

The NuPIC.Core repository consists of two parts;

- Main C++ core library ([`src`](../src))
- Python based bindings for Algorithms and Network API
  ([`py_interface`](../py_interface))

Please put any new pybind11 bindings in the [`py_interface/helpers`](../py_interface/helpers) folder.

NuPIC Core build overview
========================

The NuPIC Core C++ build generates two static nupic_core libraries:

- `%NUPIC_CORE%/release/lib/nupic_core.<staticlib-ext>` contains _only_ the core library
- `%NUPIC_CORE%/release/lib/nupic_core_py.<staticlib-ext>` contains the C++ core and pybind helper libraries

and nupic.bindings shared libraries.

Where `NUPIC_CORE` is an environment variable that points to the git cloned directory and `<staticlib-ext>` is the toolchain-specific extension for a static library.

CMake based build files are used to define the entire build process. The nupic.bindings use pybind11.

For build information on Unix platforms (e.g., Mac OS X and linux), refer to https://github.com/numenta/nupic.core#developer-installation.

