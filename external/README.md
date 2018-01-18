NuPIC Core external libraries
=============================

NuPIC Core depends on a number of external libraries that are distributed with
the source. The build of these libraries is integrated into the cmake-based
build of nupic.core.

These libraries are found in the nupic.core sources under external/common/share.

The C++ library is linked to other languages using SWIG, such as Python x64.

The NuPIC.Core repository consists of two parts;

- Main C++ core library ([`src`](../src))
- Python SWIG based bindings for Algorithms and Network API
  ([`bindings/py`](../bindings/py))

Please put any new SWIG bindings in the [`bindings`](../bindings) folder named by
language.

NuPIC Core build overview
========================

The NuPIC Core C++ build generates two static nupic_core libraries:

- `%NUPIC_CORE%/release/lib/nupic_core_solo.<staticlib-ext>` contains _only_ the core library
- `%NUPIC_CORE%/release/lib/nupic_core.<staticlib-ext>` contains the C++ core and external support libraries

and nupic.bindings shared libraries.

Where `NUPIC_CORE` is an environment variable that points to the git cloned directory and `<staticlib-ext>` is the toolchain-specific extension for a static library.

CMake based build files are used to define the entire build process. The nupic.bindings SWIG side uses Python distutil and setuptools.

For build information on Unix platforms (e.g., Mac OS X and linux), refer to https://github.com/numenta/nupic.core#developer-installation.

Platform- and toolchain-specific information may be found in README.md files located in platform/toolchain-specific subdirectories under nupic.core/external. E.g., the nupic.core/external/windows64-gcc directory contains information specific to Windows 64-bit builds using the mingwpy "gcc" toolchain.
