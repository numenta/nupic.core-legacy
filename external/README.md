HTM Core external libraries
=============================

HTM Core depends on a number of external libraries. The download and build of these libraries is
integrated into the cmake-based build of htm.core.  The code that does this are in external/*.cmake

- Boost.cmake   If needed, finds the boost installation 1.69.0. Boost needs to be built with -fPIC so cannot use externally installed.
- gtest.cmake   Downloads and installs googletest 1.8.1
- pybind.cmake  Downloads and installs pybind11 2.2.4  (header only)
- yaml-cpp.cmake Downloads and installs yaml-cpp master (something wrong with release 0.6.2)
- eigen cmake   Downloads eigen 3.3.7  (header only)
- mnist.zip     Downloads the mnist data set from repository master.

External packages included within this repository are built into the common library.
This includes:
- MurmurHash3

