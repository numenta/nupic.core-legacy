NuPIC Core external libraries
=============================

NuPIC Core depends on a number of external libraries. The download and build of these libraries is 
integrated into the cmake-based build of nupic.core.  The code that does this are in external/*.cmake

- Boost.cmake   If needed, finds the boost installation. Boost needs to be built with -fPIC so cannot use externally installed.
- gtest.cmake   Downloads and installs googletest 1.8.1
- pybind.cmake  Downloads and installs pybind11 2.2.3
- yaml-cpp.cmake Downloads and installs yaml-cpp 0.6.2 


