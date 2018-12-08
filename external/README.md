NuPIC Core external libraries
=============================

NuPIC Core depends on a number of external libraries. The download and build of these libraries is 
integrated into the cmake-based build of nupic.core.  The code that does this are in external/*.cmake

- Boost.cmake   If needed, finds the boost installation.  Boost must already be installed with system and filesystem modules.
- gtest.cmake   Downloads and installs googletest 1.8.1
- pybind.cmake  Downloads and installs pybind11 2.2.3
- yaml-cpp.cmake Unwraps and installs yaml-cpp 0.6.2 from included file in external/common/share


