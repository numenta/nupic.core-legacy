# Copyright 2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Set up Swig
#
# OUTPUT VARIABLES:
#
#   SWIG_EXECUTABLE: the path to the swig executable as defined by FindSWIG.
#   SWIG_DIR: the directory where swig is installed (.i files, etc.) as defined
#             by FindSWIG.

set(swig_path "${REPOSITORY_DIR}/external/common/src/swig-3.0.2.tar.gz")
set(swigwin_path "${REPOSITORY_DIR}/external/common/src/swigwin-3.0.2.zip")
set(pcre_path "${REPOSITORY_DIR}/external/common/src/pcre-8.37.tar.gz")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  ExternalProject_Add(
    Swig
    URL ${swigwin_path}
    SOURCE_DIR "${EP_BASE}/Install/swig"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  set(swig_executable ${EP_BASE}/Install/swig/swig.exe)
  set(swig_dir ${EP_BASE}/Install/swig/Lib)  
else()
  # Build Swig from source on non-Windows (e.g., Linux and OS X)
  ExternalProject_Add(
    Swig
    URL ${swig_path}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
      mkdir -p ${EP_BASE}/Source/Swig/Tools/ &&
      cp ${pcre_path} ${EP_BASE}/Build/Swig/ &&
      ${EP_BASE}/Source/Swig/Tools/pcre-build.sh &&
      ${EP_BASE}/Source/Swig/configure --prefix=${EP_BASE}/Install --disable-ccache --enable-cpp11-testing
  )
  set(swig_executable ${EP_BASE}/Install/bin/swig)
  set(swig_dir ${EP_BASE}/Install/share/swig/3.0.2)
endif()

set(SWIG_EXECUTABLE ${swig_executable} PARENT_SCOPE)
set(SWIG_DIR ${swig_dir} PARENT_SCOPE)
