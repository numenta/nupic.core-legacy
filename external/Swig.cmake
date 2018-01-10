# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# -----------------------------------------------------------------------------

# Set up Swig
#
# OUTPUT VARIABLES:
#
#   SWIG_EXECUTABLE: the path to the swig executable as defined by FindSWIG.
#   SWIG_DIR: the directory where swig is installed (.i files, etc.) as defined
#             by FindSWIG.

set(swig_path "${REPOSITORY_DIR}/external/common/src/swig-3.0.2.tar.gz")
set(pcre_path "${REPOSITORY_DIR}/external/common/src/pcre-8.37.tar.gz")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # We bundle pre-built Swig for Windows (presumably because we can't build it?)
  add_custom_target(Swig)
  set(swig_executable
      ${PROJECT_SOURCE_DIR}/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/bin/swig.exe)
  set(swig_dir ${PROJECT_SOURCE_DIR}/common/share/swig/3.0.2)
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
