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

option(FIND_SWIG "Use preinstalled SWIG." OFF)

if (${FIND_SWIG})
  find_package(SWIG)
  # Create a dummy target to depend on.
  add_custom_target(Swig)

else ()
  # Build SWIG from source.
  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_custom_target(Swig)
    set(SWIG_EXECUTABLE
        ${PROJECT_SOURCE_DIR}/${PLATFORM}${BITNESS}/bin/swig.exe)
    set(SWIG_DIR ${PROJECT_SOURCE_DIR}/common/share/swig/3.0.2)
  else()
    # TODO: Remove dependency on curl by using native CMake functionality for
    # fetching pcre.
    ExternalProject_Add(
      Swig
      URL http://prdownloads.sourceforge.net/swig/swig-3.0.2.tar.gz
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND
        curl -OL http://downloads.sourceforge.net/project/pcre/pcre/8.37/pcre-8.37.tar.gz &&
        ${EP_BASE}/Source/Swig/Tools/pcre-build.sh &&
        ${EP_BASE}/Source/Swig/configure --prefix=${EP_BASE}/Install --enable-cpp11-testing
    )
    set(SWIG_EXECUTABLE ${EP_BASE}/Install/bin/swig)
    set(SWIG_DIR ${EP_BASE}/Install/share/swig/3.0.2)
  endif()
endif ()

set(SWIG_EXECUTABLE ${SWIG_EXECUTABLE} PARENT_SCOPE)
set(SWIG_DIR ${SWIG_DIR} PARENT_SCOPE)
