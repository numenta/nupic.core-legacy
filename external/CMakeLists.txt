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

cmake_minimum_required(VERSION 2.8)
project(nupic_core_main CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set_directory_properties(PROPERTIES EP_BASE "${EP_BASE}")

# Shorter aliases for static library prefix and suffix.
set(STATIC_PRE ${CMAKE_STATIC_LIBRARY_PREFIX})
set(STATIC_SUF ${CMAKE_STATIC_LIBRARY_SUFFIX})
# This is the library prefix path.
set(LIB_PRE ${EP_BASE}/Install/lib)
# This is the include path.
set(INCLUDE_PRE ${EP_BASE}/Install/include)
# The full prefix path for binaries.
set(BIN_PRE ${EP_BASE}/Install/bin)

# Add vendored prebuilt library include paths.
list(APPEND EXTERNAL_INCLUDE_DIRS
     "${PROJECT_SOURCE_DIR}/${PLATFORM}${BITNESS}/include"
     "${PROJECT_SOURCE_DIR}/common/include")

include(ExternalProject)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

include(CapnProto)

# Convenience variable that wraps all external include directories.
list(APPEND EXTERNAL_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS})

set(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS} PARENT_SCOPE)
