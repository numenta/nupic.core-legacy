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
project(nupic_core_main)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(EP_PREFIX ${CMAKE_BINARY_DIR})
set_directory_properties(
  PROPERTIES EP_PREFIX "${EP_PREFIX}"
)

include(ExternalProject)

# Build Cap'n Proto first so it can be used to generate C++ from .capnp files.
ExternalProject_Add(
  capnproto
  GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  GIT_TAG v0.5.2
  UPDATE_COMMAND ""
  #INSTALL_COMMAND ""
  CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_TESTING=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_INSTALL_PREFIX=${EP_PREFIX}/src/externals-install
      -G "Unix Makefiles"
      ${EP_PREFIX}/src/capnproto/c++
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${EP_PREFIX}/src/capnproto/c++/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${EP_PREFIX}/src/externals-install)

# Now build nupic_core project
ExternalProject_Add(
  nupic_core
  DEPENDS capnproto
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
  CMAKE_ARGS
      -DEP_PREFIX=${EP_PREFIX}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
)
