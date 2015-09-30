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

set(EP_BASE ${CMAKE_BINARY_DIR})
set_directory_properties(
  PROPERTIES EP_BASE "${EP_BASE}"
)

include(ExternalProject)

# Build Cap'n Proto first so it can be used to generate C++ from .capnp files.
ExternalProject_Add(
  capnproto
  GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  GIT_TAG v0.5.2
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_TESTING=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
      -G "${CMAKE_GENERATOR}"
      ${EP_BASE}/Source/capnproto/c++
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${EP_BASE}/Source/capnproto/c++/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${EP_BASE}/Install)

# Now build nupic_core project
ExternalProject_Add(
  nupic_core
  DEPENDS capnproto
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  # Don't install nupic_core during build.
  INSTALL_COMMAND ""
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
  CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
      -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
)

# Include the nupic_core install in the top level install step.
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" --build ${EP_BASE}/Build/nupic_core --target install)")
