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
include(ExternalProject)

# Build Cap'n Proto first so it can be used to generate C++ from .capnp files.
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(CAPNP_ARGS "-DCAPNP_LITE=1")
else()
  set(CAPNP_ARGS "")
endif()
ExternalProject_Add(
  capnproto
  GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  GIT_TAG v0.5.2
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      ${CAPNP_ARGS}
      -DBUILD_TESTING=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
      -G "${CMAKE_GENERATOR}"
      ${EP_BASE}/Source/capnproto/c++
)

add_library(kj STATIC IMPORTED GLOBAL)
set(libkj ${CMAKE_STATIC_LIBRARY_PREFIX}kj${CMAKE_STATIC_LIBRARY_SUFFIX})
set_target_properties(kj PROPERTIES
                      IMPORTED_LOCATION ${EP_BASE}/Install/lib/${libkj})
add_dependencies(kj capnproto)

set(libcapnp ${CMAKE_STATIC_LIBRARY_PREFIX}capnp${CMAKE_STATIC_LIBRARY_SUFFIX})
add_library(capnp STATIC IMPORTED GLOBAL)
set_target_properties(capnp PROPERTIES
                      IMPORTED_LOCATION ${EP_BASE}/Install/lib/${libcapnp})
add_dependencies(capnp capnproto)

set(libcapnpc ${CMAKE_STATIC_LIBRARY_PREFIX}capnpc${CMAKE_STATIC_LIBRARY_SUFFIX})
add_library(capnpc STATIC IMPORTED GLOBAL)
set_target_properties(capnpc PROPERTIES
                      IMPORTED_LOCATION ${EP_BASE}/Install/lib/${libcapnpc})
add_dependencies(capnpc capnproto)

add_executable(bin_capnp IMPORTED GLOBAL)
set_target_properties(bin_capnp PROPERTIES
                      IMPORTED_LOCATION ${EP_BASE}/Install/bin/capnp
                      OUTPUT_NAME capnp)
add_dependencies(bin_capnp capnproto)

add_executable(bin_capnpc_cpp IMPORTED GLOBAL)
set_target_properties(bin_capnpc_cpp PROPERTIES
                      IMPORTED_LOCATION ${EP_BASE}/Install/bin/capnpc-c++
                      OUTPUT_NAME capnpc-c++)
add_dependencies(bin_capnpc_cpp capnproto)

# Install headers.
install(DIRECTORY "${EP_BASE}/Install/include/capnp"
        DESTINATION include)
install(DIRECTORY "${EP_BASE}/Install/include/kj"
        DESTINATION include)
