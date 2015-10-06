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

set(STATIC_PRE ${CMAKE_STATIC_LIBRARY_PREFIX})
set(STATIC_SUF ${CMAKE_STATIC_LIBRARY_SUFFIX})
# This is the library prefix path.
set(LIB_PRE ${EP_BASE}/Install/lib)
# The full prefix path for binaries.
set(BIN_PRE ${EP_BASE}/Install/bin)

include(ExternalProject)

# First arg is variable, second is static lib, third is generic lib fallback.
find_library(lib_kj ${STATIC_PRE}kj${STATIC_SUF} kj)
find_library(lib_capnp ${STATIC_PRE}capnp${STATIC_SUF} capnp)
find_library(lib_capnpc ${STATIC_PRE}capnpc${STATIC_SUF} capnpc)

find_program(bin_capnp capnp)
find_program(bin_capnpc_cpp capnpc-c++)

if ((lib_kj STREQUAL lib_kj-NOTFOUND) OR
    (lib_capnp STREQUAL lib_capnp-NOTFOUND) OR
    (lib_canpnc STREQUAL lib_capnpc-NOTFOUND) OR
    (bin_capnp STREQUAL bin_capnp-NOTFOUND) OR
    (bin_capnpc_cpp STREQUAL bin_capnpc_cpp-NOTFOUND))
  # Build Cap'n Proto from source.
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set(CAPNP_ARGS "-DCAPNP_LITE=1")
  else()
    set(CAPNP_ARGS "")
  endif()
  ExternalProject_Add(
    CapnProto
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

  set(lib_kj ${LIB_PRE}/${STATIC_PRE}kj${STATIC_SUF} PARENT)
  set(lib_capnp ${LIB_PRE}/${STATIC_PRE}capnp${STATIC_SUF} PARENT)
  set(lib_capnpc ${LIB_PRE}/${STATIC_PRE}capnpc${STATIC_SUF} PARENT)
  set(bin_capnp ${BIN_PRE}/capnp PARENT)
  set(bin_capnpc_cpp ${BIN_PRE}/capnpc-c++ PARENT)

  # Install headers.
  install(DIRECTORY "${EP_BASE}/Install/include/capnp"
          DESTINATION include)
  install(DIRECTORY "${EP_BASE}/Install/include/kj"
          DESTINATION include)
else()
  # Create a dummy target to depend on.
  add_custom_target(CapnProto)
endif()

function(CREATE_CAPNPC_TARGET
         TARGET_NAME SPEC_FILES SRC_PREFIX INCLUDE_DIR TARGET_DIR)
  add_custom_target(
    ${TARGET_NAME}
    COMMAND ${bin_capnp}
    compile -o ${bin_capnpc_cpp}:${TARGET_DIR}
    --src-prefix ${SRC_PREFIX} -I ${INCLUDE_DIR}
        ${CAPNP_SPECS}
    DEPENDS CapnProto
    COMMENT "Executing Cap'n Proto compiler"
  )
endfunction(CREATE_CAPNPC_TARGET)
