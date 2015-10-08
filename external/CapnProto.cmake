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

option(SOURCE_CAPNP "Build Cap'n Proto from source even if it is found." OFF)

if (NOT ${SOURCE_CAPNP})
  find_library(LIB_KJ ${STATIC_PRE}kj${STATIC_SUF})
  find_library(LIB_CAPNP ${STATIC_PRE}capnp${STATIC_SUF})
  find_library(LIB_CAPNPC ${STATIC_PRE}capnpc${STATIC_SUF})

  find_path(CAPNP_INCLUDE_DIRS capnp/generated-header-support.h)

  find_program(BIN_CAPNP capnp)
  find_program(BIN_CAPNPC_CPP capnpc-c++)
endif (NOT ${SOURCE_CAPNP})

if ((NOT DEFINED LIB_KJ OR LIB_KJ STREQUAL LIB_KJ-NOTFOUND) OR
    (NOT DEFINED LIB_CAPNP OR LIB_CAPNP STREQUAL LIB_CAPNP-NOTFOUND) OR
    (NOT DEFINED LIB_CAPNPC OR LIB_CAPNPC STREQUAL LIB_CAPNPC-NOTFOUND) OR
    (NOT DEFINED CAPNP_INCLUDE_DIRS OR
         CAPNP_INCLUDE_DIRS STREQUAL CAPNP_INCLUDE_DIRS-NOTFOUND) OR
    (NOT DEFINED BIN_CAPNP OR BIN_CAPNP STREQUAL BIN_CAPNP-NOTFOUND) OR
    (NOT DEFINED BIN_CAPNPC_CPP OR
         BIN_CAPNPC_CPP STREQUAL BIN_CAPNPC_CPP-NOTFOUND))
  # Build Cap'n Proto from source.
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set(CAPNP_ARGS "-DCAPNP_LITE=1")
  else()
    set(CAPNP_ARGS "")
  endif()
  set(CAPNP_CXX_FLAGS "-fPIC -std=c++11 -m64")
  ExternalProject_Add(
    CapnProto
    GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
    GIT_TAG v0.5.2
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND}
        ${CAPNP_ARGS}
        -DCMAKE_CXX_FLAGS=${CAPNP_CXX_FLAGS}
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
        -G "${CMAKE_GENERATOR}"
        ${EP_BASE}/Source/CapnProto/c++
  )

  set(LIB_KJ ${LIB_PRE}/${STATIC_PRE}kj${STATIC_SUF})
  set(LIB_CAPNP ${LIB_PRE}/${STATIC_PRE}capnp${STATIC_SUF})
  set(LIB_CAPNPC ${LIB_PRE}/${STATIC_PRE}capnpc${STATIC_SUF})
  set(CAPNP_INCLUDE_DIRS ${INCLUDE_PRE})
  set(BIN_CAPNP ${BIN_PRE}/capnp)
  set(BIN_CAPNPC_CPP ${BIN_PRE}/capnpc-c++)
else()
  # Create a dummy target to depend on.
  add_custom_target(CapnProto)
endif()

function(CREATE_CAPNPC_TARGET
         TARGET_NAME SPEC_FILES SRC_PREFIX INCLUDE_DIR TARGET_DIR)
  add_custom_target(
    ${TARGET_NAME}
    COMMAND ${BIN_CAPNP}
        compile -o ${BIN_CAPNPC_CPP}:${TARGET_DIR}
        --src-prefix ${SRC_PREFIX} -I ${INCLUDE_DIR}
        ${CAPNP_SPECS}
    DEPENDS CapnProto
    COMMENT "Executing Cap'n Proto compiler"
  )
endfunction(CREATE_CAPNPC_TARGET)

set(CAPNP_LIBRARIES ${LIB_KJ} ${LIB_CAPNP} ${LIB_CAPNPC} PARENT_SCOPE)
set(CAPNP_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS} PARENT_SCOPE)
# These are only needed by CREATE_CAPNPC_TARGET but are evalauted in the
# caller's scope so make sure they are accessible.
set(CAPNP_EXECUTABLE ${BIN_CAPNP} PARENT_SCOPE)
set(CAPNPC_CXX_EXECUTABLE ${BIN_CAPNPC_CPP} PARENT_SCOPE)

# Install headers.
foreach (INCLUDE_DIR ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY ${INCLUDE_DIR}/kj
          DESTINATION include/)
  install(DIRECTORY ${INCLUDE_DIR}/capnp
          DESTINATION include/)
endforeach ()
install(FILES ${LIB_KJ} ${LIB_CAPNP} ${LIB_CAPNPC}
        DESTINATION lib/)
