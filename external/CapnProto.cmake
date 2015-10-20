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
  find_package(CapnProto)
  # Find static libraries
  find_library(LIB_KJ ${STATIC_PRE}kj${STATIC_SUF})
  find_library(LIB_CAPNP ${STATIC_PRE}capnp${STATIC_SUF})
  find_library(LIB_CAPNPC ${STATIC_PRE}capnpc${STATIC_SUF})
  set(CAPNP_LIBRARIES ${LIB_KJ} ${LIB_CAPNP} ${LIB_CAPNPC})
endif ()

if (NOT CAPNP_FOUND)
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
  set(CAPNP_LIBRARIES ${LIB_KJ} ${LIB_CAPNP} ${LIB_CAPNPC})
  set(CAPNP_INCLUDE_DIRS ${INCLUDE_PRE})
  set(CAPNP_EXECUTABLE ${BIN_PRE}/capnp)
  set(CAPNPC_CXX_EXECUTABLE ${BIN_PRE}/capnpc-c++)
else()
  # Create a dummy target to depend on.
  add_custom_target(CapnProto)
endif()

function(CREATE_CAPNPC_COMMAND
         GROUP_NAME SPEC_FILES SRC_PREFIX INCLUDE_DIR TARGET_DIR OUTPUT_FILES)
  add_custom_command(
    OUTPUT ${OUTPUT_FILES}
    COMMAND ${CAPNP_EXECUTABLE}
        compile -o ${CAPNPC_CXX_EXECUTABLE}:${TARGET_DIR}
        --src-prefix ${SRC_PREFIX} -I ${INCLUDE_DIR}
        ${SPEC_FILES}
    DEPENDS CapnProto
    COMMENT "Executing Cap'n Proto compiler"
  )
  add_custom_target(${GROUP_NAME} ALL SOURCES ${CAPNP_SPECS})
endfunction(CREATE_CAPNPC_COMMAND)

# Set the relevant variables in the parent scope.
set(CAPNP_LIBRARIES ${CAPNP_LIBRARIES} PARENT_SCOPE)
set(CAPNP_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS} PARENT_SCOPE)
set(CAPNP_EXECUTABLE ${CAPNP_EXECUTABLE} PARENT_SCOPE)
set(CAPNPC_CXX_EXECUTABLE ${CAPNPC_CXX_EXECUTABLE} PARENT_SCOPE)

# Install headers.
foreach (INCLUDE_DIR ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY ${INCLUDE_DIR}/kj
          DESTINATION include/)
  install(DIRECTORY ${INCLUDE_DIR}/capnp
          DESTINATION include/)
endforeach ()
install(FILES ${CAPNP_LIBRARIES}
        DESTINATION lib/)
