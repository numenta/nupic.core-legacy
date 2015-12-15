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

option(FIND_CAPNP "Use preinstalled Cap'n Proto." OFF)

if (${FIND_CAPNP})
  find_package(CapnProto)
  # Most CAPNP* variables are set correctly but make sure we have the
  # static libraries.
  find_library(LIB_KJ ${STATIC_PRE}kj${STATIC_SUF})
  find_library(LIB_CAPNP ${STATIC_PRE}capnp${STATIC_SUF})
  find_library(LIB_CAPNPC ${STATIC_PRE}capnpc${STATIC_SUF})
  set(CAPNP_LIBRARIES ${LIB_CAPNPC} ${LIB_CAPNP} ${LIB_KJ})
  set(CAPNP_LIBRARIES_LITE ${LIB_CAPNP} ${LIB_KJ})

  # Create a dummy target to depend on.
  add_custom_target(CapnProto)

else ()
  # Build Cap'n Proto from source.
  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(CAPNP_DEFINITIONS "-DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1")
    set(CAPNP_CXX_FLAGS "-m${BITNESS}")
  else()
    set(CAPNP_DEFINITIONS "-DCAPNP_LITE=0")
    set(CAPNP_CXX_FLAGS "-fPIC -m${BITNESS}")
  endif()
  ExternalProject_Add(
    CapnProto
    GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
    GIT_TAG v0.5.3
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND}
        ${CAPNP_DEFINITIONS}
        -DCMAKE_CXX_FLAGS=${CAPNP_CXX_FLAGS}
        -DBUILD_TESTING=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
        -G "${CMAKE_GENERATOR}"
        ${EP_BASE}/Source/CapnProto/c++
  )

  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    # Download and install Cap'n Proto compilers
    ExternalProject_Add(
      CapnProtoTools
      DEPENDS CapnProto
      URL https://capnproto.org/capnproto-c++-win32-0.5.3.zip
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        ${CMAKE_COMMAND} -E make_directory ${BIN_PRE}
      INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_directory
          "${EP_BASE}/Source/CapnProtoTools/capnproto-tools-win32-0.5.3"
          ${BIN_PRE}
    )
  endif()

  set(LIB_KJ ${LIB_PRE}/${STATIC_PRE}kj${STATIC_SUF})
  set(LIB_CAPNP ${LIB_PRE}/${STATIC_PRE}capnp${STATIC_SUF})
  set(LIB_CAPNPC ${LIB_PRE}/${STATIC_PRE}capnpc${STATIC_SUF})
  set(CAPNP_LIBRARIES ${LIB_CAPNPC} ${LIB_CAPNP} ${LIB_KJ})
  set(CAPNP_LIBRARIES_LITE ${LIB_CAPNP} ${LIB_KJ})
  set(CAPNP_INCLUDE_DIRS ${INCLUDE_PRE})
  set(CAPNP_EXECUTABLE ${BIN_PRE}/capnp${CMAKE_EXECUTABLE_SUFFIX})
  set(CAPNPC_CXX_EXECUTABLE ${BIN_PRE}/capnpc-c++${CMAKE_EXECUTABLE_SUFFIX})
endif ()

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
set(CAPNP_LIBRARIES_LITE ${CAPNP_LIBRARIES_LITE} PARENT_SCOPE)
set(CAPNP_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS} PARENT_SCOPE)
set(CAPNP_EXECUTABLE ${CAPNP_EXECUTABLE} PARENT_SCOPE)
set(CAPNPC_CXX_EXECUTABLE ${CAPNPC_CXX_EXECUTABLE} PARENT_SCOPE)
set(CAPNP_DEFINITIONS ${CAPNP_DEFINITIONS} PARENT_SCOPE)

# Install headers and libraries.
foreach (INCLUDE_DIR ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY ${INCLUDE_DIR}/kj
          DESTINATION include/)
  install(DIRECTORY ${INCLUDE_DIR}/capnp
          DESTINATION include/)
endforeach ()
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  install(FILES ${CAPNP_LIBRARIES_LITE}
          DESTINATION lib/)
else()
  install(FILES ${CAPNP_LIBRARIES}
          DESTINATION lib/)
endif()
