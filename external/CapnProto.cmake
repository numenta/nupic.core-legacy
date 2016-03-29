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

# Build Cap'n Proto from source.
set(capnprotolib_url "${REPOSITORY_DIR}/external/common/share/capnproto/capnproto-c++-0.5.3.tar.gz")
set(capnproto_win32_tools_url "${REPOSITORY_DIR}/external/common/share/capnproto/capnproto-c++-win32-0.5.3.zip")

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(CAPNP_DEFINITIONS "-DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1")
else()
  set(CAPNP_DEFINITIONS "-DCAPNP_LITE=0")
endif()

set(capnp_c_flags "${COMMON_C_FLAGS} ${COMMON_COMPILER_DEFINITIONS_STR}")
set(capnp_cxx_flags "${COMMON_CXX_FLAGS} ${COMMON_COMPILER_DEFINITIONS_STR}")

ExternalProject_Add(CapnProto
  #GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  #GIT_TAG v0.5.3
  URL ${capnprotolib_url}

  UPDATE_COMMAND ""

  CMAKE_GENERATOR ${CMAKE_GENERATOR}

  CMAKE_ARGS
      ${CAPNP_DEFINITIONS}
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_TESTING=OFF
      -DCMAKE_C_FLAGS=${capnp_c_flags}
      -DCMAKE_CXX_FLAGS=${capnp_cxx_flags}
      -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # Install prebuilt Cap'n Proto compilers for Windows
  ExternalProject_Add(CapnProtoTools
    DEPENDS CapnProto

    #URL https://capnproto.org/capnproto-c++-win32-0.5.3.zip
    URL ${capnproto_win32_tools_url}

    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      ${CMAKE_COMMAND} -E make_directory ${BIN_PRE}
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory
        "<SOURCE_DIR>/capnproto-tools-win32-0.5.3"
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
