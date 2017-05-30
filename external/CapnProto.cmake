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
#
# OUTPUT VARIABLES:
#
#   CAPNP_STATIC_LIB_TARGET: name of static library target that contains all of
#                            capnproto library objects.
#
#   CAPNP_INCLUDE_DIRS
#   CAPNP_EXECUTABLE
#   CAPNPC_CXX_EXECUTABLE
#   CAPNP_CMAKE_DEFINITIONS: informational; platform-specific cmake defintions
#                            used by capnproto build
#   CAPNP_COMPILER_DEFINITIONS: list of -D compiler defintions needed by apps
#                               that are built against this library (e.g.,
#                               -DCAPNP_LITE)
#   CAPNP_BINARIES:          Binaries location
#
# EXPORTED FUNCTIONS:
#
#   CREATE_CAPNPC_COMMAND: Create a custom command that runs the capnp compiler.

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(CAPNP_STATIC_LIB_TARGET capnp-bundle)

set(capnp_lib_url
    "${REPOSITORY_DIR}/external/common/share/capnproto/capnproto-c++-0.5.3.tar.gz")
set(capnp_win32_tools_url
    "${REPOSITORY_DIR}/external/common/share/capnproto/capnproto-c++-win32-0.5.3.zip")

set(capnp_lib_kj ${LIB_PRE}/${STATIC_PRE}kj${STATIC_SUF})
set(capnp_lib_capnp ${LIB_PRE}/${STATIC_PRE}capnp${STATIC_SUF})
set(capnp_lib_capnpc ${LIB_PRE}/${STATIC_PRE}capnpc${STATIC_SUF})

set(CAPNP_INCLUDE_DIRS ${INCLUDE_PRE})
set(CAPNP_EXECUTABLE ${BIN_PRE}/capnp${CMAKE_EXECUTABLE_SUFFIX})
set(CAPNPC_CXX_EXECUTABLE ${BIN_PRE}/capnpc-c++${CMAKE_EXECUTABLE_SUFFIX})

set(CAPNP_COMPILER_DEFINITIONS)


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(CAPNP_CMAKE_DEFINITIONS -DCAPNP_LITE=1 -DEXTERNAL_CAPNP=1 -DBUILD_TOOLS=OFF)
  # NOTE nupic.core's swig wraps depend on the macro CAPNP_LITE to have a value
  set(CAPNP_COMPILER_DEFINITIONS ${CAPNP_COMPILER_DEFINITIONS} -DCAPNP_LITE=1)
  set(capnp_link_libraries ${capnp_lib_capnp} ${capnp_lib_kj})
else()
  set(CAPNP_CMAKE_DEFINITIONS -DCAPNP_LITE=0)
  set(capnp_link_libraries ${capnp_lib_capnpc} ${capnp_lib_capnp} ${capnp_lib_kj})
endif()


# NOTE Capnproto link fails with segfault on Travis and Ubuntu when using
# a combination -flto optimization together with -O2
# Reference https://github.com/sandstorm-io/capnproto/issues/300
set(capnp_cxx_flags "${EXTERNAL_CXX_FLAGS_UNOPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")
set(capnp_linker_flags "${EXTERNAL_LINKER_FLAGS_UNOPTIMIZED}")


# Print diagnostic info to debug whether -fuse-linker-plugin is being suppressed
message(STATUS "CapnProto CXX_FLAGS=${capnp_cxx_flags}")

ExternalProject_Add(CapnProto
  URL ${capnp_lib_url}

  UPDATE_COMMAND ""

  CMAKE_GENERATOR ${CMAKE_GENERATOR}

  CMAKE_ARGS
      ${CAPNP_CMAKE_DEFINITIONS}
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_TESTING=OFF
      -DCMAKE_CXX_FLAGS=${capnp_cxx_flags}
      -DCMAKE_EXE_LINKER_FLAGS=${capnp_linker_flags}
      -DCMAKE_INSTALL_PREFIX=${EP_BASE}/Install
)


# Merge capnproto-generated static libraries into a single static library.
# This creates an `add_library` static library target that serves as the
# abstraction to all of capnproto library objects
merge_static_libraries(${CAPNP_STATIC_LIB_TARGET} "${capnp_link_libraries}")
add_dependencies(${CAPNP_STATIC_LIB_TARGET} CapnProto)


if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  # Install prebuilt Cap'n Proto compilers for Windows
  ExternalProject_Add(CapnProtoTools
    DEPENDS CapnProto

    URL ${capnp_win32_tools_url}

    CONFIGURE_COMMAND ""
    BUILD_COMMAND
      ${CMAKE_COMMAND} -E make_directory ${BIN_PRE}
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory
        "<SOURCE_DIR>/capnproto-tools-win32-0.5.3"
        ${BIN_PRE}
  )
endif()


function(CREATE_CAPNPC_COMMAND
         SPEC_FILES SRC_PREFIX INCLUDE_DIR TARGET_DIR OUTPUT_FILES)
  # Create a custom command that runs the capnp compiler on ${SPEC_FILES} and
  # generates ${OUTPUT_FILES} in directory ${TARGET_DIR}

  set(dependencies ${SPEC_FILES} CapnProto)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    list(APPEND dependencies CapnProtoTools)
  endif()

  add_custom_command(
    OUTPUT ${OUTPUT_FILES}
    COMMAND ${CAPNP_EXECUTABLE}
        compile -o ${CAPNPC_CXX_EXECUTABLE}:${TARGET_DIR}
        --src-prefix ${SRC_PREFIX} -I ${INCLUDE_DIR}
        ${SPEC_FILES}
    DEPENDS ${dependencies}
    COMMENT "Executing Cap'n Proto compiler"
  )
endfunction(CREATE_CAPNPC_COMMAND)

# Set the relevant variables in the parent scope.
set(CAPNP_STATIC_LIB_TARGET ${CAPNP_STATIC_LIB_TARGET} PARENT_SCOPE)
set(CAPNP_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS} PARENT_SCOPE)
set(CAPNP_EXECUTABLE ${CAPNP_EXECUTABLE} PARENT_SCOPE)
set(CAPNPC_CXX_EXECUTABLE ${CAPNPC_CXX_EXECUTABLE} PARENT_SCOPE)
set(CAPNP_CMAKE_DEFINITIONS ${CAPNP_CMAKE_DEFINITIONS} PARENT_SCOPE)
set(CAPNP_COMPILER_DEFINITIONS ${CAPNP_COMPILER_DEFINITIONS} PARENT_SCOPE)
set(CAPNP_BINARIES ${BIN_PRE} PARENT_SCOPE)

## Install headers and libraries.
## TODO It's confusing that these same INC and LIB installation steps are duplicated
##      in src/CMakeLists.txt
#foreach (INCLUDE_DIR ${CAPNP_INCLUDE_DIRS})
#  install(DIRECTORY ${INCLUDE_DIR}/kj
#          DESTINATION include/)
#  install(DIRECTORY ${INCLUDE_DIR}/capnp
#          DESTINATION include/)
#endforeach ()
