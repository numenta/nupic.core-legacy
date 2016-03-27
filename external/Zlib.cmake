# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

# Creates ExternalProject for building z lib static library
#
# Exports:
#   LIB_STATIC_Z_INC_DIR: directory of installed z lib headers
#   LIB_STATIC_Z_LOC: path to installed static z lib

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

set(ZLIB_SOURCE_DIR "${REPOSITORY_DIR}/external/common/share/zlib/zlib-1.2.8")
set(ZLIB_INSTALL_PREFIX "${EP_BASE}/Install/ZStaticLib")
set(ZLIB_INSTALL_INC_DIR "${ZLIB_INSTALL_PREFIX}/include")
set(ZLIB_INSTALL_LIB_DIR "${ZLIB_INSTALL_PREFIX}/lib")

if(UNIX)
    # On unix-like platforms the library is almost always called libz
   set(ZLIB_OUTPUT_ROOT z)
else()
   set(ZLIB_OUTPUT_ROOT zlibstatic)
endif()


# Export directory of installed z lib headers to parent
set(LIB_STATIC_Z_INC_DIR "${ZLIB_INSTALL_INC_DIR}")
set(LIB_STATIC_Z_INC_DIR "${LIB_STATIC_Z_INC_DIR}" PARENT_SCOPE)

# Export path to installed static z lib to parent
set(LIB_STATIC_Z_LOC "${ZLIB_INSTALL_LIB_DIR}/${STATIC_PRE}${ZLIB_OUTPUT_ROOT}${STATIC_SUF}" PARENT_SCOPE)

set(C_FLAGS "${COMMON_C_FLAGS} ${COMMON_COMPILER_DEFINITIONS_STR}")

set(LINKER_FLAGS ${COMMON_LINK_FLAGS})

ExternalProject_Add(ZStaticLib
    SOURCE_DIR ${ZLIB_SOURCE_DIR}
    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_C_FLAGS=${C_FLAGS}
        -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_PREFIX}
        #-DCMAKE_STATIC_LINKER_FLAGS=${LINKER_FLAGS}
        -DINSTALL_BIN_DIR=${ZLIB_INSTALL_PREFIX}/bin
        -DINSTALL_INC_DIR=${ZLIB_INSTALL_INC_DIR}
        -DINSTALL_LIB_DIR=${ZLIB_INSTALL_LIB_DIR}
        -DINSTALL_MAN_DIR=${ZLIB_INSTALL_PREFIX}/man
        -DINSTALL_PKGCONFIG_DIR=${ZLIB_INSTALL_PREFIX}/pkgconfig

)
