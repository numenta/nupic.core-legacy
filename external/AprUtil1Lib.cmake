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

# Creates ExternalProject for building the aprutil-1 static library
# (apache public runtime utilities)
#
# Exports:
#   LIB_STATIC_APRUTIL1_INC_DIR: directory of installed aprutil-1 lib headers
#   LIB_STATIC_APRUTIL1_LOC: path to installed static aprutil-1 lib

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

set(APRUTILLIB_SOURCE_DIR ${REPOSITORY_DIR}/external/common/share/apr-util/apr-util-1.5.4)
set(APRUTILLIB_INSTALL_PREFIX ${EP_BASE}/Install/AprUtil1StaticLib)
set(LIB_STATIC_APRUTIL1_INC_DIR ${APRUTILLIB_INSTALL_PREFIX}/include)
set(APRUTILLIB_INSTALL_LIB_DIR ${APRUTILLIB_INSTALL_PREFIX}/lib)

# Export directory of installed aprutil-1 lib headers to parent
set(LIB_STATIC_APRUTIL1_INC_DIR ${LIB_STATIC_APRUTIL1_INC_DIR} PARENT_SCOPE)

# Export path to installed static aprutil-1 lib to parent
set(LIB_STATIC_APRUTIL1_LOC ${APRUTILLIB_INSTALL_LIB_DIR}/${STATIC_PRE}aprutil-1${STATIC_SUF} PARENT_SCOPE)

ExternalProject_Add(
    AprUtil1StaticLib
    SOURCE_DIR ${APRUTILLIB_SOURCE_DIR}
    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    # NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors in apr-1
    # on Win32 (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_C_FLAGS="-DCOM_NO_WINDOWS_H"
        -DCMAKE_INSTALL_PREFIX=${APRUTILLIB_INSTALL_PREFIX}
        -DAPR_HAS_LDAP=OFF
        -DAPU_HAVE_ODBC=OFF
        -DAPR_INCLUDE_DIR=${LIB_STATIC_APR1_INC_DIR}
        -DAPR_LIBRARIES=""
        -DTEST_STATIC_LIBS=ON
)
