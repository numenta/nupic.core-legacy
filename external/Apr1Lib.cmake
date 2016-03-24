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

# Creates ExternalProject for building the apr-1 static library
# (apache public runtime)
#
# Exports:
#   LIB_STATIC_APR1_INC_DIR: directory of installed apr-1 lib headers
#   LIB_STATIC_APR1_LOC: path to installed static apr-1 lib

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

set(APRLIB_SOURCE_DIR ${REPOSITORY_DIR}/external/common/share/apr/apr-1.5.2)
set(APRLIB_INSTALL_PREFIX ${EP_BASE}/Install/Apr1StaticLib)
set(LIB_STATIC_APR1_INC_DIR ${APRLIB_INSTALL_PREFIX}/include)
set(APRLIB_INSTALL_LIB_DIR ${APRLIB_INSTALL_PREFIX}/lib)

# Export directory of installed apr-1 lib headers to parent
set(LIB_STATIC_APR1_INC_DIR ${LIB_STATIC_APR1_INC_DIR} PARENT_SCOPE)

# Export path to installed static apr-1 lib to parent
set(LIB_STATIC_APR1_LOC ${APRLIB_INSTALL_LIB_DIR}/${STATIC_PRE}apr-1${STATIC_SUF} PARENT_SCOPE)

ExternalProject_Add(Apr1StaticLib
    SOURCE_DIR ${APRLIB_SOURCE_DIR}
    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    # NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors on Win32
    # (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        #-DBUILD_SHARED_LIBS=OFF
        -DCMAKE_C_FLAGS="-DCOM_NO_WINDOWS_H"
        -DCMAKE_INSTALL_PREFIX=${APRLIB_INSTALL_PREFIX}
)
