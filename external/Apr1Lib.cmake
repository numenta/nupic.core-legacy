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

set(APRLIB_INSTALL_PREFIX ${EP_BASE}/Install/Apr1StaticLib)
set(APRLIB_INSTALL_LIB_DIR ${APRLIB_INSTALL_PREFIX}/lib)
set(LIB_STATIC_APR1_INC_DIR ${APRLIB_INSTALL_PREFIX}/include)
set(LIB_STATIC_APR1_LOC ${APRLIB_INSTALL_LIB_DIR}/${STATIC_PRE}apr-1${STATIC_SUF})

# Export directory of installed apr-1 lib headers to parent
set(LIB_STATIC_APR1_INC_DIR ${LIB_STATIC_APR1_INC_DIR} PARENT_SCOPE)

# Export path to installed static apr-1 lib to parent
set(LIB_STATIC_APR1_LOC ${LIB_STATIC_APR1_LOC} PARENT_SCOPE)

# NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors on Win32
# (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
set(APRLIB_CFLAGS "-DCOM_NO_WINDOWS_H")

if (UNIX)
    set(APRLIB_CONFIG_OPTIONS
        "--enable-static"
        "--disable-shared"
        "--disable-ipv6")

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(APRLIB_CONFIG_OPTIONS
            ${APRLIB_CONFIG_OPTIONS}
            "--enable-debug")
    endif()


    ExternalProject_Add(Apr1StaticLib
        URL ${REPOSITORY_DIR}/external/common/share/apr/unix/apr-1.5.2.tar.gz
        UPDATE_COMMAND ""

        CONFIGURE_COMMAND
            ${EP_BASE}/Source/Apr1StaticLib/configure
                --prefix=${APRLIB_INSTALL_PREFIX}
                ${APRLIB_CONFIG_OPTIONS}
                CFLAGS=${APRLIB_CFLAGS}

        BUILD_COMMAND
            make -f Makefile all

        INSTALL_COMMAND
            make -f Makefile install
    )

    ExternalProject_Add_Step(Apr1StaticLib unix_post_install
        COMMENT "Unix/Linux/MacOS Apr1StaticLib install completed"
        DEPENDEES install
        ALWAYS 1

        COMMAND echo listing ${LIB_STATIC_APR1_INC_DIR} COMMAND ls ${LIB_STATIC_APR1_INC_DIR}
        COMMAND echo listing ${LIB_STATIC_APR1_INC_DIR}/apr-1 COMMAND ls ${LIB_STATIC_APR1_INC_DIR}/apr-1
        COMMAND echo listing ${LIB_STATIC_APR1_LOC} COMMAND ls ${LIB_STATIC_APR1_LOC}
    )

else()
    # NOT UNIX - i.e., Windows

    set(APRLIB_SOURCE_DIR ${REPOSITORY_DIR}/external/common/share/apr/win/apr-1.5.2)

    ExternalProject_Add(Apr1StaticLib
        SOURCE_DIR ${APRLIB_SOURCE_DIR}
        UPDATE_COMMAND ""

        CMAKE_GENERATOR ${CMAKE_GENERATOR}

        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=OFF
            -DAPR_HAVE_IPV6=OFF
            -DCMAKE_C_FLAGS=${APRLIB_CFLAGS}
            -DCMAKE_INSTALL_PREFIX=${APRLIB_INSTALL_PREFIX}

        INSTALL_COMMAND
            mingw32-make -f Makefile install
    )

    set(LIST_TOP_APR1_INC_DIR_CMD "dir ${LIB_STATIC_APR1_INC_DIR}")
    set(LIST_INNER_APR1_INC_DIR_CMD "dir ${LIB_STATIC_APR1_INC_DIR}/apr-1")
    set(LIST_LIB_STATIC_APR1_LOC_CMD "dir ${LIB_STATIC_APR1_LOC}")

    ExternalProject_Add_Step(Apr1StaticLib windows_post_build
        COMMENT "Windows Apr1StaticLib build completed"
        DEPENDEES build
        ALWAYS 1

        COMMAND echo "Executing ${LIST_TOP_APR1_INC_DIR_CMD}"
        #COMMAND ${LIST_TOP_APR1_INC_DIR_CMD}
        COMMAND echo "Executing ${LIST_INNER_APR1_INC_DIR_CMD}"
        #COMMAND ${LIST_INNER_APR1_INC_DIR_CMD}
        COMMAND echo "Executing ${LIST_LIB_STATIC_APR1_LOC_CMD}"
        #COMMAND ${LIST_LIB_STATIC_APR1_LOC_CMD}
    )
endif()