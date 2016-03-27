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

set(APRLIB_INSTALL_PREFIX "${EP_BASE}/Install/Apr1StaticLib")
set(APRLIB_INSTALL_LIB_DIR "${APRLIB_INSTALL_PREFIX}/lib")
set(LIB_STATIC_APR1_INC_DIR "${APRLIB_INSTALL_PREFIX}/include")
set(LIB_STATIC_APR1_LOC "${APRLIB_INSTALL_LIB_DIR}/${STATIC_PRE}apr-1${STATIC_SUF}")

# Export directory of installed apr-1 lib headers to parent
set(LIB_STATIC_APR1_INC_DIR "${LIB_STATIC_APR1_INC_DIR}" PARENT_SCOPE)

# Export path to installed static apr-1 lib to parent
set(LIB_STATIC_APR1_LOC "${LIB_STATIC_APR1_LOC}" PARENT_SCOPE)

# NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors on Win32
# (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
set(APRLIB_CFLAGS "-DCOM_NO_WINDOWS_H -DAPR_DECLARE_STATIC")
set(APRLIB_CFLAGS "${COMMON_C_FLAGS} ${COMMON_COMPILER_DEFINITIONS_STR} ${APRLIB_CFLAGS}")

message(STATUS "ZZZ APRLIB_CFLAGS=${APRLIB_CFLAGS}")

if (UNIX)
    set(APRLIB_CONFIG_OPTIONS --enable-static --disable-shared --disable-ipv6)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(APRLIB_CONFIG_OPTIONS ${APRLIB_CONFIG_OPTIONS} --enable-debug)
    endif()

    # PROBLEMS:
    # -std=c++11
    # -Werror: checking which type to use for apr_off_t... configure: error: could not determine the size of off_t

    ExternalProject_Add(Apr1StaticLib
        URL ${REPOSITORY_DIR}/external/common/share/apr/unix/apr-1.5.2.tar.gz
        UPDATE_COMMAND ""

        CONFIGURE_COMMAND
            ${EP_BASE}/Source/Apr1StaticLib/configure
                --prefix=${APRLIB_INSTALL_PREFIX}
                ${APRLIB_CONFIG_OPTIONS}
                CFLAGS=${APRLIB_CFLAGS}
                #LDFLAGS=${COMMON_LINK_FLAGS}

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

    set(APRLIB_SOURCE_DIR "${REPOSITORY_DIR}/external/common/share/apr/win/apr-1.5.2")

    ExternalProject_Add(Apr1StaticLib
        URL ${APRLIB_SOURCE_DIR}
        UPDATE_COMMAND ""

        CMAKE_GENERATOR ${CMAKE_GENERATOR}

        # TODO Figure out what to do with INSTALL_PDB. We disabled it because our manual INSTALL_COMMAND was not finding the pdb file and failing.
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=OFF
            -DAPR_HAVE_IPV6=OFF
            -DCMAKE_C_FLAGS=${APRLIB_CFLAGS}
            -DCMAKE_INSTALL_PREFIX=${APRLIB_INSTALL_PREFIX}
            #-DCMAKE_STATIC_LINKER_FLAGS=${COMMON_LINK_FLAGS}
            -DINSTALL_PDB=OFF

        #COMMAND ${CMAKE_COMMAND} -E echo "\"INSTALL_COMMAND: EP_BASE=${EP_BASE} CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR} CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}\""
        #COMMAND ${CMAKE_COMMAND} -E echo "\"SOURCE_DIR=<SOURCE_DIR>, BINARY_DIR=<BINARY_DIR>, INSTALL_DIR=<INSTALL_DIR>, and TMP_DIR=<TMP_DIR>\""

        #LOG_INSTALL 1
    )


    ExternalProject_Add_Step(Apr1StaticLib move_installed_headers_to_apr_1
        COMMENT "Windows: moving installed apr headers to include/apr-1, as expected by nupic.core"

        # Move the installed ${LIB_STATIC_APR1_INC_DIR}/*.h to
        # ${LIB_STATIC_APR1_INC_DIR}/apr-1
        COMMAND
            ${CMAKE_COMMAND} -DGLOBBING_EXPR=${LIB_STATIC_APR1_INC_DIR}/*.h
                -DDEST_DIR_PATH=${LIB_STATIC_APR1_INC_DIR}/apr-1
                -P ${CMAKE_SOURCE_DIR}/external/MoveFilesToNewDir.cmake
        # Copy ${EP_BASE}/Source/Apr1StaticLib/include/arch to
        # ${LIB_STATIC_APR1_INC_DIR}/apr-1 as expected by nupic.core
        COMMAND
            ${CMAKE_COMMAND} -E make_directory ${LIB_STATIC_APR1_INC_DIR}/apr-1/arch
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${EP_BASE}/Source/Apr1StaticLib/include/arch ${LIB_STATIC_APR1_INC_DIR}/apr-1/arch

        DEPENDEES install
        ALWAYS 0

        #LOG 1
    )
endif()