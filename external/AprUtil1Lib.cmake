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
# OUTPUT VARIABLES:
#
#   APRUTIL1_STATIC_LIB_TARGET: name of static library target that contains all
#                               of the aprutil-1 library objects.
#   APRUTIL1_STATIC_LIB_INC_DIR: directory of installed aprutil-1 lib headers

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(APRUTIL1_STATIC_LIB_TARGET aprutil-1-bundle)


set(aprutillib_install_prefix "${EP_BASE}/Install/AprUtil1StaticLib")
set(aprutillib_install_lib_dir "${aprutillib_install_prefix}/lib")

# Export directory of installed aprutil-1 lib headers to parent
set(APRUTIL1_STATIC_LIB_INC_DIR "${aprutillib_install_prefix}/include")

# Path to static aprutil-1 lib installed by external project
set(aprutillib_built_archive_file
    "${aprutillib_install_lib_dir}/${STATIC_PRE}aprutil-1${STATIC_SUF}")

# NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors in apr-1
# on Win32 (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
set(aprutillib_cflags "-DCOM_NO_WINDOWS_H -DAPR_DECLARE_STATIC")
set(aprutillib_cflags "${aprutillib_cflags} -I${APR1_STATIC_LIB_INC_DIR}/apr-1")
set(aprutillib_cflags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR} ${aprutillib_cflags}")

set(aprutillib_url "${REPOSITORY_DIR}/external/common/share/apr-util/unix/apr-util-1.5.4.tar.gz")

if (UNIX)
    set(aprutillib_config_options
        --disable-util-dso --with-apr=${APR1_STATIC_LIB_INC_DIR}/..)

    ExternalProject_Add(AprUtil1StaticLib
        DEPENDS Apr1StaticLib

        URL ${aprutillib_url}

        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        CONFIGURE_COMMAND
            <SOURCE_DIR>/configure
                ${EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED}
                --prefix=${aprutillib_install_prefix}
                ${aprutillib_config_options}
                CFLAGS=${aprutillib_cflags}

        BUILD_COMMAND
            make -f Makefile all

        INSTALL_COMMAND
            make -f Makefile install
    )

else()
    # NOT UNIX - i.e., Windows

    ExternalProject_Add(AprUtil1StaticLib
        DEPENDS Apr1StaticLib

        URL ${aprutillib_url}

        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        CMAKE_GENERATOR ${CMAKE_GENERATOR}

        # TODO Figure out what to do with INSTALL_PDB. We disabled it because
        # our manual INSTALL_COMMAND was not finding the pdb file and failing.
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_C_FLAGS=${aprutillib_cflags}
            -DCMAKE_INSTALL_PREFIX=${aprutillib_install_prefix}
            -DAPR_HAS_LDAP=OFF
            -DAPU_HAVE_ODBC=OFF
            -DAPR_INCLUDE_DIR=${APR1_STATIC_LIB_INC_DIR}/apr-1
            -DAPR_LIBRARIES=
            -DINSTALL_PDB=OFF

        #LOG_INSTALL 1
    )


    #
    # Add step to organize apr-util headers under include/apr-1 subdirectory
    # NOTE the unix configure-based installation does that and nupic.core
    #      depends on this include directory organization.
    #

    ExternalProject_Add_Step(AprUtil1StaticLib move_installed_headers_to_apr_1
        COMMENT "Windows: moving installed apr-util headers to include/apr-1, as expected by nupic.core"

        DEPENDEES install
        ALWAYS 0
        #LOG 1

        # Move the installed ${APRUTIL1_STATIC_LIB_INC_DIR}/*.h to
        # ${APRUTIL1_STATIC_LIB_INC_DIR}/apr-1
        COMMAND
            ${CMAKE_COMMAND} -DGLOBBING_EXPR=${APRUTIL1_STATIC_LIB_INC_DIR}/*.h
                -DDEST_DIR_PATH=${APRUTIL1_STATIC_LIB_INC_DIR}/apr-1
                -P ${CMAKE_SOURCE_DIR}/external/MoveFilesToNewDir.cmake

    )
endif()


#
# Add step to patch aprutil-1 sources
#

# Patch file path
set(aprutillib_patch_file "${CMAKE_SOURCE_DIR}/external/common/share/apr-util/apru.patch")

ExternalProject_Add_Step(AprUtil1StaticLib patch_sources
    COMMENT "Patching aprutil-1 sources in <SOURCE_DIR> via ${aprutillib_patch_file}"

    DEPENDEES update
    DEPENDERS configure
    ALWAYS 0
    #LOG 1

    COMMAND
        patch -f -p1 --directory=<SOURCE_DIR> --input=${aprutillib_patch_file}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${APRUTIL1_STATIC_LIB_TARGET}
                       ${aprutillib_built_archive_file})
add_dependencies(${APRUTIL1_STATIC_LIB_TARGET} AprUtil1StaticLib)
