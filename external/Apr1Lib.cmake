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
# OUTPUT VARIABLES:
#
#   APR1_STATIC_LIB_TARGET: name of static library target that contains all
#                           of the apr-1 library objects.
#   APR1_STATIC_LIB_INC_DIR: directory of installed apr-1 lib headers

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(APR1_STATIC_LIB_TARGET apr-1-bundle)


set(aprlib_install_prefix "${EP_BASE}/Install/Apr1StaticLib")
set(aprlib_install_lib_dir "${aprlib_install_prefix}/lib")

# Export directory of installed apr-1 lib headers
set(APR1_STATIC_LIB_INC_DIR "${aprlib_install_prefix}/include")

# Path to static apr-1 lib installed by external project
set(aprlib_built_archive_file
    "${aprlib_install_lib_dir}/${STATIC_PRE}apr-1${STATIC_SUF}")

# NOTE -DCOM_NO_WINDOWS_H fixes a bunch of OLE-related build errors on Win32
# (reference: https://bz.apache.org/bugzilla/show_bug.cgi?id=56342)
set(aprlib_cflags "-DCOM_NO_WINDOWS_H")
set(aprlib_cflags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR} ${aprlib_cflags}")

# Location of apr sources
set(aprlib_url "${REPOSITORY_DIR}/external/common/share/apr/unix/apr-1.5.2.tar.gz")

# Get it built!
if (UNIX)
    set(aprlib_config_options --enable-static --disable-shared --disable-ipv6)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(aprlib_config_options ${aprlib_config_options} --enable-debug)
    endif()

    ExternalProject_Add(Apr1StaticLib
        URL ${aprlib_url}

        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        CONFIGURE_COMMAND
            <SOURCE_DIR>/configure
                ${EXTERNAL_STATICLIB_CONFIGURE_DEFINITIONS_OPTIMIZED}
                --prefix=${aprlib_install_prefix}
                ${aprlib_config_options}
                CFLAGS=${aprlib_cflags}

        BUILD_COMMAND
            make -f Makefile all

        INSTALL_COMMAND
            make -f Makefile install
    )
else()
    # NOT UNIX - i.e., Windows

    ExternalProject_Add(Apr1StaticLib
        URL ${aprlib_url}

        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        CMAKE_GENERATOR ${CMAKE_GENERATOR}

        # TODO Figure out what to do with INSTALL_PDB. We disabled it because
        # our manual INSTALL_COMMAND was not finding the pdb file and failing.
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBUILD_SHARED_LIBS=OFF
            -DAPR_HAVE_IPV6=OFF
            -DCMAKE_C_FLAGS=${aprlib_cflags}
            -DCMAKE_INSTALL_PREFIX=${aprlib_install_prefix}
            -DINSTALL_PDB=OFF


        #LOG_INSTALL 1
    )


    #
    # Add step to organize generic and architecture-specific apr headers under
    # include/apr-1 subdirectory
    # NOTE the unix configure-based installation does that and nupic.core
    #      depends on this include directory organization.
    #

    ExternalProject_Add_Step(Apr1StaticLib move_installed_headers_to_apr_1
        COMMENT "Windows: moving installed apr headers to include/apr-1, as expected by nupic.core"

        DEPENDEES install
        ALWAYS 0
        #LOG 1

        # Move the installed ${APR1_STATIC_LIB_INC_DIR}/*.h to
        # ${APR1_STATIC_LIB_INC_DIR}/apr-1
        COMMAND
            ${CMAKE_COMMAND} -DGLOBBING_EXPR=${APR1_STATIC_LIB_INC_DIR}/*.h
                -DDEST_DIR_PATH=${APR1_STATIC_LIB_INC_DIR}/apr-1
                -P ${CMAKE_SOURCE_DIR}/external/MoveFilesToNewDir.cmake
        # Copy <SOURCE_DIR>/include/arch to ${APR1_STATIC_LIB_INC_DIR}/apr-1 as
        # expected by nupic.core
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
                             ${APR1_STATIC_LIB_INC_DIR}/apr-1/arch
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory
                             <SOURCE_DIR>/include/arch
                             ${APR1_STATIC_LIB_INC_DIR}/apr-1/arch
    )
endif()


#
# Add step to patch apr-1 sources
#

# Patch file path
set(aprlib_patch_file "${CMAKE_SOURCE_DIR}/external/common/share/apr/apr.patch")

ExternalProject_Add_Step(Apr1StaticLib patch_sources
    COMMENT "Patching apr-1 sources in <SOURCE_DIR> via ${aprlib_patch_file}"

    DEPENDEES update
    DEPENDERS configure
    ALWAYS 0
    #LOG 1

    COMMAND
        patch -f -p1 --directory=<SOURCE_DIR> --input=${aprlib_patch_file}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${APR1_STATIC_LIB_TARGET}
                       ${aprlib_built_archive_file})
add_dependencies(${APR1_STATIC_LIB_TARGET} Apr1StaticLib)
