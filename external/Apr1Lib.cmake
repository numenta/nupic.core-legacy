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

set(aprlib_install_prefix "${EP_BASE}/Install/Apr1StaticLib")
set(aprlib_install_lib_dir "${aprlib_install_prefix}/lib")

# Export directory of installed apr-1 lib headers
set(LIB_STATIC_APR1_INC_DIR "${aprlib_install_prefix}/include")

# Export path to installed static apr-1 lib
set(LIB_STATIC_APR1_LOC "${aprlib_install_lib_dir}/${STATIC_PRE}apr-1${STATIC_SUF}")

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

    # gcc v4.9 requires its own binutils-wrappers for LTO (flag -flto)
    # fixes #981
    if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9" OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "4.9"))
        ExternalProject_Add(Apr1StaticLib
        	URL ${aprlib_url}

        	UPDATE_COMMAND ""
        	PATCH_COMMAND ""

        	CONFIGURE_COMMAND
            	<SOURCE_DIR>/configure AR=gcc-ar NM=gcc-nm RANLIB=gcc-ranlib
                	--prefix=${aprlib_install_prefix}
                	${aprlib_config_options}
                	CFLAGS=${aprlib_cflags}

        	BUILD_COMMAND
            	make -f Makefile all

        	INSTALL_COMMAND
            	make -f Makefile install
    	)
    else()
		ExternalProject_Add(Apr1StaticLib
        	URL ${aprlib_url}

        	UPDATE_COMMAND ""
        	PATCH_COMMAND ""

        	CONFIGURE_COMMAND
            	<SOURCE_DIR>/configure
                	--prefix=${aprlib_install_prefix}
                	${aprlib_config_options}
                	CFLAGS=${aprlib_cflags}

        	BUILD_COMMAND
            	make -f Makefile all

        	INSTALL_COMMAND
            	make -f Makefile install
    	)
	endif()
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

        # Move the installed ${LIB_STATIC_APR1_INC_DIR}/*.h to
        # ${LIB_STATIC_APR1_INC_DIR}/apr-1
        COMMAND
            ${CMAKE_COMMAND} -DGLOBBING_EXPR=${LIB_STATIC_APR1_INC_DIR}/*.h
                -DDEST_DIR_PATH=${LIB_STATIC_APR1_INC_DIR}/apr-1
                -P ${CMAKE_SOURCE_DIR}/external/MoveFilesToNewDir.cmake
        # Copy <SOURCE_DIR>/include/arch to ${LIB_STATIC_APR1_INC_DIR}/apr-1 as
        # expected by nupic.core
        COMMAND
            ${CMAKE_COMMAND} -E make_directory ${LIB_STATIC_APR1_INC_DIR}/apr-1/arch
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/arch ${LIB_STATIC_APR1_INC_DIR}/apr-1/arch
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
