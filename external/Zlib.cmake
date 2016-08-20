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
# OUTPUT VARIABLES:
#
#   Z_STATIC_LIB_TARGET: name of static library target that contains all
#                        of the z library objects.
#   Z_STATIC_LIB_INC_DIR: directory of installed z lib headers

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(Z_STATIC_LIB_TARGET z-bundle)


set(zlib_url "${REPOSITORY_DIR}/external/common/share/zlib/zlib-1.2.8.tar.gz")
set(zlib_source_dir "${REPOSITORY_DIR}/external/common/share/zlib/zlib-1.2.8")
set(zlib_install_prefix "${EP_BASE}/Install/ZStaticLib")
set(zlib_install_lib_dir "${zlib_install_prefix}/lib")

if(UNIX)
    # On unix-like platforms the library is almost always called libz
   set(zlib_output_root z)
else()
   set(zlib_output_root zlibstatic)
endif()


# Export directory of installed z lib headers to parent
set(Z_STATIC_LIB_INC_DIR "${zlib_install_prefix}/include")

# Path to static z lib installed by external project
set(zlib_built_archive_file
    "${zlib_install_lib_dir}/${STATIC_PRE}${zlib_output_root}${STATIC_SUF}")

set(c_flags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")

set(zlib_patch_file "${CMAKE_SOURCE_DIR}/external/common/share/zlib/zlib.patch")

ExternalProject_Add(ZStaticLib
    URL ${zlib_url}

    PATCH_COMMAND patch -f -p1 --directory=<SOURCE_DIR> --input=${zlib_patch_file}

    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_C_FLAGS=${c_flags}
        -DCMAKE_INSTALL_PREFIX=${zlib_install_prefix}
        -DINSTALL_BIN_DIR=${zlib_install_prefix}/bin
        -DINSTALL_INC_DIR=${Z_STATIC_LIB_INC_DIR}
        -DINSTALL_LIB_DIR=${zlib_install_lib_dir}
        -DINSTALL_MAN_DIR=${zlib_install_prefix}/man
        -DINSTALL_PKGCONFIG_DIR=${zlib_install_prefix}/pkgconfig
        ${EXTERNAL_STATICLIB_CMAKE_DEFINITIONS_OPTIMIZED}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${Z_STATIC_LIB_TARGET}
                       "${zlib_built_archive_file}")
add_dependencies(${Z_STATIC_LIB_TARGET} ZStaticLib)
