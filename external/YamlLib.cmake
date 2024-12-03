# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Creates ExternalProject for building yaml lib static library
#
# OUTPUT VARIABLES:
#
#   YAML_STATIC_LIB_TARGET: name of static library target that contains all
#                           of yaml library objects.

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(YAML_STATIC_LIB_TARGET yaml-bundle)


set(yamllib_url "${REPOSITORY_DIR}/external/common/share/yaml/yaml-0.1.5.tar.gz")

# NOTE Yaml lib doesn't have an install target and leaves artifacts in build dir
set(yamllib_build_dir "${EP_BASE}/Build/YamlStaticLib")

# Path to static yaml installed by external project
set(yamllib_built_archive_file
    "${yamllib_build_dir}/${STATIC_PRE}yaml${STATIC_SUF}")

set(c_flags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")

ExternalProject_Add(YamlStaticLib
    URL ${yamllib_url}

    UPDATE_COMMAND ""

    # NOTE Yaml provides no rule for install
    INSTALL_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_C_FLAGS=${c_flags}
        -DCMAKE_INSTALL_PREFIX=${yamllib_build_dir}
        ${EXTERNAL_STATICLIB_CMAKE_DEFINITIONS_OPTIMIZED}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${YAML_STATIC_LIB_TARGET}
                       "${yamllib_built_archive_file}")
add_dependencies(${YAML_STATIC_LIB_TARGET} YamlStaticLib)
