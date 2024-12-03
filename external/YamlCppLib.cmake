# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Creates ExternalProject for building the yaml-cpp static library
#
# OUTPUT VARIABLES:
#
#   YAML_CPP_STATIC_LIB_TARGET: name of static library target that contains all
#                               of yaml-cpp library objects.
#   YAML_CPP_STATIC_LIB_INC_DIR: directory of installed yaml-cpp lib headers

include(../src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(YAML_CPP_STATIC_LIB_TARGET yaml-cpp-bundle)


set(yamlcpplib_url "${REPOSITORY_DIR}/external/common/share/yaml-cpp/yaml-cpp-release-0.3.0.tar.gz")
set(yamlcpplib_install_prefix "${EP_BASE}/Install/YamlCppStaticLib")
set(yamlcpplib_install_lib_dir "${yamlcpplib_install_prefix}/lib")

# Export directory of installed yaml-cpp lib headers to parent
set(YAML_CPP_STATIC_LIB_INC_DIR "${yamlcpplib_install_prefix}/include")

# Path to static yaml-cpp installed by external project
set(yamlcpplib_built_archive_file
    "${yamlcpplib_install_lib_dir}/${STATIC_PRE}yaml-cpp${STATIC_SUF}")

set(c_flags "${EXTERNAL_C_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")
set(cxx_flags "${EXTERNAL_CXX_FLAGS_OPTIMIZED} ${COMMON_COMPILER_DEFINITIONS_STR}")

ExternalProject_Add(YamlCppStaticLib
    DEPENDS YamlStaticLib

    URL ${yamlcpplib_url}

    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_CPP_BUILD_CONTRIB=OFF
        -DCMAKE_C_FLAGS=${c_flags}
        -DCMAKE_CXX_FLAGS=${cxx_flags}
        -DCMAKE_INSTALL_PREFIX=${yamlcpplib_install_prefix}
        ${EXTERNAL_STATICLIB_CMAKE_DEFINITIONS_OPTIMIZED}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${YAML_CPP_STATIC_LIB_TARGET}
                       "${yamlcpplib_built_archive_file}")
add_dependencies(${YAML_CPP_STATIC_LIB_TARGET} YamlCppStaticLib)
