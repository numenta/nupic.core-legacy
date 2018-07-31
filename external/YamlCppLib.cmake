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


set(yamlcpplib_url "${REPOSITORY_DIR}/external/common/share/yaml-cpp/yaml-cpp-release-0.6.2.tar.gz")
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
	-DYAML_CPP_BUILD_TESTS=OFF # causes build errors with gtest (as of YamlCpp 0.6.2)
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
