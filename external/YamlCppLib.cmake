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
# Exports:
#   LIB_STATIC_YAML_CPP_INC_DIR: directory of installed yaml-cpp lib headers
#   LIB_STATIC_YAML_CPP_LOC: path to installed static yaml-cpp lib

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

set(YAMLCPPLIB_SOURCE_DIR ${REPOSITORY_DIR}/external/common/share/yaml-cpp/yaml-cpp-release-0.3.0)
set(YAMLCPPLIB_INSTALL_PREFIX ${EP_BASE}/Install/YamlCppStaticLib)
set(LIB_STATIC_YAML_CPP_INC_DIR ${YAMLCPPLIB_INSTALL_PREFIX}/include)
set(YAMLCPPLIB_INSTALL_LIB_DIR ${YAMLCPPLIB_INSTALL_PREFIX}/lib)

# Export directory of installed yaml-cpp lib headers to parent
set(LIB_STATIC_YAML_CPP_INC_DIR ${LIB_STATIC_YAML_CPP_INC_DIR} PARENT_SCOPE)

# Export path to installed static yaml-cpp to parent
set(LIB_STATIC_YAML_CPP_LOC ${YAMLCPPLIB_INSTALL_LIB_DIR}/${STATIC_PRE}yaml-cpp${STATIC_SUF} PARENT_SCOPE)

ExternalProject_Add(YamlCppStaticLib
    DEPENDS YamlStaticLib
    SOURCE_DIR ${YAMLCPPLIB_SOURCE_DIR}
    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${YAMLCPPLIB_INSTALL_PREFIX}
)
