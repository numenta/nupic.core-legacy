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

# Creates ExternalProject for building yaml lib static library
#
# Exports:
#   LIB_STATIC_YAML_LOC: path to installed static yaml lib


get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

set(YAMLLIB_SOURCE_DIR ${REPOSITORY_DIR}/external/common/share/yaml/yaml-0.1.5)
# NOTE Yaml lib doesn't have an install target and leaves artifacts in build dir
set(YAMLLIB_BUILD_DIR ${EP_BASE}/Build/YamlStaticLib)
set(YAMLLIB_INSTALL_LIB_DIR ${YAMLLIB_INSTALL_PREFIX}/lib)

# Export path to installed static yaml lib to parent
set(LIB_STATIC_YAML_LOC ${YAMLLIB_BUILD_DIR}/${STATIC_PRE}yaml${STATIC_SUF} PARENT_SCOPE)

ExternalProject_Add(YamlStaticLib
    SOURCE_DIR ${YAMLLIB_SOURCE_DIR}
    UPDATE_COMMAND ""

    # NOTE Yaml provides no rule for install
    INSTALL_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${YAMLLIB_BUILD_DIR}
)
