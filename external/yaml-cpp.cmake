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

# Expands the distribution file and builds the yaml-cpp static library
#
# OUTPUT VARIABLES:
#   exports 'yaml-cpp' as a target

set(YAML_CPP_STATIC_LIB_TARGET yaml-cpp-lib)
set(url "${REPOSITORY_DIR}/external/common/share/yaml-cpp/yaml-cpp-release-0.6.2.tar.gz")

include(FetchContent)
FetchContent_Declare(yaml-cpp-lib
    URL ${url}
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
)
FetchContent_GetProperties(yaml-cpp-lib)
if (NOT YamlCppStaticLib_POPULATED)
	FetchContent_Populate(yaml-cpp-lib)
	option(YAML_CPP_BUILD_TESTS "Enable testing (builds gtest)" OFF)
	add_subdirectory(${yaml-cpp-lib_SOURCE_DIR} ${yaml-cpp-lib_BINARY_DIR})
endif()

