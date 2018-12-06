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
# Note: yaml-cpp includes an older version of gtest so 
#       turn off YAML_CPP_BUILD_TESTS to prevent it from building gtest.

message(STATUS "Obtaining yaml-cpp")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ yaml-cpp
	PREFIX ${EP_BASE}/yaml-cpp
	URL https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz
	UPDATE_DISCONNECTED 1
	QUIET
	)
set(YAML_CPP_INSTALL OFF CACHE BOOL "prevent install, not needed." FORCE) 
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "prevent tools from being build" FORCE) 
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "prevent gtest from being build" FORCE) 
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "prevent contrib modules" FORCE) 
add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})

find_package(yaml-cpp REQUIRED HINT ${yaml-cpp_SOURCE_DIR})
set(yaml-cpp_INCLUDE_DIRS ${yaml-cpp_SOURCE_DIR}/include CACHE BOOL "include directory" FORCE) 
set(yaml-cpp_LIBRARIES   ${YAML_CPP_LIBRARIES}  CACHE BOOL "libraries to link with" FORCE) 

