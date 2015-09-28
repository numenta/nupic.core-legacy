# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
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

cmake_minimum_required(VERSION 2.8)
project(nupic_core_main)

set(CMAKE_VERBOSE_MAKEFILE ON)

set_directory_properties(
  PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}
)

include(ExternalProject)

# Set up dependencies and build Cap'n Proto
ExternalProject_Add(
  externals
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/external
)


MESSAGE(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# Build nupic_core project
ExternalProject_Add(
  nupic_core
  DEPENDS externals
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
)
