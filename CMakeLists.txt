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
project(nupic_core_main CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

# Identify platform name.
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)
string(TOUPPER ${PLATFORM} PLATFORM_UPPERCASE)

# Identify platform "bitness".
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()

# Make sure we can link against any specified paths.
include_directories(SYSTEM ${CMAKE_INCLUDE_PATH})

set(EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

# Set up external dependencies.
#
# This will try find_library and find_program and if that fails use
# ExternalProject to fetch and build the libraries.
#
# Sets the following variables:
#   CAPNP_LIBRARIES
#   EXTERNAL_INCLUDE_DIRS
#
# Defines the follow functions:
#   create_capnpc_target - Creates target that generates C++ files from
#       .capnp schema files.
add_subdirectory(external)
include_directories(SYSTEM ${EXTERNAL_INCLUDE_DIRS})

# Now build nupic_core project.
add_subdirectory(src)
