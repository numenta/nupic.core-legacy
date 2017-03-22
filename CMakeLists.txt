# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2017, Numenta, Inc.  Unless you have purchased from
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

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

set(CMAKE_VERBOSE_MAKEFILE OFF)

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} ABSOLUTE)

if( POLICY CMP0046 )
  cmake_policy(VERSION 2.8)
  # The new policy for add_dependencies is to now error if a dependency target
  # is not found. The old policy didn't care and continued through configuration
  cmake_policy(SET CMP0046 OLD)
endif()

if( POLICY CMP0054 )
  # The OLD behavior for this policy is to dereference variables and interpret
  # keywords even if they are quoted or bracketed. The NEW behavior is to not
  # dereference variables or interpret keywords that have been quoted or bracketed.
  cmake_policy(SET CMP0054 OLD)
endif()


#
# Global NuPIC CMake options
#
option(NUPIC_BUILD_PYEXT_MODULES
       "Turn on building of python extension modules for nupic.bindings; turn off to build only static nupic_core lib with full symbol visibility."
       ON)

message(STATUS "NUPIC_BUILD_PYEXT_MODULES = ${NUPIC_BUILD_PYEXT_MODULES}")
message(STATUS "PY_EXTENSIONS_DIR         = ${PY_EXTENSIONS_DIR}")

message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_C_COMPILER      = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER    = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE      = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX  = ${CMAKE_INSTALL_PREFIX}")


# Identify platform name.
string(TOLOWER ${CMAKE_SYSTEM_NAME} PLATFORM)

# Define a platform suffix, eg ${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}
if (MSYS OR MINGW)
  set(PLATFORM_SUFFIX -gcc)
endif()


# Make sure we can link against any specified paths.
include_directories(SYSTEM ${CMAKE_INCLUDE_PATH})

set(EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

# Determine common toolchain settings, compiler and link flags
include(CommonCompilerConfig)

# Set up builds of external dependencies and get their exports.
# (see individual external/*.cmake modules for exported settings and functions)
add_subdirectory(external)
include_directories(SYSTEM ${EXTERNAL_INCLUDE_DIRS})

# Now build nupic_core project.
add_subdirectory(src)
