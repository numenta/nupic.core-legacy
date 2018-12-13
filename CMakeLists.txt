# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2018, Numenta, Inc.  Unless you have purchased from
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

cmake_minimum_required(VERSION 3.7)
project(nupic_core CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} ABSOLUTE)


#
# Global NuPIC CMake options
#
option(USE_CPP11 "Force compiler to use C++11 standard." ON)
option(NEEDS_BOOST "Force compiler to install and use Boost." ON)
option(PYTHON2_BUILD "Toggle whether to build for Python2.7, default ON." OFF)
option(PYTHON3_BUILD "Toggle whether to build for Python3.x, default ON." OFF)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}" CACHE PATH "Default Install location" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


# Make sure we can link against any specified paths.
include_directories(SYSTEM ${CMAKE_INCLUDE_PATH})


# Determine common toolchain settings, compiler and link flags
include(CommonCompilerConfig)

message(STATUS "CMAKE_CXX_COMPILER_ID= ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_C_COMPILER     = ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER   = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE     = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "INTERNAL_CPP_STANDARD= ${INTERNAL_CPP_STANDARD}")
message(STATUS "PLATFORM             = ${PLATFORM}")
message(STATUS "BITNESS              = ${BITNESS}")
message(STATUS "NEEDS_BOOST          = ${NEEDS_BOOST}")
message(STATUS "INTERNAL_CXX_FLAGS   = ${INTERNAL_CXX_FLAGS}")
message(STATUS "INTERNAL_LINKER_FLAGS= ${INTERNAL_LINKER_FLAGS}")
message(STATUS "COMMON_COMPILER_DEFINITIONS_STR=${COMMON_COMPILER_DEFINITIONS_STR}")
message(STATUS "COMMON_OS_LIBS       = ${COMMON_OS_LIBS}")
message(STATUS "PYTHON2_BUILD        = ${PYTHON2_BUILD}")
message(STATUS "PYTHON3_BUILD        = ${PYTHON3_BUILD}")



# Set up builds of external dependencies and get their exports.
# (see individual external/*.cmake modules for exported settings and functions)
add_subdirectory(external)

# Now build nupic_core project.
add_subdirectory(src)

if (PYTHON2_BUILD OR PYTHON3_BUILD)
  add_subdirectory(bindings/py/cpp_src)
endif()
