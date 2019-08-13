# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2015-2018, Numenta, Inc.
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
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.7)
project(htm_core CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} ABSOLUTE)
set($ENV{HTM_CPP} ${REPOSITORY_DIR})

#
# Global NuPIC CMake options
#
option(FORCE_CPP11 "Force compiler to use C++11 standard." OFF)
option(FORCE_BOOST "Force compiler to install and use Boost." OFF)
set(BINDING_BUILD "none" CACHE STRING "Specify the Binding to build 'Python2','Python3' or 'none', default 'none'." )
# Note: by setting the CXX environment variable, a non-default c++ compiler can be specified.

if(MSVC AND BINDING_BUILD STREQUAL "Python2")
  message(FATAL_ERROR "The Extension library cannot be built for Python 2 using MSVC.  Use Python 3.x")
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}" CACHE PATH "Default Install location" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# obtain the version from git and put it in the VERSION file
include(versioning.cmake)
get_versions(VERSION MAJOR MINOR PATCH)

message(STATUS "REPOSITORY_DIR       = ${REPOSITORY_DIR}")
message(STATUS "CMAKE_BINARY_DIR     = ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE     = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "FORCE_CPP11          = ${FORCE_CPP11}")
message(STATUS "FORCE_BOOST          = ${FORCE_BOOST}")
message(STATUS "BINDING_BUILD        = ${BINDING_BUILD}")
message(STATUS "VERSION              = ${VERSION}")
message(STATUS "MAJOR                = ${MAJOR}")
message(STATUS "MINOR                = ${MINOR}")
message(STATUS "PATCH                = ${PATCH}")
message("")

# Determine common toolchain settings, compiler and link flags
include(CommonCompilerConfig)

message(STATUS "CMAKE_CXX_COMPILER_ID= ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER   = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_STANDARD   = ${CMAKE_CXX_STANDARD}")
message(STATUS "PLATFORM             = ${PLATFORM}")
message(STATUS "BITNESS              = ${BITNESS}")
message(STATUS "NEEDS_BOOST          = ${NEEDS_BOOST}")
message(STATUS "INTERNAL_CXX_FLAGS   = ${INTERNAL_CXX_FLAGS}")
message(STATUS "INTERNAL_LINKER_FLAGS= ${INTERNAL_LINKER_FLAGS}")
message(STATUS "INTERNAL_LINKER_FLAGS_STR= ${INTERNAL_LINKER_FLAGS_STR}")
message(STATUS "COMMON_COMPILER_DEFINITIONS_STR=${COMMON_COMPILER_DEFINITIONS_STR}")
message(STATUS "COMMON_OS_LIBS       = ${COMMON_OS_LIBS}")

# Set up builds of external dependencies and get their exports.
# (see individual external/*.cmake modules for exported settings and functions)
include(external/bootstrap.cmake)
 
# Now build htm_core project.
set(core_library   htm_core)
if(MSVC)
    set(src_lib_shared)
else()
    set(src_lib_shared htm_core_shared)
endif()
add_subdirectory(src)

string(REGEX MATCH "Python" match ${BINDING_BUILD})
if(match)
  add_subdirectory(bindings/py/cpp_src)
endif()

