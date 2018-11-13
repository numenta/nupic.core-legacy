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

cmake_minimum_required(VERSION 3.3)
project(nupic_core CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
include(GNUInstallDirs)

set_directory_properties(PROPERTIES EP_BASE "${EP_BASE}")

# Shorter aliases for static library prefix and suffix.
set(STATIC_PRE ${CMAKE_STATIC_LIBRARY_PREFIX})
set(STATIC_SUF ${CMAKE_STATIC_LIBRARY_SUFFIX})
# This is the library prefix path.
set(LIB_PRE ${EP_BASE}/Install/${CMAKE_INSTALL_LIBDIR})
# This is the include path.
set(INCLUDE_PRE ${EP_BASE}/Install/include)
# The full prefix path for binaries.
set(BIN_PRE ${EP_BASE}/Install/bin)

include(ExternalProject)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

include(Swig)

include(Apr1Lib)
set(APR1_STATIC_LIB_TARGET ${APR1_STATIC_LIB_TARGET} PARENT_SCOPE)
set(APR1_STATIC_LIB_INC_DIR ${APR1_STATIC_LIB_INC_DIR} PARENT_SCOPE)
list(APPEND EXTERNAL_INCLUDE_DIRS
     ${APR1_STATIC_LIB_INC_DIR}
     "${APR1_STATIC_LIB_INC_DIR}/apr-1"
)

# AprUtil1Lib MUST be imported after Apr1Lib due to depencies on exported
# properties
include(AprUtil1Lib)
set(APRUTIL1_STATIC_LIB_TARGET ${APRUTIL1_STATIC_LIB_TARGET} PARENT_SCOPE)
set(APRUTIL1_STATIC_LIB_INC_DIR ${APRUTIL1_STATIC_LIB_INC_DIR} PARENT_SCOPE)
list(APPEND EXTERNAL_INCLUDE_DIRS ${APRUTIL1_STATIC_LIB_INC_DIR})

include(YamlLib)
set(YAML_STATIC_LIB_TARGET ${YAML_STATIC_LIB_TARGET} PARENT_SCOPE)

include(YamlCppLib)
set(YAML_CPP_STATIC_LIB_TARGET ${YAML_CPP_STATIC_LIB_TARGET} PARENT_SCOPE)
set(YAML_CPP_STATIC_LIB_INC_DIR ${YAML_CPP_STATIC_LIB_INC_DIR} PARENT_SCOPE)
list(APPEND EXTERNAL_INCLUDE_DIRS ${YAML_CPP_STATIC_LIB_INC_DIR})

##################################################
#  BOOST
#  Boost must be pre-installed if it is needed.
if(NEEDS_BOOST)
  set(BOOST_ROOT ${external_BOOST_ROOT})
  message(STATUS "Hints for Boost find_package()")
  message(STATUS "  BOOST_ROOT  = ${BOOST_ROOT}")
  message(STATUS "Expect some warnings from Boost about missing dependancies.  It is ok.")

  #set(Boost_USE_STATIC_RUNTIME OFF)
  #set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
  #set(Boost_USE_MULTITHREADED      ON)
  #set(Boost_DEBUG ON)
  #set(Boost_DETAILED_FAILURE_MSG ON)
  # info about "--layout versioned" https://stackoverflow.com/questions/32991736/boost-lib-naming-are-missing/52208574#52208574
  # filenames created by boost: https://www.boost.org/doc/libs/1_68_0/more/getting_started/unix-variants.html#library-naming
  # We already found or created Boost in Boost.cmake, so just take whatever we find in ${BOOST_ROOT}.

  find_package(Boost COMPONENTS system filesystem REQUIRED)
  if(Boost_FOUND)
        ADD_DEFINITIONS( "-DHAS_BOOST" )
        message(STATUS "  Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
        message(STATUS "  Boost_LIBRARIES = ${Boost_LIBRARIES}")
  else()
        message(FATAL_ERROR "Fatal Error: Boost not found.")
  endif()
endif()

##################################################

# Add vendored prebuilt library include paths.
list(APPEND EXTERNAL_INCLUDE_DIRS
     "${PROJECT_SOURCE_DIR}/common/include")

# Convenience variable that wraps all external include directories.
list(APPEND EXTERNAL_INCLUDE_DIRS ${CAPNP_INCLUDE_DIRS})

set(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS} PARENT_SCOPE)
