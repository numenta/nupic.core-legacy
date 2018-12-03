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
#
# This loads Boost if it is needed.
#


if(${NEEDS_BOOST})
  # Known versions of Boost......
  # 	Boost 1.63 requires CMake 3.7 or newer.  -- our minimum CMake version
  # 	Boost 1.64 requires CMake 3.8 or newer.
  # 	Boost 1.65 and 1.65.1 require CMake 3.9.3 or newer.
  # 	Boost 1.66 requires CMake 3.11 or newer.
  # 	Boost 1.67 requires CMake 3.12 or newer.
  # 	Boost 1.68, 1.69 require CMake 3.13 or newer.
  #  If your version of CMake does not know about your version of Boost,
  #     add your Boost version to this list to avoid warnings:
  set(Boost_ADDITIONAL_VERSIONS}
	"1.69.0" "1.69"
    "1.68.0" "1.68" "1.67.0" "1.67" "1.66.0" "1.66" "1.65.1" "1.65.0" "1.65"
    "1.64.0" "1.64" "1.63.0" "1.63" )    

  message(STATUS "Hints for find_package()")
  message(STATUS "  BOOST_ROOT  = ${BOOST_ROOT}")
  message(STATUS "--Ignore Warnings about 'incorrect or missing Boost dependencies'")
  if(MSVC OR MINGW)
    # For windows, library name has lots of fields that must be matched.
    # info about "--layout versioned" https://stackoverflow.com/questions/32991736/boost-lib-naming-are-missing/52208574#52208574
    # filenames created by boost: https://www.boost.org/doc/libs/1_68_0/more/getting_started/unix-variants.html#library-naming
    set(Boost_USE_STATIC_RUNTIME     OFF) 
    set(Boost_USE_MULTITHREADED      ON)
  endif()
  set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
  #set(Boost_DEBUG ON)
  set(Boost_DETAILED_FAILURE_MSG ON)
  
  find_package(Boost 1.56.0 REQUIRED COMPONENTS system filesystem)
  if(${Boost_FOUND})
	include_directories(${Boost_INCLUDE_DIRS})
	set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} PARENT_SCOPE)
	set(Boost_LIBRARIES ${Boost_LIBRARIES} PARENT_SCOPE)
	list(APPEND EXTERNAL_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
	message(STATUS "  Boost_INCLUDE_DIRS        = ${Boost_INCLUDE_DIRS}")
	message(STATUS "  Boost_LIBRARIES           = ${Boost_LIBRARIES}")
  else()
  	message(FATAL_ERROR "Boost not found.")
  endif()
endif()