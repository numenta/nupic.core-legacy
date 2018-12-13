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

# Creates ExternalProject for building the boost system and filesystem static libraries
#  Documentation: https://boostorg.github.io/build/manual/develop/index.html
#                 https://boostorg.github.io/build/tutorial.html
#
#  info about "--layout versioned" https://stackoverflow.com/questions/32991736/boost-lib-naming-are-missing/52208574#52208574
#               filenames created by boost: https://www.boost.org/doc/libs/1_68_0/more/getting_started/unix-variants.html#library-naming
#
#  NOTE: We are looking for a static library containing just filesystem and system modules.
#        This library must be merged into a shared library so it must be compiled with -fPIC.
#        This is the reason we cannot use an externally installed version.
#
#        We may not need Boost at all.  If using C++17 standard and the compiler version supports
#        std::filesystem then we will skip this module entirely. See logic in CommonCompilerConfig.cmake.
#
#######################################




# Download the boost distribution (at configure time).
message(STATUS "obtaining Boost (download & install takes a while, go get a coffee :) ...")
include(DownloadProject/DownloadProject.cmake)

  if (MSVC OR MSYS OR MINGW)
    set(BOOST_URL "${REPOSITORY_DIR}/external/common/share/boost/boost_1_69_0_subset.zip")
    set(BOOST_HASH "d074bcbcc0501c4917b965fc890e303ee70d8b01ff5712bae4a6c54f2b6b4e52")
  else()
    set(BOOST_URL "https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz")
    set(BOOST_HASH "9a2c2819310839ea373f42d69e733c339b4e9a19deab6bfec448281554aa4dbb")
  endif()

#  note: this will not download if it already exists.
download_project(PROJ Boost_download
	PREFIX ${EP_BASE}/boost
	URL ${BOOST_URL}
	URL_HASH SHA256=${BOOST_HASH}
	UPDATE_DISCONNECTED 1
	QUIET
	)
	

# Set some parameters
set(BOOST_ROOT ${Boost_download_SOURCE_DIR})
set(BOOST_ROOT ${BOOST_ROOT} CACHE STRING  "BOOST_ROOT points to the boost Installation." FORCE)
set(Boost_INCLUDE_DIRS ${BOOST_ROOT})

file(GLOB Boost_LIBRARIES ${BOOST_ROOT}/stage/lib/*)
set(qty_libs 0)
if(Boost_LIBRARIES)
  list(LENGTH ${Boost_LIBRARIES} qty_libs)
endif()
if(${qty_libs} LESS 2)
  message(STATUS "Boost being installed at BOOST_ROOT = ${BOOST_ROOT}")
  if (MSVC OR MSYS OR MINGW)
    if (MSYS OR MINGW)
	  set(bootstrap "bootstrap.bat gcc")
	  set(toolset toolset=gcc architecture=x86)
    elseif(MSVC)
	  set(bootstrap "bootstrap.bat vc141")
	  set(toolset toolset=msvc-15.0 architecture=x86)
    endif()
  else()
    set(bootstrap "./bootstrap.sh")
    set(toolset) # b2 will figure out the toolset
  endif()
  # On Windows this will build 4 libraries per module.  32/64bit and release/debug variants
  # All will be Static, multithreaded, shared runtime link, compiled with -fPIC
  
  execute_process(COMMAND "${bootstrap}" 
        WORKING_DIRECTORY ${BOOST_ROOT}
	OUTPUT_QUIET
	RESULT_VARIABLE error_result
	)
  if(error_result)
    message(FATAL_ERROR "Boost bootstrap has errors.   ${error_result}")
  else()
    execute_process(COMMAND "./b2"
  	--prefix=${BOOST_ROOT}
  	--with-filesystem 
	--with-system 
	--layout=system
	variant=release
	threading=multi 
	runtime-link=shared 
	link=static 
	cxxflags="-fPIC"
	stage
  	WORKING_DIRECTORY ${BOOST_ROOT} 
	OUTPUT_QUIET
	RESULT_VARIABLE error_result
  	)
    if(error_result)
      message(FATAL_ERROR "Boost build has errors. ${error_result}")
    else()
      file(GLOB Boost_LIBRARIES ${BOOST_ROOT}/stage/lib/*)
      set(Boost_INCLUDE_DIRS ${BOOST_ROOT})
    endif()
  endif()
endif()

set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} PARENT_SCOPE)
set(Boost_LIBRARIES ${Boost_LIBRARIES} PARENT_SCOPE)
message(STATUS "  Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
message(STATUS "  Boost_LIBRARIES = ${Boost_LIBRARIES}")




