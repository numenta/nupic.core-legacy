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
#
# This will load the gtest module.
# exports 'gtest' as a target
#


# Prior to gtest 1.8.1, gtest required 'std::tr1::tuple' which does not exist in MSVC 2015,2017
# The latest version of gtest fixes the problem as follows, so we need at least gtest 1.8.1.
# Summary of tuple support for Microsoft Visual Studio:
# Compiler    version(MS)  version(cmake)  Support
# ----------  -----------  --------------  -----------------------------
# <= VS 2010  <= 10        <= 1600         Use Google Tests's own tuple.
# VS 2012     11           1700            std::tr1::tuple + _VARIADIC_MAX=10
# VS 2013     12           1800            std::tr1::tuple
# VS 2015     14           1900            std::tuple
# VS 2017     15           >= 1910         std::tuple

if(EXISTS ${REPOSITORY_DIR}/build/ThirdParty/share/googletest-release-1.8.1.tar.gz)
    set(URL ${REPOSITORY_DIR}/build/ThirdParty/share/googletest-release-1.8.1.tar.gz)
else()
    set(URL https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz)
endif()

#
# Build gtest lib
#
message(STATUS "Obtaining gtest")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ googletest
	PREFIX ${EP_BASE}/gtest
	URL https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz
	UPDATE_DISCONNECTED 1
	QUIET
	)
set(INSTALL_GTEST OFF CACHE BOOL "prevents installing gtest" FORCE)
set(BUILD_GMOCK   OFF CACHE BOOL "prevents building gmock"   FORCE)
#set(gtest_force_shared_crt ON CACHE BOOL "Prevent GoogleTest from overriding our compiler/linker options" FORCE)
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

if(MSVC)
  set(gtest_LIBRARIES ${googletest_BINARY_DIR}/googletest/$<$<CONFIG:Release>:Release>$<$<CONFIG:Debug>:Debug>/${CMAKE_STATIC_LIBRARY_PREFIX}gtest$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEBUG_POSTFIX d)
  endif()
  set(gtest_LIBRARIES ${googletest_BINARY_DIR}/googletest/${CMAKE_STATIC_LIBRARY_PREFIX}gtest$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()
FILE(APPEND "${EXPORT_FILE_NAME}" "gtest_INCLUDE_DIRS@@@${googletest_SOURCE_DIR}/googletest/include\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "gtest_LIBRARIES@@@${gtest_LIBRARIES}\n")


