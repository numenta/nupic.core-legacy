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

  # This will fetch gtest from the repository at configure time.
  # The FetchContent directive requires CMake 3.11  
  include(FetchContent)
  if(MSVC)
    set(url https://github.com/abseil/googletest/archive/release-1.8.1.zip)
  else()
    set(url https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz)
  endif()
  message(STATUS "Fetching gtest from ${url}")
  FetchContent_Declare( googletest
#    GIT_REPOSITORY https://github.com/google/googletest.git
#    GIT_TAG        release-1.8.1
    URL ${url}
  )
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()
