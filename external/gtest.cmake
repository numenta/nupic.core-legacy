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

#
# Build gtest lib
#
add_library(gtest STATIC ${REPOSITORY_DIR}/external/common/src/gtest/gtest-all.cpp)
target_compile_options(gtest PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(gtest PUBLIC -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
target_include_directories(gtest PUBLIC ${REPOSITORY_DIR}/external/common/include)

