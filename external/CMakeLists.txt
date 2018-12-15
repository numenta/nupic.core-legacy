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
include(ExternalProject)
set(EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

################
# Yaml-cpp
include(YamlCppLib.cmake)

##################
#  Boost
include(Boost.cmake)

##################
# gtest
include(gtest.cmake)

##################
# pybind11
if(PYTHON2_BUILD OR PYTHON3_BUILD)
  include(pybind11.cmake)
endif()
