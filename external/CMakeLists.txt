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
#  Features:
#  - All third party modules are downloaded form the internet and build.
#
#  - Isolated so does not require being downloaded more than once.
#
#  - It is possible to manually preload the distribution packages (for air gap computers)
#     1) On some other machine, download the following:
#        https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
#        https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz
#        https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz
#        https://github.com/abseil/googletest/archive/release-1.8.1.tar.gz
#        http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2
#     2) copy these to <repository>/build/ThirdParty/share/
#
#  NOTE:  This is executed by the top level CMakeLists.txt via the bootstrap.cmake
#         by executing "cd build/scripts; cmake ../.."
#         Resulting paths are passed back via the file "build/ThirdParty/results.txt".
#
#         This has a separate cache so not affected by make clean.
#          
# If you need to re-build the ThirdParty libaries, 
#  - cd to repository
#  - delete build  (rm -r build)
#  - create folders build/scripts (mkdir -d build/scripts)
#  - cd build/scripts
#  - cmake ../..
# Note: on MSVC, run startupMSVC.bat after deleting the build folder.

cmake_minimum_required(VERSION 3.3)
project(nupic_core CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(GNUInstallDirs)
include(ExternalProject)
set(EP_BASE ${CMAKE_BINARY_DIR})
STRING(REGEX REPLACE "@@@" " " EXTERNAL_CXX_FLAGS "${EXTERNAL_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${EXTERNAL_CXX_FLAGS}" CACHE STRING "Sets compile flags" FORCE)


message(STATUS "external import")
message(STATUS "   CMAKE_INSTALL_PREFIX  = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "   EXTERNAL_CXX_FLAGS    = ${EXTERNAL_CXX_FLAGS}")
message(STATUS "   NEEDS_BOOST           = ${NEEDS_BOOST}")
message(STATUS "   BINDING_BUILD         = ${BINDING_BUILD}")


set(EXPORT_FILE_NAME "${EP_BASE}/results.txt")
FILE(REMOVE "${EXPORT_FILE_NAME}")

################
# Yaml-cpp
include(YamlCppLib.cmake)


################
# Eigen
include(eigen.cmake)


##################
#  Boost
if(NEEDS_BOOST)
  include(Boost.cmake)
endif()


##################
# gtest
include(gtest.cmake)


##################
# pybind11
string(REGEX MATCH "Python" match ${BINDING_BUILD})
if(match)
  include(pybind11.cmake)
endif()



