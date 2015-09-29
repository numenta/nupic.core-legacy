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

cmake_minimum_required(VERSION 2.8)
project(nupic_core_externals)

set(CMAKE_VERBOSE_MAKEFILE ON)

set_directory_properties(
  PROPERTIES EP_PREFIX ${EP_PREFIX}
)

include(ExternalProject)

ExternalProject_Add(
  CapnProto
  GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  GIT_TAG v0.5.2
  UPDATE_COMMAND ""
  #INSTALL_COMMAND ""
  CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_TESTING=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -G "Unix Makefiles"
      ${EP_PREFIX}/src/CapnProto/c++
)
#ExternalProject_Get_Property(CapnProto source_dir)
#MESSAGE(STATUS "source_dir: " ${source_dir})
#ExternalProject_Get_Property(CapnProto binary_dir)
#MESSAGE(STATUS "binary_dir: " ${binary_dir})

# TODO: needed?
#ExternalProject_Add_Step(CapnProto prebuild
#  COMMAND ${CMAKE_COMMAND} -E make_directory ${binary_dir}
#  DEPENDEES download
#  DEPENDEES configure
#)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${source_dir}/c++/cmake PARENT_SCOPE)
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${binary_dir} PARENT_SCOPE)

# TODO: This makes CapnProto depend on gtest right? Shouldn't it already from
# ExternalProject_Add?
#add_dependencies(CapnProto gtest)

#install(TARGETS CapnProto EXPORT CapnProto)
#export(TARGETS CapnProto FILE ${CMAKE_BINARY_DIR}/CapnProto-config.cmake)
#export(PACKAGE CapnProto)

#ExternalProject_Get_Property(CapnProto cmake_command)
#MESSAGE(STATUS "cmake_command: " ${cmake_command})
#ExternalProject_Get_Property(CapnProto cmake_args)
#MESSAGE(STATUS "cmake_args: " ${cmake_args})
#include_directories(${source_dir}/include)
