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
  PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}
)

include(ExternalProject)

ExternalProject_Add(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.7.0
  CMAKE_ARGS
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(gtest source_dir)
MESSAGE(STATUS "source_dir" ${source_dir})
ExternalProject_Get_Property(gtest binary_dir)
MESSAGE(STATUS "binary_dir" ${binary_dir})

ExternalProject_Add(
  CapnProto
  GIT_REPOSITORY https://github.com/sandstorm-io/capnproto.git
  GIT_TAG v0.5.2
  DEPENDS gtest
  #CMAKE_ARGS
  #    #-DBUILD_TESTING=OFF
  #    -DUSE_EXTERNAL_GTEST=ON
  #    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  #    -Dgtest_library=${binary_dir}/libgtest.a
  #    -Dgtest_main_library=${binary_dir}/libgtest_main.a
  #    -Dgtest_include_dir=${source_dir}/include
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  # Since CMakeLists.txt isn't in the repo root, we have to point
  # SOURCE_DIR to the "c++" dir inside the DOWNLOAD_DIR that contains it.
  SOURCE_DIR ${CMAKE_BINARY_DIR}/ThirdParty/src/CapnProto
  CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      -DUSE_EXTERNAL_GTEST=ON
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -Dgtest_library=${binary_dir}/libgtest.a
      -Dgtest_main_library=${binary_dir}/libgtest_main.a
      -Dgtest_include_dir=${source_dir}/include
      -G "Unix Makefiles"
      ${CMAKE_BINARY_DIR}/ThirdParty/src/CapnProto/c++
  #DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/ThirdParty/src/CapnProto
)
ExternalProject_Get_Property(CapnProto source_dir)
MESSAGE(STATUS "source_dir: " ${source_dir})
ExternalProject_Get_Property(CapnProto binary_dir)
MESSAGE(STATUS "binary_dir: " ${binary_dir})

# TODO: needed?
#ExternalProject_Add_Step(CapnProto prebuild
#  COMMAND ${CMAKE_COMMAND} -E make_directory ${binary_dir}
#  DEPENDEES download
#  DEPENDEES configure
#)

set(CAPNP_LIBRARY "${binary_dir}/c++/libcapnp.so")
set(CAPNP_RPC_LIBRARY "${binary_dir}/c++/libcapnp-rpc.so")
set(CAPNPC_LIBRARY "${binary_dir}/c++/libcapnpc.so")
set(KJ_LIBRARY "${binary_dir}/c++/libkj.so")
set(KJ_ASYNC_LIBRARY "${binary_dir}/c++/libkj-async.so")
set(CAPNPC_EXECUTABLE "${binary_dir}/c++/capnp")
set(CAPNPC_CXX_EXECUTABLE "${binary_dir}/c++/capnpc-c++")
set(CAPNP_INCLUDE_DIRECTORIES "${source_dir}/c++/src")
list(APPEND CAPNP_LIBRARIES ${CAPNP_RPC_LIBRARY})
list(APPEND CAPNP_LIBRARIES ${CAPNP_LIBRARY})
list(APPEND CAPNP_LIBRARIES ${CAPNPC_LIBRARY})
list(APPEND CAPNP_LIBRARIES ${KJ_ASYNC_LIBRARY})
list(APPEND CAPNP_LIBRARIES ${KJ_LIBRARY})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${source_dir}/c++/cmake PARENT_SCOPE)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${binary_dir} PARENT_SCOPE)

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
