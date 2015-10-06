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
project(nupic_core_main CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(EP_BASE ${CMAKE_BINARY_DIR}/ThirdParty)

# Set up external dependencies.
#
# This will try find_library and if that fails use ExternalProject to fetch
# and build the library.
#
# Sets the following library targets:
#   kj
#   capnp
#   capnpc
#
# Sets the following executable targets:
#   bin_capnp (capnp)
#   bin_capnpc_cpp (capnpc-c++)
add_subdirectory(external)

# Make sure we can link against the external libraries.
include_directories(SYSTEM ${EP_BASE}/Install/include)

# Now build nupic_core project.
add_subdirectory(src)
