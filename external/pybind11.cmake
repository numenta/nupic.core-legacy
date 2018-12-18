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

# Fetch pybind11 from GitHub archive
#

message(STATUS "obtaining PyBind11")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ pybind11
	PREFIX ${EP_BASE}/pybind11
	URL https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz
	UPDATE_DISCONNECTED 1
	QUIET
	)
	
# Cannot do the build here. It depends on which Python is being executed.

set(pybind11_SOURCE_DIR ${pybind11_SOURCE_DIR} CACHE BOOL "location of pybind11 sources" FORCE) 
set(pybind11_BINARY_DIR ${pybind11_BINARY_DIR} CACHE BOOL "location of pybind11 binary" FORCE) 

