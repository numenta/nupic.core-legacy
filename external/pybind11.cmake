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
if(EXISTS ${REPOSITORY_DIR}/build/ThirdParty/share/pybind11.tar.gz)
    set(URL ${REPOSITORY_DIR}/build/ThirdParty/share/pybind11.tar.gz)
else()
    set(URL https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz)
endif()

message(STATUS "obtaining PyBind11")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ pybind11
	PREFIX ${EP_BASE}/pybind11
	URL ${URL}
	UPDATE_DISCONNECTED 1
	QUIET
	)
	
# No build. This is a header only package


FILE(APPEND "${EXPORT_FILE_NAME}" "pybind11_SOURCE_DIR@@@${pybind11_SOURCE_DIR}\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "pybind11_BINARY_DIR@@@${pybind11_BINARY_DIR}\n")
