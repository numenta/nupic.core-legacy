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

#  Fetch pybind11 from Git
# This will download pybind11 at configure time, and add_subdirectory it. Then you're ready to call pybind11_add_module.
# See https://stackoverflow.com/questions/47027741/smart-way-to-cmake-a-project-using-pybind11-by-externalproject-add?rq=1
include(FetchContent)
FetchContent_Declare(
	pybind11
    FETCHCONTENT_BASE_DIR ${EP_BASE}
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()	

