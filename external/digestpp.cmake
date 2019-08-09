# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Numenta, Inc.
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
# -----------------------------------------------------------------------------

# Fetch digestpp hash digest package from GitHub archive
#
if(EXISTS "${REPOSITORY_DIR}/build/ThirdParty/share/digestpp.zip")
    set(URL "${REPOSITORY_DIR}/build/ThirdParty/share/digestpp.zip")
else()
    set(URL https://github.com/kerukuro/digestpp/archive/36fa6ca2b85808bd171b13b65a345130dbe1d774.zip)
endif()

message(STATUS "obtaining digestpp")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ digestpp
	PREFIX ${EP_BASE}/digestpp
	URL ${URL}
	UPDATE_DISCONNECTED 1
	QUIET
	)

# No build. This is a header only package


FILE(APPEND "${EXPORT_FILE_NAME}" "digestpp_INCLUDE_DIRS@@@${digestpp_SOURCE_DIR}\n")
