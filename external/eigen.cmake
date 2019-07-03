# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
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

# Fetch Eigen from GitHub archive
#
if(EXISTS "${REPOSITORY_DIR}/build/ThirdParty/share/eigen.tar.bz2")
    set(URL "${REPOSITORY_DIR}/build/ThirdParty/share/eigen.tar.bz2")
else()
    set(URL http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2)
endif()

message(STATUS "obtaining Eigen")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ eigen
	PREFIX ${EP_BASE}/eigen
	URL ${URL}
	UPDATE_DISCONNECTED 1
	QUIET
	)
	
# No build. This is a header only package


FILE(APPEND "${EXPORT_FILE_NAME}" "eigen_INCLUDE_DIRS@@@${eigen_SOURCE_DIR}\n")
