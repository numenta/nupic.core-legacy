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

# Fetch Cereal serialization package from GitHub archive
#
if(EXISTS "${REPOSITORY_DIR}/build/ThirdParty/share/cereal.tar.gz")
    set(URL "${REPOSITORY_DIR}/build/ThirdParty/share/cereal.tar.gz")
else()
    set(URL https://github.com/USCiLab/cereal/archive/v1.2.2.tar.gz)
endif()

message(STATUS "obtaining Cereal")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ cereal
	PREFIX ${EP_BASE}/cereal
	URL ${URL}
	UPDATE_DISCONNECTED 1
	QUIET
	)
	
# No build. This is a header only package


FILE(APPEND "${EXPORT_FILE_NAME}" "cereal_INCLUDE_DIRS@@@${cereal_SOURCE_DIR}/include\n")
