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

# Fetch MNIST dataset from online archive
#
if(EXISTS "${REPOSITORY_DIR}/build/ThirdParty/share/mnist.zip")
    set(URL "${REPOSITORY_DIR}/build/ThirdParty/share/mnist.zip")
else()
    set(URL "https://github.com/wichtounet/mnist/archive/3b65c35ede53b687376c4302eeb44fdf76e0129b.zip")
endif()

message(STATUS "obtaining MNIST data")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ mnist
	PREFIX ${EP_BASE}/mnist_data
	URL    ${URL}
	#URL_HASH SHA256=${HASH}
    UPDATE_DISCONNECTED 1
#    QUIET
   )	
# Note: no check for failure.  This package is optional.
   
# No build. This is a data only package
# But we do need to run its CMakeLists.txt to unpack the files.

add_subdirectory(${mnist_SOURCE_DIR}/example/ ${mnist_BINARY_DIR})
FILE(APPEND "${EXPORT_FILE_NAME}" "mnist_INCLUDE_DIRS@@@${mnist_SOURCE_DIR}/include\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "mnist_SOURCE_DIR@@@${mnist_SOURCE_DIR}\n")

# includes will be found with  #include <mnist/mnist_reader_less.hpp>
# data will be found in folder ${mnist_SOURCE_DIR}
