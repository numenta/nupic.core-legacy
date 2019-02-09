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

# Fetch MNIST dataset from online archive
#
if(EXISTS ${REPOSITORY_DIR}/build/ThirdParty/share/eigen-master.zip)
    set(URL ${REPOSITORY_DIR}/build/ThirdParty/share/eigen-master.zip)
else()
    set(URL "https://github.com/wichtounet/mnist/archive/master.zip")
    set(HASH "0bcecef3aaed7c619f0baac02697a76dff3da54f0c17f1aaaa92a2928ca300f3")
endif()

message(STATUS "obtaining MNIST data")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ mnist
	            PREFIX ${EP_BASE}/mnist_data
		    URL    ${URL}
		    URL_HASH SHA256=${HASH}
                    UPDATE_DISCONNECTED 1
		    #           QUIET
   )	
# No build. This is a data only package
add_subdirectory(${mnist_SOURCE_DIR}/example/ ${mnist_BINARY_DIR})
set(mnist_INCLUDE_DIRS ${mnist_SOURCE_DIR}/include/mnist)
if (MSVC)
set(mnist_LIBRARIES   "${mnist_BINARY_DIR}$<$<CONFIG:Release>:/Release/mnist.lib>$<$<CONFIG:Debug>:/Debug/mnist.lib>")
else()
set(mnist_LIBRARIES   ${mnist_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mnist${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()
FILE(APPEND "${EXPORT_FILE_NAME}" "mnist_INCLUDE_DIRS@@@${mnist_SOURCE_DIR}/include/mnist\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "mnist_LIBRARIES@@@${mnist_LIBRARIES}\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "mnist_SOURCE_DIR@@@${mnist_SOURCE_DIR}\n")
