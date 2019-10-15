# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have purchased from
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
# This downloads and builds the libyaml library.
#
# libyaml  - This is a SAX parser which means that it performs callbacks
#            for each token it parses from the yaml text.  Therefore
#            the interface (Value.cpp) must create the internal structure.
#            This is a YAML 1.1 implementation.
#
#            The current release is 0.2.2
#            The repository is at https://github.com/yaml/libyaml
#            Documentation is at https://pyyaml.org/wiki/LibYAML
#
#            There is a problem in version 0.2.2; it does not build.
#            Use the current master.
#
if(EXISTS   ${REPOSITORY_DIR}/build/ThirdParty/share/libyaml.zip)
    set(URL ${REPOSITORY_DIR}/build/ThirdParty/share/libyaml.zip)
elif(EXISTS ${REPOSITORY_DIR}/build/ThirdParty/share/libyaml.tar.gz)
    set(URL ${REPOSITORY_DIR}/build/ThirdParty/share/libyaml.tar.gz)
else()
    set(URL "https://github.com/yaml/libyaml/archive/master.zip")
    #set(URL "http://pyyaml.org/download/libyaml/yaml-0.2.2.tar.gz")
endif()

message(STATUS "Obtaining libyaml")
include(DownloadProject/DownloadProject.cmake)
download_project(PROJ libyaml
	PREFIX ${EP_BASE}/libyaml
	URL ${URL}
	GIT_SHALLOW ON
	UPDATE_DISCONNECTED 1
	QUIET
	)
    
set(YAML_DECLARE_STATIC ON)
set(YAML_STATIC_LIB_NAME "yaml" CACHE STRING "The core library name." )
add_subdirectory(${libyaml_SOURCE_DIR} ${libyaml_BINARY_DIR})

set(yaml_INCLUDE_DIRS ${libyaml_SOURCE_DIR}/include) 
if (MSVC)
  set(yaml_LIBRARIES   "${libyaml_BINARY_DIR}$<$<CONFIG:Release>:/Release/yaml.lib>$<$<CONFIG:Debug>:/Debug/yaml.lib>") 
else()
  set(yaml_LIBRARIES   ${libyaml_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}yaml${CMAKE_STATIC_LIBRARY_SUFFIX}) 
endif()
FILE(APPEND "${EXPORT_FILE_NAME}" "yaml_INCLUDE_DIRS@@@${yaml_INCLUDE_DIRS}\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "yaml_LIBRARIES@@@${yaml_LIBRARIES}\n")
FILE(APPEND "${EXPORT_FILE_NAME}" "yaml_DEFINE@@@YAML_PARSER_libYaml\n")

