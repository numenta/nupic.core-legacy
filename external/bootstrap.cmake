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
#
#######################################
# This bootstrap cmake file is used to start the ThirdParty
# build with its own cache.  Once built, the main build
# will not be able to affect these file even if doing a 'make clean'.
# If you need to re-build the ThirdParty libaries, 
#  - cd to repository
#  - delete build  (rm -r build)
#  - create build/scripts (mkdir -d build/scripts)
#  - cd build/scripts
#  - cmake ../..

if (MSVC)
  set(build_type --config Release)
else()
  set(build_type CMAKE_BUILD_TYPE=Release)
endif()


FILE(MAKE_DIRECTORY  ${REPOSITORY_DIR}/build/ThirdParty)
execute_process(COMMAND ${CMAKE_COMMAND} 
                        -G ${CMAKE_GENERATOR}
			-D CMAKE_INSTALL_PREFIX=. 
                        -D NEEDS_BOOST:BOOL=${NEEDS_BOOST}
                        -D BINDING_BUILD:STRING=${BINDING_BUILD}
			-D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
			 ../../external
                WORKING_DIRECTORY ${REPOSITORY_DIR}/build/ThirdParty
                RESULT_VARIABLE result
#                OUTPUT_QUIET      ### Disable this to debug external configuration
	)
if(result)
    message(FATAL_ERROR "CMake step for Third Party builds failed: ${result}")
endif()

if(MSVC)
  # We need to build for both Release and Debug for MSVC because
  # it will not re-run this build if the build_type changes in ide.
  execute_process(COMMAND ${CMAKE_COMMAND} --build . --config Release
                    WORKING_DIRECTORY ${REPOSITORY_DIR}/build/ThirdParty
                    RESULT_VARIABLE result
  #                    OUTPUT_QUIET      ### Disable this to debug external buiilds
  )
  if(result)
    message(FATAL_ERROR "build step for MSVC Release Third Party builds failed: ${result}")
  endif()
  
  execute_process(COMMAND ${CMAKE_COMMAND} --build . --config Debug
                    WORKING_DIRECTORY ${REPOSITORY_DIR}/build/ThirdParty
                    RESULT_VARIABLE result
  #                    OUTPUT_QUIET      ### Disable this to debug external buiilds
  )
  if(result)
    message(FATAL_ERROR "build step for MSVC Debug Third Party builds failed: ${result}")
  endif()
  
else(MSVC)
  # for linux we only do this once for the current build_type.  To switch
  # build type the user would have to clear everyting and re-run cmake.
  execute_process(COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} 
                    WORKING_DIRECTORY ${REPOSITORY_DIR}/build/ThirdParty
                    RESULT_VARIABLE result
#                    OUTPUT_QUIET      ### Disable this to debug external buiilds
  )
  if(result)
    message(FATAL_ERROR "build step for Third Party builds failed: ${result}")
  endif()
endif(MSVC)

# extract the external directory paths
#    The external third party modules are being built
#    in a seperate build with a different cache.  So variables
#    being passed back must be passed via a file.
message(STATUS "Results from external build:")
set(IMPORT_FILE "${REPOSITORY_DIR}/build/ThirdParty/results.txt")
FILE(READ "${IMPORT_FILE}" contents)
STRING(REGEX REPLACE "\n" ";" lines "${contents}")
FOREACH(line ${lines})
  STRING(REGEX REPLACE "@@@" ";" line "${line}")
  LIST(GET line 0 name)
  LIST(REMOVE_AT line 0)
  SET(${name} ${line})
  message(STATUS "  ${name} = ${${name}}")
ENDFOREACH()
set(EXTERNAL_INCLUDES
	${yaml-cpp_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${eigen_INCLUDE_DIRS}
	${REPOSITORY_DIR}/external/common/include
)

