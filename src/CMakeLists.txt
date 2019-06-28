# -----------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2013-2015, Numenta, Inc.
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

cmake_minimum_required(VERSION 3.7)
project(htm_core CXX)

message(STATUS "Configuring htm_core src")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE ON) # toggle for cmake debug
include(src/NupicLibraryUtils) # for combining static libs
include(src/ExpandStaticLib)   # for creating objects from a static lib

#
# NuPIC CMake options
#
option(HTM_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program, with a minimum CMake version of 3.3.")
if(${HTM_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()

#--------------------------------------------------------
# Identify includes from this directory
set(CORE_LIB_INCLUDES  ${PROJECT_SOURCE_DIR}  # for htm/xxx/*.h
		       ${PROJECT_BINARY_DIR}  # for Version.hpp
)
set(CORE_LIB_INCLUDES  ${CORE_LIB_INCLUDES} PARENT_SCOPE)


#--------------------------------------------------------
# Create Version.hpp
# Extract current git sha and record in htm/Version.hpp.in
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE HTM_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/htm/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")
  

############ htm_core Library ######################################
# Setup htm_core static library, consisting of our own sources and yaml libraries;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the htm_core "combined" static
# library.
#
set(src_objlib LibrarySource)


set(algorithm_files
    htm/algorithms/Anomaly.cpp
    htm/algorithms/Anomaly.hpp
    htm/algorithms/AnomalyLikelihood.cpp
    htm/algorithms/AnomalyLikelihood.hpp
    htm/algorithms/Connections.cpp
    htm/algorithms/Connections.hpp
    htm/algorithms/SDRClassifier.cpp
    htm/algorithms/SDRClassifier.hpp
    htm/algorithms/SpatialPooler.cpp
    htm/algorithms/SpatialPooler.hpp
    htm/algorithms/TemporalMemory.cpp
    htm/algorithms/TemporalMemory.hpp
)


set(encoders_files 
    htm/encoders/BaseEncoder.hpp
    htm/encoders/ScalarEncoder.cpp
    htm/encoders/ScalarEncoder.hpp
    htm/encoders/RandomDistributedScalarEncoder.hpp
    htm/encoders/RandomDistributedScalarEncoder.cpp
)
    
set(engine_files
    htm/engine/Input.cpp
    htm/engine/Input.hpp
    htm/engine/Link.cpp
    htm/engine/Link.hpp
    htm/engine/Network.cpp
    htm/engine/Network.hpp
    htm/engine/NuPIC.cpp
    htm/engine/NuPIC.hpp
    htm/engine/Output.cpp
    htm/engine/Output.hpp
    htm/engine/Region.cpp
    htm/engine/Region.hpp
    htm/engine/RegionImpl.cpp
    htm/engine/RegionImpl.hpp
    htm/engine/RegionImplFactory.cpp
    htm/engine/RegionImplFactory.hpp
    htm/engine/RegisteredRegionImpl.hpp
    htm/engine/RegisteredRegionImplCpp.hpp
    htm/engine/Spec.cpp
    htm/engine/Spec.hpp
    htm/engine/YAMLUtils.cpp
    htm/engine/YAMLUtils.hpp
    htm/engine/Watcher.cpp
    htm/engine/Watcher.hpp
)


set(ntypes_files
    htm/ntypes/Array.hpp
    htm/ntypes/ArrayBase.cpp
    htm/ntypes/ArrayBase.hpp
    htm/ntypes/BasicType.cpp
    htm/ntypes/BasicType.hpp
    htm/ntypes/Collection.hpp
    htm/ntypes/Dimensions.hpp
    htm/ntypes/Scalar.cpp
    htm/ntypes/Scalar.hpp
    htm/ntypes/Value.cpp
    htm/ntypes/Value.hpp
)
  
set(os_files
    htm/os/Directory.cpp
    htm/os/Directory.hpp
    htm/os/Env.cpp
    htm/os/Env.hpp
    htm/os/ImportFilesystem.hpp
    htm/os/OS.cpp
    htm/os/OS.hpp
    htm/os/OSUnix.cpp
    htm/os/OSWin.cpp
    htm/os/Path.cpp
    htm/os/Path.hpp
    htm/os/Timer.cpp
    htm/os/Timer.hpp    
)

set(regions_files
    htm/regions/ScalarSensor.cpp
    htm/regions/ScalarSensor.hpp
    htm/regions/SPRegion.cpp
    htm/regions/SPRegion.hpp
    htm/regions/TestNode.cpp
    htm/regions/TestNode.hpp
    htm/regions/TMRegion.cpp
    htm/regions/TMRegion.hpp
    htm/regions/VectorFile.cpp
    htm/regions/VectorFile.hpp
    htm/regions/VectorFileEffector.cpp
    htm/regions/VectorFileEffector.hpp
    htm/regions/VectorFileSensor.cpp
    htm/regions/VectorFileSensor.hpp   
)

set(types_files
    htm/types/Exception.hpp
    htm/types/Types.hpp
    htm/types/Serializable.hpp
    htm/types/Sdr.hpp
    htm/types/Sdr.cpp
)

set(utils_files
    htm/utils/GroupBy.hpp
    htm/utils/Log.hpp
    htm/utils/LoggingException.cpp
    htm/utils/LoggingException.hpp
    htm/utils/LogItem.cpp
    htm/utils/LogItem.hpp
    htm/utils/MovingAverage.cpp
    htm/utils/MovingAverage.hpp
    htm/utils/Random.cpp
    htm/utils/Random.hpp
    htm/utils/SlidingWindow.hpp
    htm/utils/VectorHelpers.hpp
    htm/utils/SdrMetrics.cpp
    htm/utils/SdrMetrics.hpp
    htm/utils/StlIo.cpp
    htm/utils/StlIo.hpp
    htm/utils/Topology.cpp
    htm/utils/Topology.hpp
)

set(examples_files
    examples/hotgym/Hotgym.cpp # contains conflicting main()
    examples/hotgym/HelloSPTP.cpp
    examples/hotgym/HelloSPTP.hpp
    examples/mnist/MNIST_SP.cpp
)

#set up file tabs in Visual Studio
source_group("algorithms" FILES ${algorithm_files})
source_group("encoders" FILES ${encoders_files})
source_group("engine" FILES ${engine_files})
source_group("ntypes" FILES ${ntypes_files})
source_group("os" FILES ${os_files})
source_group("regions" FILES ${regions_files})
source_group("types" FILES ${types_files})
source_group("utils" FILES ${utils_files})
source_group("examples" FILES ${examples_files})


#--------------------------------------------------------
# Create an OBJECT library (compiles sources only once)
# from the CMake docs:
#   An object library compiles source files but does not archive or link their object 
#   files into a library. Instead other targets created by  add_library() or add_executable() 
#   may reference the objects using an expression of the form $<TARGET_OBJECTS:objlib> as a 
#   source, where objlib is the object library name.
#
# The price you pay is that the object files must be built as position-independent code.
#
add_library(${src_objlib} OBJECT 
    ${algorithm_files} 
    ${encoders_files} 
    ${engine_files} 
    ${ntypes_files} 
    ${os_files} 
    ${regions_files} 
    ${types_files} 
    ${utils_files}
)
# shared libraries need PIC
target_compile_options( ${src_objlib} PUBLIC ${INTERNAL_CXX_FLAGS})
if(MSVC)
  set_property(TARGET ${src_objlib} PROPERTY LINK_LIBRARIES ${INTERNAL_LINKER_FLAGS})
endif()
target_compile_definitions(${src_objlib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${src_objlib} PRIVATE 
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		)

if(${HTM_IWYU})
  set_target_properties(${src_objlib}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()

############ Building Static LIB #############################################
# build static libhtm_core_solo.a without yaml and boost.
# uses objects compiled for src_objlib

set(src_lib_static htm_core_solo)
add_library(${src_lib_static} STATIC $<TARGET_OBJECTS:${src_objlib}>)
if(MSVC)
  set_property(TARGET ${src_lib_static} PROPERTY LINK_LIBRARIES ${INTERNAL_LINKER_FLAGS})
endif()
#
# Create the libhtm_core.a as a "combined" static library by merging htm_core_static
# with our external static libraries
# NOTE: cannot use target_link_libraries( ) with the add_library for STATIC.
#       so we have to merge them in a different way.
#
set(src_combined_htmcore_source_archives
    ${src_lib_static}
    ${yaml-cpp_LIBRARIES}
    ${Boost_LIBRARIES}
    ${common_LIBRARIES}
    )
merge_static_libraries(${core_library} "${src_combined_htmcore_source_archives}")

############ Building Shared LIB #############################################
# builds libhtm_core.so for Linux,OSx and 
# NOTE: This does not work for Windows.
#       If there is community demand for a Windows DLL interface, we can add it later.
#
# For Linux, OSx; the .so file will contain all symbols.
#
if(MSVC)
	# NOTE: this disables shared lib for Windows.
else()
    # First extract static libraries into an Object Library
	# Note that these libraries must already exist (from depenancies)
	set(all_object_locations)
	set(src_externLibs
            ${yaml-cpp_LIBRARIES}
            ${Boost_LIBRARIES}
            ${common_LIBRARIES}
            )
	set(lib_locations_separator "++++")
    string(REPLACE ";" ${lib_locations_separator}
           escaped_lib_locations_arg "${src_externLibs}")

	ExpandStaticLib( ${src_lib_shared}
	    "${escaped_lib_locations_arg}"
	    "${lib_locations_separator}"
	    "${CMAKE_CURRENT_BINARY_DIR}"
	    "${CMAKE_AR}"
	    )
	# build shared lib with yaml and boost resolved
	# uses objects compiled for src_objlib
	add_library(${src_lib_shared} SHARED 
		$<TARGET_OBJECTS:${src_objlib}>
		${all_object_locations})
	set_target_properties(${src_lib_shared} PROPERTIES OUTPUT_NAME ${core_library})
	target_link_libraries(${src_lib_shared}  
		${COMMON_OS_LIBS}
	        )
endif()



		
############ TEST #############################################
# Test
# The tests were placed in a separate file to avoid clutering up this file.
add_subdirectory(test)


########### EXAMPLES ##############################################
## Example custom executable
#
## Setup benchmark_hotgym
#
set(src_executable_hotgym benchmark_hotgym)
add_executable(${src_executable_hotgym} examples/hotgym/Hotgym.cpp examples/hotgym/HelloSPTP.hpp examples/hotgym/HelloSPTP.cpp)
# link with the static library
target_link_libraries(${src_executable_hotgym} 
    ${INTERNAL_LINKER_FLAGS}
    ${core_library}
    ${COMMON_OS_LIBS}
)
    
# for future reference; to link to shared library, (Not on MSVC)
#          link with ${src_lib_shared} rather than ${core_library}
                  
target_compile_options( ${src_executable_hotgym} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${src_executable_hotgym} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${src_executable_hotgym} PRIVATE 
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		)
add_custom_target(hotgym
          COMMAND ${src_executable_hotgym}
          DEPENDS ${src_executable_hotgym}
          COMMENT "Executing ${src_executable_hotgym}"
          VERBATIM)

#########################################################
## Dynamicly linked version of hotgym
if(MSVC)
else()
  set(src_dyn_executable_hotgym dynamic_hotgym)
  add_executable(${src_dyn_executable_hotgym} examples/hotgym/Hotgym.cpp examples/hotgym/HelloSPTP.hpp examples/hotgym/HelloSPTP.cpp)
  # for Linux,OSx, link with the shared library
  target_link_libraries(${src_dyn_executable_hotgym} 
    ${INTERNAL_LINKER_FLAGS}
    ${src_lib_shared}
    ${COMMON_OS_LIBS}
  )
  target_compile_options( ${src_dyn_executable_hotgym} PUBLIC ${INTERNAL_CXX_FLAGS})
  target_compile_definitions(${src_dyn_executable_hotgym} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
  target_include_directories(${src_dyn_executable_hotgym} PRIVATE 
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		)
endif()
#########################################################
## MNIST Spatial Pooler Example
#
set(src_executable_mnistsp mnist_sp)
add_executable(${src_executable_mnistsp} examples/mnist/MNIST_SP.cpp)
target_link_libraries(${src_executable_mnistsp} 
        ${INTERNAL_LINKER_FLAGS}
        ${core_library}
        ${COMMON_OS_LIBS}
)
  target_compile_options(${src_executable_mnistsp} PUBLIC ${INTERNAL_CXX_FLAGS})
  target_compile_definitions(${src_executable_mnistsp} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
  # Pass MNIST data directory to main.cpp
  target_compile_definitions(${src_executable_mnistsp} PRIVATE MNIST_DATA_LOCATION=${mnist_SOURCE_DIR})
  target_include_directories(${src_executable_mnistsp} PRIVATE 
        ${CORE_LIB_INCLUDES} 
        ${EXTERNAL_INCLUDES}
        )
add_custom_target(mnist
                  COMMAND ${src_executable_mnistsp}
                  DEPENDS ${src_executable_mnistsp}
                  COMMENT "Executing ${src_executable_mnistsp}"
                  VERBATIM)


############ INSTALL ######################################
#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${core_library}
        ${src_executable_hotgym}
        ${src_executable_mnistsp}
	RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

if(MSVC)
else()	
  install(TARGETS
        ${src_dyn_executable_hotgym}
        ${src_lib_shared}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()

# Version.hpp is also used by the htm.bindings release/deployment system
install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/htm)

install(DIRECTORY ${PROJECT_SOURCE_DIR} DESTINATION include
	  MESSAGE_NEVER
	  FILES_MATCHING PATTERN "htm/*/*.h*"
	  PATTERN "htm/*/*.hpp.in" EXCLUDE)


install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
	  MESSAGE_NEVER
	  DESTINATION include)

          
#
# `make package` results in
# htm_cpp-${HTM_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "htm_cpp-${HTM_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
