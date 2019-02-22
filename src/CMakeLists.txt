# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have purchased from
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

cmake_minimum_required(VERSION 3.7)
project(nupic_core CXX)

message(STATUS "Configuring nupic_core src")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE ON) # toggle for cmake debug
include(src/NupicLibraryUtils) # for combining static libs
include(src/ExpandStaticLib)   # for creating objects from a static lib

#
# NuPIC CMake options
#
option(NUPIC_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program, with a minimum CMake version of 3.3.")
if(${NUPIC_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()

#--------------------------------------------------------
# Identify includes from this directory
set(CORE_LIB_INCLUDES  ${PROJECT_SOURCE_DIR}  # for nupic/xxx/*.h
		       ${PROJECT_BINARY_DIR}  # for Version.hpp
)
set(CORE_LIB_INCLUDES  ${CORE_LIB_INCLUDES} PARENT_SCOPE)


#--------------------------------------------------------
# Create Version.hpp
# Extract current git sha and record in nupic/Version.hpp.in
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")
  

############ nupic_core Library ######################################
# Setup nupic_core static library, consisting of our own sources and yaml libraries;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the nupic_core "combined" static
# library.
#
set(src_objlib LibrarySource)


set(algorithm_files
    nupic/algorithms/Anomaly.cpp
    nupic/algorithms/Anomaly.hpp
    nupic/algorithms/AnomalyLikelihood.cpp
    nupic/algorithms/AnomalyLikelihood.hpp
    nupic/algorithms/ArrayBuffer.hpp
    nupic/algorithms/BacktrackingTMCpp.cpp
    nupic/algorithms/BacktrackingTMCpp.hpp
    nupic/algorithms/Cell.cpp
    nupic/algorithms/Cell.hpp
    nupic/algorithms/Cells4.cpp
    nupic/algorithms/Cells4.hpp
    nupic/algorithms/ClassifierResult.cpp
    nupic/algorithms/ClassifierResult.hpp
    nupic/algorithms/Connections.cpp
    nupic/algorithms/Connections.hpp
    nupic/algorithms/InSynapse.cpp
    nupic/algorithms/InSynapse.hpp
    nupic/algorithms/OutSynapse.cpp
    nupic/algorithms/OutSynapse.hpp
    nupic/algorithms/SDRClassifier.cpp
    nupic/algorithms/SDRClassifier.hpp
    nupic/algorithms/Segment.cpp
    nupic/algorithms/Segment.hpp
    nupic/algorithms/SegmentUpdate.cpp
    nupic/algorithms/SegmentUpdate.hpp
    nupic/algorithms/SpatialPooler.cpp
    nupic/algorithms/SpatialPooler.hpp
    nupic/algorithms/TemporalMemory.cpp
    nupic/algorithms/TemporalMemory.hpp
)


set(encoders_files 
    nupic/encoders/ScalarEncoder.cpp
    nupic/encoders/ScalarEncoder.hpp
)
    
set(engine_files
    nupic/engine/Input.cpp
    nupic/engine/Input.hpp
    nupic/engine/Link.cpp
    nupic/engine/Link.hpp
    nupic/engine/LinkPolicy.hpp
    nupic/engine/LinkPolicyFactory.cpp
    nupic/engine/LinkPolicyFactory.hpp
    nupic/engine/Network.cpp
    nupic/engine/Network.hpp
    nupic/engine/NuPIC.cpp
    nupic/engine/NuPIC.hpp
    nupic/engine/Output.cpp
    nupic/engine/Output.hpp
    nupic/engine/Region.cpp
    nupic/engine/Region.hpp
    nupic/engine/RegionImpl.cpp
    nupic/engine/RegionImpl.hpp
    nupic/engine/RegionImplFactory.cpp
    nupic/engine/RegionImplFactory.hpp
    nupic/engine/RegisteredRegionImpl.hpp
    nupic/engine/RegisteredRegionImplCpp.hpp
    nupic/engine/Spec.cpp
    nupic/engine/Spec.hpp
    nupic/engine/TestFanIn2LinkPolicy.cpp
    nupic/engine/TestFanIn2LinkPolicy.hpp
    nupic/engine/UniformLinkPolicy.cpp
    nupic/engine/UniformLinkPolicy.hpp
    nupic/engine/YAMLUtils.cpp
    nupic/engine/YAMLUtils.hpp
)
  
set(math_files
    nupic/math/ArrayAlgo.hpp
    nupic/math/DenseMatrix.hpp
    nupic/math/Domain.hpp
    nupic/math/Functions.hpp
    nupic/math/Index.hpp
    nupic/math/Math.hpp
    nupic/math/Set.hpp
    nupic/math/SparseBinaryMatrix.hpp
    nupic/math/SparseMatrix.hpp
    nupic/math/SparseTensor.hpp
    nupic/math/StlIo.cpp
    nupic/math/StlIo.hpp
    nupic/math/Topology.cpp
    nupic/math/Topology.hpp
    nupic/math/Types.hpp
    nupic/math/Utils.hpp
)


set(ntypes_files
    nupic/ntypes/Array.hpp
    nupic/ntypes/ArrayBase.cpp
    nupic/ntypes/ArrayBase.hpp
    nupic/ntypes/BundleIO.hpp
    nupic/ntypes/Collection.hpp
    nupic/ntypes/Dimensions.cpp
    nupic/ntypes/Dimensions.hpp
    nupic/ntypes/MemParser.cpp
    nupic/ntypes/MemParser.hpp
    nupic/ntypes/MemStream.hpp
    nupic/ntypes/Scalar.cpp
    nupic/ntypes/Scalar.hpp
    nupic/ntypes/Value.cpp
    nupic/ntypes/Value.hpp
)
  
set(os_files
    nupic/os/Directory.cpp
    nupic/os/Directory.hpp
    nupic/os/Env.cpp
    nupic/os/Env.hpp
    nupic/os/ImportFilesystem.hpp
    nupic/os/OS.cpp
    nupic/os/OS.hpp
    nupic/os/OSUnix.cpp
    nupic/os/OSWin.cpp
    nupic/os/Path.cpp
    nupic/os/Path.hpp
    nupic/os/Timer.cpp
    nupic/os/Timer.hpp    
)

set(regions_files
    nupic/regions/ScalarSensor.cpp
    nupic/regions/ScalarSensor.hpp
    nupic/regions/SPRegion.cpp
    nupic/regions/SPRegion.hpp
    nupic/regions/TestNode.cpp
    nupic/regions/TestNode.hpp
    nupic/regions/VectorFile.cpp
    nupic/regions/VectorFile.hpp
    nupic/regions/VectorFileEffector.cpp
    nupic/regions/VectorFileEffector.hpp
    nupic/regions/VectorFileSensor.cpp
    nupic/regions/VectorFileSensor.hpp   
)

set(types_files
    nupic/types/BasicType.cpp
    nupic/types/BasicType.hpp
    nupic/types/Exception.hpp
    nupic/types/Fraction.cpp
    nupic/types/Fraction.hpp
    nupic/types/Serializable.hpp
    nupic/types/Types.hpp
    nupic/types/Serializable.hpp
    nupic/types/Sdr.hpp
    nupic/types/Sdr.cpp
    nupic/types/SdrProxy.hpp
)

set(utils_files
    nupic/utils/GroupBy.hpp
    nupic/utils/Log.hpp
    nupic/utils/LoggingException.cpp
    nupic/utils/LoggingException.hpp
    nupic/utils/LogItem.cpp
    nupic/utils/LogItem.hpp
    nupic/utils/MovingAverage.cpp
    nupic/utils/MovingAverage.hpp
    nupic/utils/Random.cpp
    nupic/utils/Random.hpp
    nupic/utils/SlidingWindow.hpp
    nupic/utils/StringUtils.cpp
    nupic/utils/StringUtils.hpp
    nupic/utils/VectorHelpers.hpp
    nupic/utils/SdrMetrics.hpp
    nupic/utils/Watcher.cpp
    nupic/utils/Watcher.hpp
)

set(examples_files
    examples/hotgym/Hotgym.cpp
    examples/hotgym/HelloSPTP.cpp
    examples/hotgym/HelloSPTP.hpp
    examples/mnist/MNIST_SP.cpp
)

#set up file tabs in Visual Studio
source_group("algorithms" FILES ${algorithm_files})
source_group("encoders" FILES ${encoders_files})
source_group("engine" FILES ${engine_files})
source_group("math" FILES ${math_files})
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
    ${math_files} 
    ${ntypes_files} 
    ${os_files} 
    ${regions_files} 
    ${types_files} 
    ${utils_files}
    ${examples_files} 
)
# shared libraries need PIC
target_compile_options( ${src_objlib} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${src_objlib} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${src_objlib} PRIVATE 
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		)

if(${NUPIC_IWYU})
  set_target_properties(${src_objlib}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()

############ Building Static LIB #############################################
# build static libnupic_core_solo.a without yaml and boost.
# uses objects compiled for src_objlib

set(src_lib_static nupic_core_solo)
add_library(${src_lib_static} STATIC $<TARGET_OBJECTS:${src_objlib}>)

#
# Create the libnupic_core.a as a "combined" static library by merging nupic_core_static
# with our external static libraries
# NOTE: cannot use target_link_libraries( ) with the add_library for STATIC.
#       so we have to merge them in a different way.
#
set(src_combined_nupiccore_source_archives
    ${src_lib_static}
    ${yaml-cpp_LIBRARIES}
    ${Boost_LIBRARIES}
    )
merge_static_libraries(${core_library} "${src_combined_nupiccore_source_archives}")

############ Building Shared LIB #############################################
# builds libnupic_core.so for Linux,OSx and 
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
add_executable(${src_executable_hotgym} examples/hotgym/Hotgym.cpp examples/hotgym/HelloSPTP.hpp)
if(MSVC)
  # for Windows, link with the static library
  target_link_libraries(${src_executable_hotgym} 
    ${core_library}
    ${COMMON_OS_LIBS}
		)
else()
  # for Linux,OSx, link with the shared library
  target_link_libraries(${src_executable_hotgym} 
    ${src_lib_shared}
    ${COMMON_OS_LIBS}
		)
endif()
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
## MNIST Spatial Pooler Example
#
set(src_executable_mnistsp mnist_sp)
add_executable(${src_executable_mnistsp} examples/mnist/MNIST_SP.cpp)
target_link_libraries(${src_executable_mnistsp} ${src_exacutable_mnistsp}
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
        ${src_lib_shared}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
endif()

# Version.hpp is also used by the nupic.bindings release/deployment system
install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic)

install(DIRECTORY ${PROJECT_SOURCE_DIR} DESTINATION include
	  MESSAGE_NEVER
	  FILES_MATCHING PATTERN "nupic/*/*.h*"
	  PATTERN "nupic/*/*.hpp.in" EXCLUDE)


install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
	  MESSAGE_NEVER
	  DESTINATION include)

          
#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
