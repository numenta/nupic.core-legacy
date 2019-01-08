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

include(src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES

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

###########
# Identify includes from this directory
set(CORE_LIB_INCLUDES  ${PROJECT_SOURCE_DIR}  # for nupic/xxx/*.h
		       ${PROJECT_BINARY_DIR}  # for Version.hpp
)
set(CORE_LIB_INCLUDES  ${CORE_LIB_INCLUDES} PARENT_SCOPE)


###############
# Create Version.hpp
# Extract current git sha and record in nupic/Version.hpp.in
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")
  

#################
# Setup nupic_core static library, consisting of our own sources and yaml libraries;
#
# this is an intermediate archive that will be merged with external static
# libraries in a subsequent step to produce the nupic_core "combined" static
# library.
#
set(src_lib_static_nupiccore_solo nupiccore_solo)


set(algorithm_files
    nupic/algorithms/Anomaly.cpp
    nupic/algorithms/Anomaly.hpp
    nupic/algorithms/AnomalyLikelihood.cpp
    nupic/algorithms/AnomalyLikelihood.hpp
    nupic/algorithms/ArrayBuffer.hpp
    nupic/algorithms/BacktrackingTMCpp.cpp
    nupic/algorithms/BacktrackingTMCpp.hpp
    nupic/algorithms/BitHistory.cpp
    nupic/algorithms/BitHistory.hpp
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
    nupic/algorithms/Scanning.hpp
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
    nupic/engine/RegionIo.cpp
    nupic/engine/RegionParameters.cpp
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
    nupic/ntypes/ArrayRef.hpp
    nupic/ntypes/BundleIO.hpp
    nupic/ntypes/Collection.hpp
    nupic/ntypes/Dimensions.cpp
    nupic/ntypes/Dimensions.hpp
    nupic/ntypes/MemParser.cpp
    nupic/ntypes/MemParser.hpp
    nupic/ntypes/MemStream.hpp
    nupic/ntypes/NodeSet.hpp
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
    nupic/regions/TestNode.cpp
    nupic/regions/TestNode.hpp
    nupic/regions/ScalarSensor.cpp
    nupic/regions/ScalarSensor.hpp
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
    nupic/types/ptr_types.hpp
    nupic/types/Serializable.hpp
    nupic/types/Types.hpp
    nupic/types/Serializable.hpp
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
    nupic/utils/Watcher.cpp
    nupic/utils/Watcher.hpp
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

########################################################
# Lib
add_library(${src_lib_static_nupiccore_solo} STATIC 
    ${algorithm_files} 
    ${encoders_files} 
    ${engine_files} 
    ${math_files} 
    ${ntypes_files} 
    ${os_files} 
    ${regions_files} 
    ${types_files} 
    ${utils_files} 
)


target_compile_options( ${src_lib_static_nupiccore_solo} PUBLIC ${INTERNAL_CXX_FLAGS})
target_compile_definitions(${src_lib_static_nupiccore_solo} PRIVATE ${COMMON_COMPILER_DEFINITIONS})
target_include_directories(${src_lib_static_nupiccore_solo} PRIVATE 
		${CORE_LIB_INCLUDES} 
		${EXTERNAL_INCLUDES}
		)

if(${NUPIC_IWYU})
  set_target_properties(${src_lib_static_nupiccore_solo}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()


#
# Create the nupic_core as a "combined" static library by merging nupic_core_solo
# with our external static libraries
#
set(src_combined_nupiccore_source_archives
    ${src_lib_static_nupiccore_solo}
    ${yaml-cpp_LIBRARIES}
    ${Boost_LIBRARIES}
    )
merge_static_libraries(${core_library} "${src_combined_nupiccore_source_archives}")


#########################################################
# Test
# The tests were placed in a separate file to avoid clutering up this file.
add_subdirectory(test)


#########################################################
## Example custom executable
#
## Setup benchmark_hotgym
#
set(src_executable_hotgym benchmark_hotgym)
add_executable(${src_executable_hotgym} examples/Hotgym.cpp)
target_link_libraries(${src_executable_hotgym} 
		${core_library}
		${COMMON_OS_LIBS}
		)
target_compile_options( ${src_executable_hotgym} PUBLIC ${INTERNAL_CXX_FLAGS}) #TODO how to use OPTIMIZED CXX flags? 
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


##################################################
#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${core_library}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Version.hpp is also used by the nupic.bindings release/deployment system
install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic)

# install Data.csv for a test
install(FILES "${PROJECT_SOURCE_DIR}/test/unit/engine/Data.csv"
        DESTINATION "data/")

install(DIRECTORY ${PROJECT_SOURCE_DIR} DESTINATION include
	  MESSAGE_NEVER
	  FILES_MATCHING PATTERN "nupic/*/*.h*"
	  PATTERN "nupic/*/*.hpp.in" EXCLUDE)

install(DIRECTORY ${REPOSITORY_DIR}/external/common/include/gtest
	  MESSAGE_NEVER
	  DESTINATION include/gtest
	  FILES_MATCHING PATTERN "*.h*")

install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
	  MESSAGE_NEVER
	  DESTINATION include)

install(TARGETS
        ${src_executable_hotgym}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

          
#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
