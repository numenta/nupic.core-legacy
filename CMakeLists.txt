# Minimal CMake configuration for nupic.base
#

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(project_name nupic.base)

project(${project_name} LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(_include_directories ${CMAKE_SOURCE_DIR}/src)

# Common compile definitions
set(_compile_definitions)
if (MSVC)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-DVC_EXTRALEAN)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
    add_definitions(-DNOMINMAX)
    add_definitions(-DNOGDI)

    add_definitions(-DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE)

    add_definitions(-DNTA_OS_WINDOWS)
    add_definitions(-DNTA_COMPILER_MSVC)
endif()

# Boost dependency
set(boost_components filesystem)
find_package(Boost REQUIRED COMPONENTS ${boost_components})
include_directories(${Boost_INCLUDE_DIRS})

# yaml-cpps dependency
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# gtest
enable_testing()
find_package(GTest REQUIRED)


###################
# nupic static lib
###################
message(STATUS "Configuring library")

set(lib_name ${project_name})

set(nupic_src ${CMAKE_SOURCE_DIR}/src/nupic)

set(algorithm_files
    ${nupic_src}/algorithms/Anomaly.cpp
    ${nupic_src}/algorithms/Anomaly.hpp
    ${nupic_src}/algorithms/ArrayBuffer.hpp
    ${nupic_src}/algorithms/BitHistory.cpp
    ${nupic_src}/algorithms/BitHistory.hpp
    ${nupic_src}/algorithms/Cell.cpp
    ${nupic_src}/algorithms/Cell.hpp
    ${nupic_src}/algorithms/Cells4.cpp
    ${nupic_src}/algorithms/Cells4.hpp
    ${nupic_src}/algorithms/ClassifierResult.cpp
    ${nupic_src}/algorithms/ClassifierResult.hpp
    ${nupic_src}/algorithms/CondProbTable.cpp
    ${nupic_src}/algorithms/CondProbTable.hpp
    ${nupic_src}/algorithms/Connections.cpp
    ${nupic_src}/algorithms/Connections.hpp
    ${nupic_src}/algorithms/GaborNode.cpp
    ${nupic_src}/algorithms/GaborNode.hpp
    ${nupic_src}/algorithms/ImageSensorLite.cpp
    ${nupic_src}/algorithms/ImageSensorLite.hpp
    ${nupic_src}/algorithms/InSynapse.cpp
    ${nupic_src}/algorithms/InSynapse.hpp
    ${nupic_src}/algorithms/OutSynapse.cpp
    ${nupic_src}/algorithms/OutSynapse.hpp
    ${nupic_src}/algorithms/Scanning.hpp
    ${nupic_src}/algorithms/SDRClassifier.cpp
    ${nupic_src}/algorithms/SDRClassifier.hpp
    ${nupic_src}/algorithms/Segment.cpp
    ${nupic_src}/algorithms/Segment.hpp
    ${nupic_src}/algorithms/SegmentUpdate.cpp
    ${nupic_src}/algorithms/SegmentUpdate.hpp
    ${nupic_src}/algorithms/SpatialPooler.cpp
    ${nupic_src}/algorithms/SpatialPooler.hpp
    ${nupic_src}/algorithms/Svm.cpp
    ${nupic_src}/algorithms/Svm.hpp
    ${nupic_src}/algorithms/SvmT.hpp
    ${nupic_src}/algorithms/TemporalMemory.cpp
    ${nupic_src}/algorithms/TemporalMemory.hpp
)

set(encoders_files 
    ${nupic_src}/encoders/ScalarEncoder.cpp
    ${nupic_src}/encoders/ScalarEncoder.hpp
    ${nupic_src}/encoders/ScalarSensor.cpp
    ${nupic_src}/encoders/ScalarSensor.hpp
)

set(engine_files
    ${nupic_src}/engine/Collections.cpp
    ${nupic_src}/engine/Input.cpp
    ${nupic_src}/engine/Input.hpp
    ${nupic_src}/engine/Link.cpp
    ${nupic_src}/engine/Link.hpp
    ${nupic_src}/engine/LinkPolicy.hpp
    ${nupic_src}/engine/LinkPolicyFactory.cpp
    ${nupic_src}/engine/LinkPolicyFactory.hpp
    ${nupic_src}/engine/Network.cpp
    ${nupic_src}/engine/Network.hpp
    ${nupic_src}/engine/NuPIC.cpp
    ${nupic_src}/engine/NuPIC.hpp
    ${nupic_src}/engine/Output.cpp
    ${nupic_src}/engine/Output.hpp
    ${nupic_src}/engine/PyBindRegion.cpp
    ${nupic_src}/engine/PyBindRegion.hpp
    ${nupic_src}/engine/Region.cpp
    ${nupic_src}/engine/Region.hpp
    ${nupic_src}/engine/RegionImpl.cpp
    ${nupic_src}/engine/RegionImpl.hpp
    ${nupic_src}/engine/RegionImplFactory.cpp
    ${nupic_src}/engine/RegionImplFactory.hpp
    ${nupic_src}/engine/RegionIo.cpp
    ${nupic_src}/engine/RegionParameters.cpp
    ${nupic_src}/engine/RegisteredRegionImpl.hpp
    ${nupic_src}/engine/Spec.cpp
    ${nupic_src}/engine/Spec.hpp
    ${nupic_src}/engine/TestFanIn2LinkPolicy.cpp
    ${nupic_src}/engine/TestFanIn2LinkPolicy.hpp
    ${nupic_src}/engine/TestNode.cpp
    ${nupic_src}/engine/TestNode.hpp
    ${nupic_src}/engine/UniformLinkPolicy.cpp
    ${nupic_src}/engine/UniformLinkPolicy.hpp
    ${nupic_src}/engine/YAMLUtils.cpp
    ${nupic_src}/engine/YAMLUtils.hpp
)

set(math_files
    ${nupic_src}/math/Array2D.hpp
    ${nupic_src}/math/ArrayAlgo.hpp
    ${nupic_src}/math/Convolution.hpp
    ${nupic_src}/math/DenseMatrix.hpp
    ${nupic_src}/math/Domain.hpp
    ${nupic_src}/math/Erosion.hpp
    ${nupic_src}/math/Functions.hpp
    ${nupic_src}/math/GraphAlgorithms.hpp
    ${nupic_src}/math/Index.hpp
    ${nupic_src}/math/Math.hpp
    ${nupic_src}/math/NearestNeighbor.hpp
    ${nupic_src}/math/Rotation.hpp
    ${nupic_src}/math/SegmentMatrixAdapter.hpp
    ${nupic_src}/math/Set.hpp
    ${nupic_src}/math/SparseBinaryMatrix.hpp
    ${nupic_src}/math/SparseMatrix.hpp
    ${nupic_src}/math/SparseMatrix01.hpp
    ${nupic_src}/math/SparseMatrixAlgorithms.cpp
    ${nupic_src}/math/SparseMatrixAlgorithms.hpp
    ${nupic_src}/math/SparseMatrixConnections.cpp
    ${nupic_src}/math/SparseMatrixConnections.hpp
    ${nupic_src}/math/SparseRLEMatrix.hpp
    ${nupic_src}/math/SparseTensor.hpp
    ${nupic_src}/math/StlIo.cpp
    ${nupic_src}/math/StlIo.hpp
    ${nupic_src}/math/Topology.cpp
    ${nupic_src}/math/Topology.hpp
    ${nupic_src}/math/Types.hpp
    ${nupic_src}/math/Utils.hpp
)

set(ntypes_files
    ${nupic_src}/ntypes/Array.hpp
    ${nupic_src}/ntypes/ArrayBase.cpp
    ${nupic_src}/ntypes/ArrayBase.hpp
    ${nupic_src}/ntypes/ArrayRef.hpp
    ${nupic_src}/ntypes/Buffer.cpp
    ${nupic_src}/ntypes/Buffer.hpp
    ${nupic_src}/ntypes/BundleIO.cpp
    ${nupic_src}/ntypes/BundleIO.hpp
    ${nupic_src}/ntypes/Collection.cpp
    ${nupic_src}/ntypes/Collection.hpp
    ${nupic_src}/ntypes/Dimensions.cpp
    ${nupic_src}/ntypes/Dimensions.hpp
    ${nupic_src}/ntypes/MemParser.cpp
    ${nupic_src}/ntypes/MemParser.hpp
    ${nupic_src}/ntypes/MemStream.hpp
    ${nupic_src}/ntypes/NodeSet.hpp
    ${nupic_src}/ntypes/ObjectModel.h
    ${nupic_src}/ntypes/ObjectModel.hpp
    ${nupic_src}/ntypes/Scalar.cpp
    ${nupic_src}/ntypes/Scalar.hpp
    ${nupic_src}/ntypes/Value.cpp
    ${nupic_src}/ntypes/Value.hpp
)

set(os_files
    ${nupic_src}/os/Directory.cpp
    ${nupic_src}/os/Directory.hpp
    ${nupic_src}/os/DynamicLibrary.cpp
    ${nupic_src}/os/DynamicLibrary.hpp
    ${nupic_src}/os/Env.cpp
    ${nupic_src}/os/Env.hpp
    ${nupic_src}/os/FStream.cpp
    ${nupic_src}/os/FStream.hpp
    ${nupic_src}/os/OS.cpp
    ${nupic_src}/os/OS.hpp
    ${nupic_src}/os/OSUnix.cpp
    ${nupic_src}/os/OSWin.cpp
    ${nupic_src}/os/Path.cpp
    ${nupic_src}/os/Path.hpp
    ${nupic_src}/os/Regex.cpp
    ${nupic_src}/os/Regex.hpp
    ${nupic_src}/os/Timer.cpp
    ${nupic_src}/os/Timer.hpp    
)

set(regions_files
    ${nupic_src}/regions/VectorFile.cpp
    ${nupic_src}/regions/VectorFile.hpp
    ${nupic_src}/regions/VectorFileEffector.cpp
    ${nupic_src}/regions/VectorFileEffector.hpp
    ${nupic_src}/regions/VectorFileSensor.cpp
    ${nupic_src}/regions/VectorFileSensor.hpp    
)

set(types_files
    ${nupic_src}/types/BasicType.cpp
    ${nupic_src}/types/BasicType.hpp
    ${nupic_src}/types/Exception.hpp
    ${nupic_src}/types/Fraction.cpp
    ${nupic_src}/types/Fraction.hpp
    ${nupic_src}/types/ptr_types.hpp
    ${nupic_src}/types/Types.h
    ${nupic_src}/types/Types.hpp
)

set(utils_files
    ${nupic_src}/utils/GroupBy.hpp
    ${nupic_src}/utils/Log.hpp
    ${nupic_src}/utils/LoggingException.cpp
    ${nupic_src}/utils/LoggingException.hpp
    ${nupic_src}/utils/LogItem.cpp
    ${nupic_src}/utils/LogItem.hpp
    ${nupic_src}/utils/MovingAverage.cpp
    ${nupic_src}/utils/MovingAverage.hpp
    ${nupic_src}/utils/Random.cpp
    ${nupic_src}/utils/Random.hpp
    ${nupic_src}/utils/StringUtils.cpp
    ${nupic_src}/utils/StringUtils.hpp
    ${nupic_src}/utils/TRandom.cpp
    ${nupic_src}/utils/TRandom.hpp
    ${nupic_src}/utils/Watcher.cpp
    ${nupic_src}/utils/Watcher.hpp
)

# Lib

add_library(${lib_name} STATIC 
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

target_include_directories(${lib_name} PRIVATE ${_include_directories})
target_compile_definitions(${lib_name} PRIVATE ${_compile_definitions})

source_group("algorithm" FILES ${algorithm_files})
source_group("encoders" FILES ${encoders_files})
source_group("engine" FILES ${engine_files})
source_group("math" FILES ${math_files})
source_group("ntypes" FILES ${ntypes_files})
source_group("os" FILES ${os_files})
source_group("regions" FILES ${regions_files})
source_group("types" FILES ${types_files})
source_group("utils" FILES ${utils_files})


# Test


# target_link_libraries(main PRIVATE yaml-cpp)

# target_link_libraries(main PRIVATE GTest::GTest GTest::Main)
# add_test(AllTestsInMain main)