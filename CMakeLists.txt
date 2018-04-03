# Minimal CMake configuration for nupic.base
#

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(project_name nupic.base)

project(${project_name} LANGUAGES CXX)

set(_include_directories ${CMAKE_SOURCE_DIR}/src)

# Common compile definitions
set(_compile_definitions)
if (MSVC)
  list(APPEND _compile_definitions "VC_EXTRALEAN;WIN32_LEAN_AND_MEAN;NOMINMAX;NOGDI")
endif()

message(STATUS "A message from CMakeLists.txt")

###################
# nupic static lib
###################
message(STATUS "Configuring library")

set(lib_name ${project_name})

set(algorithm_header_files)
list(APPEND algorithm_header_files
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Anomaly.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/ArrayBuffer.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/BitHistory.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Cell.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Cells4.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/ClassifierResult.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/CondProbTable.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Connections.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/GaborNode.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/ImageSensorLite.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/InSynapse.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/OutSynapse.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Scanning.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SDRClassifier.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Segment.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SegmentUpdate.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SpatialPooler.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Svm.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SvmT.hpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/TemporalMemory.hpp
)

set(algorithm_source_files)
list(APPEND algorithm_source_files
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Anomaly.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/BitHistory.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Cell.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Cells4.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/ClassifierResult.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/CondProbTable.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Connections.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/GaborNode.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/ImageSensorLite.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/InSynapse.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/OutSynapse.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SDRClassifier.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Segment.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SegmentUpdate.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/SpatialPooler.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/Svm.cpp
    ${CMAKE_SOURCE_DIR}/src/nupic/algorithms/TemporalMemory.cpp
    )

add_library(${lib_name} STATIC 
    ${algorithm_header_files} 
    ${algorithm_source_files})

target_include_directories(${lib_name} PRIVATE ${_include_directories})
target_compile_definitions(${lib_name} PRIVATE ${_compile_definitions})

source_group("algorithm" FILES ${algorithm_header_files} ${algorithm_source_files})
