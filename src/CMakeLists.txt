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

cmake_minimum_required(VERSION 2.8)
project(nupic_core CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug


if( POLICY CMP0046 )
  cmake_policy(VERSION 2.8)
  # The new policy for add_dependencies is to now error if a dependency target
  # is not found. The old policy didn't care and continued through configuration
  cmake_policy(SET CMP0046 OLD)
endif()

if( POLICY CMP0054 )
  # The OLD behavior for this policy is to dereference variables and interpret
  # keywords even if they are quoted or bracketed. The NEW behavior is to not
  # dereference variables or interpret keywords that have been quoted or bracketed.
  cmake_policy(SET CMP0054 OLD)
endif()


#
# NuPIC CMake options
#
option(NUPIC_IWYU "Enable include-what-you-use
  (http://include-what-you-use.org/). This requires the iwyu binary to be
  discoverable by CMake's find_program.")
if(${NUPIC_IWYU})
  find_program(iwyu_path NAMES include-what-you-use iwyu)
  if(NOT iwyu_path)
    message(FATAL_ERROR "Could not find the program include-what-you-use")
  endif()
endif()


#
# Set up compile flags for internal sources and for swig-generated sources
#

set(src_compile_flags "${INTERNAL_CXX_FLAGS_OPTIMIZED}")
set(swig_generated_file_compile_flags "${EXTERNAL_CXX_FLAGS_OPTIMIZED}")

if(MINGW)
  # This is for GCC 4.8.x
  # http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
  set(src_compile_flags "${src_compile_flags} -include cmath")
  set(swig_generated_file_compile_flags "${swig_generated_file_compile_flags} -include cmath")
endif()


#
# Compiler definitions specific to nupic.core code
#

string(TOUPPER ${PLATFORM} platform_uppercase)

set(src_compiler_definitions
    ${COMMON_COMPILER_DEFINITIONS}
    ${CAPNP_COMPILER_DEFINITIONS}
    -DNTA_OS_${platform_uppercase}
    -DNTA_ARCH_${BITNESS}
    -DHAVE_CONFIG_H
    -DNTA_INTERNAL
    -DBOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    -DBOOST_NO_WREGEX
    -DNUPIC2
    -DAPR_DECLARE_STATIC
    -DAPU_DECLARE_STATIC)

if(NOT "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASM)
endif()

if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_ASSERTIONS_ON)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_CLANG)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_MSVC)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MinGW")
  set(src_compiler_definitions
      ${src_compiler_definitions}
      -DNTA_COMPILER_GNU
      -D_hypot=hypot)
endif()


add_definitions(${src_compiler_definitions})


#
# Find Python Interp, min. v2.7
# And NumPy core directory
#
find_package(PythonInterp 2.7 REQUIRED)
message(STATUS "CMAKE Found python interpreter ${PYTHON_EXECUTABLE} version=${PYTHON_VERSION_STRING}")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                        "import numpy; import os; import sys; sys.stdout.write(os.path.dirname(numpy.get_include()))"
                         OUTPUT_VARIABLE src_numpy_core)

# Quick way to fixup directory paths to NumPy
get_filename_component(src_numpy_core ${src_numpy_core}/include/.. ABSOLUTE)

#
# Setup include paths
#
if("${PROJECT_BUILD_ARTIFACTS_DIR}" STREQUAL "")
  set(PROJECT_BUILD_ARTIFACTS_DIR "${PROJECT_BINARY_DIR}/artifacts")
endif()

include_directories(SYSTEM
                    ${src_numpy_core}/include
                    ${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR})

#
# Let CMake know where all of the external files are.
#
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${REPOSITORY_DIR}/external/common/
                      ${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/
                      ${src_numpy_core}/)
#
# Make sure the Cap'n Proto generation scripts are on the PATH.
#
set(ENV{PATH} "$ENV{PATH}:${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/bin")

#
# Extract current git sha and record in nupic/Version.hpp
#
execute_process(COMMAND git rev-parse HEAD
                OUTPUT_VARIABLE NUPIC_CORE_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
configure_file(
  "${PROJECT_SOURCE_DIR}/nupic/Version.hpp.in"
  "${PROJECT_BINARY_DIR}/Version.hpp")

#
# Setup gtest
#
set(src_lib_static_gtest gtest)
add_library(${src_lib_static_gtest}
            STATIC
            ${REPOSITORY_DIR}/external/common/src/gtest/gtest-all.cpp)
set_target_properties(${src_lib_static_gtest}
                      PROPERTIES COMPILE_FLAGS "${src_compile_flags}"
                      LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Setup for python
#
find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

message(STATUS "PYTHON_EXECUTABLE   = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_LIBRARIES    = ${PYTHON_LIBRARIES}")

# List all .capnp files here. The C++ files will be generated and included
# when compiling later on.
set(src_capnp_specs_rel
  nupic/proto/BitHistory.capnp
  nupic/proto/ClaClassifier.capnp
  nupic/proto/ConnectionsProto.capnp
  nupic/proto/ExtendedTemporalMemoryProto.capnp
  nupic/proto/LinkProto.capnp
  nupic/proto/Map.capnp
  nupic/proto/NetworkProto.capnp
  nupic/proto/PyRegionProto.capnp
  nupic/proto/RandomProto.capnp
  nupic/proto/RegionProto.capnp
  nupic/proto/SparseBinaryMatrixProto.capnp
  nupic/proto/SparseMatrixProto.capnp
  nupic/proto/SpatialPoolerProto.capnp
  nupic/proto/SdrClassifier.capnp
  nupic/proto/TemporalMemoryProto.capnp
  nupic/proto/TestNodeProto.capnp
  nupic/proto/VectorFileSensorProto.capnp
)

# TODO it seems that building of this information (src_capnp_specs, etc.) should
# be abstracted by the implementation of function CREATE_CAPNPC_COMMAND in
# external/CapnpProto.cmake, which would use an output variable to return the
# contents of src_capnp_srcs for linking.
foreach(spec ${src_capnp_specs_rel})
  list(APPEND src_capnp_specs ${PROJECT_SOURCE_DIR}/${spec})
  list(APPEND src_capnp_hdrs ${PROJECT_BINARY_DIR}/${spec}.h)
  list(APPEND src_capnp_srcs ${PROJECT_BINARY_DIR}/${spec}.c++)
endforeach(spec)
set(src_capnp_all_spec_hdrs_and_srcs "${src_capnp_hdrs};${src_capnp_srcs}")

# Create custom target for generating C++ code from .capnp schema files.
create_capnpc_command(capnp_specs
                      "${src_capnp_specs}"
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_BINARY_DIR}
                      "${src_capnp_all_spec_hdrs_and_srcs}")

#
# Python support
#
set(src_py_support_files
    nupic/py_support/NumpyVector.cpp
    nupic/py_support/PyArray.cpp
    nupic/py_support/PyHelpers.cpp
    nupic/py_support/PythonStream.cpp
    nupic/py_support/PyCapnp.cpp
    nupic/bindings/PySparseTensor.cpp
)
if(NOT "${PLATFORM}" STREQUAL "windows")
  list(APPEND src_py_support_files nupic/py_support/CapnpToPycapnp.cpp)
endif()

message(STATUS "CAPNP_EXECUTABLE          = ${CAPNP_EXECUTABLE}")
message(STATUS "CAPNP_CMAKE_DEFINITIONS   = ${CAPNP_CMAKE_DEFINITIONS}")
message(STATUS "CAPNP_INCLUDE_DIRS        = ${CAPNP_INCLUDE_DIRS}")
message(STATUS "CAPNP_LINK_LIBRARIES      = ${CAPNP_LINK_LIBRARIES}")

#
# Add the pycapnp Python package to includes
if(NOT "${PLATFORM}" STREQUAL "windows")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                          "import capnp; import os; import sys; sys.stdout.write(os.path.dirname(os.path.dirname(capnp.__file__)))"
                  OUTPUT_VARIABLE src_pycapnp_include_dir)
  message(STATUS "src_pycapnp_include_dir      = ${src_pycapnp_include_dir}")
  include_directories(SYSTEM ${src_pycapnp_include_dir})
endif()

#
# Setup nupic_core_solo static library
#
set(src_lib_static_nupiccore_solo nupic_core_solo)

if("${PLATFORM}" STREQUAL "linux")
  list(APPEND src_common_os_libs pthread dl)
elseif("${PLATFORM}" STREQUAL "darwin")
  list(APPEND src_common_os_libs c++abi)
elseif(MSYS OR MINGW)
  list(APPEND src_common_os_libs psapi ws2_32 wsock32 rpcrt4)
elseif("${PLATFORM}" STREQUAL "windows")
  list(APPEND src_common_os_libs oldnames.lib psapi.lib ws2_32.lib)
endif()

set(src_nupiccore_srcs
    nupic/algorithms/Anomaly.cpp
    nupic/algorithms/BitHistory.cpp
    nupic/algorithms/Cell.cpp
    nupic/algorithms/Cells4.cpp
    nupic/algorithms/ClassifierResult.cpp
    nupic/algorithms/CondProbTable.cpp
    nupic/algorithms/Connections.cpp
    nupic/algorithms/FastClaClassifier.cpp
    nupic/algorithms/GaborNode.cpp
    nupic/algorithms/ImageSensorLite.cpp
    nupic/algorithms/InSynapse.cpp
    nupic/algorithms/OutSynapse.cpp
    nupic/algorithms/Segment.cpp
    nupic/algorithms/SegmentUpdate.cpp
    nupic/algorithms/SDRClassifier.cpp
    nupic/algorithms/SpatialPooler.cpp
    nupic/algorithms/TemporalMemory.cpp
    nupic/algorithms/Svm.cpp
    nupic/encoders/ScalarEncoder.cpp
    nupic/encoders/ScalarSensor.cpp
    nupic/engine/Collections.cpp
    nupic/engine/Input.cpp
    nupic/engine/Link.cpp
    nupic/engine/LinkPolicyFactory.cpp
    nupic/engine/Network.cpp
    nupic/engine/NetworkFactory.cpp
    nupic/engine/NuPIC.cpp
    nupic/engine/Output.cpp
    nupic/engine/Region.cpp
    nupic/engine/RegionImpl.cpp
    nupic/engine/RegionImplFactory.cpp
    nupic/engine/RegionIo.cpp
    nupic/engine/RegionParameters.cpp
    nupic/engine/Spec.cpp
    nupic/engine/TestFanIn2LinkPolicy.cpp
    nupic/engine/TestNode.cpp
    nupic/engine/UniformLinkPolicy.cpp
    nupic/engine/YAMLUtils.cpp
    nupic/experimental/ExtendedTemporalMemory.cpp
    nupic/math/SparseMatrixAlgorithms.cpp
    nupic/math/StlIo.cpp
    nupic/ntypes/ArrayBase.cpp
    nupic/ntypes/Buffer.cpp
    nupic/ntypes/BundleIO.cpp
    nupic/ntypes/Collection.cpp
    nupic/ntypes/Dimensions.cpp
    nupic/ntypes/MemParser.cpp
    nupic/ntypes/Scalar.cpp
    nupic/ntypes/Value.cpp
    nupic/os/Directory.cpp
    nupic/os/DynamicLibrary.cpp
    nupic/os/Env.cpp
    nupic/os/FStream.cpp
    nupic/os/OS.cpp
    nupic/os/OSUnix.cpp
    nupic/os/OSWin.cpp
    nupic/os/Path.cpp
    nupic/os/Regex.cpp
    nupic/os/Timer.cpp
    nupic/regions/PyRegion.cpp
    nupic/regions/VectorFile.cpp
    nupic/regions/VectorFileEffector.cpp
    nupic/regions/VectorFileSensor.cpp
    nupic/types/BasicType.cpp
    nupic/types/Fraction.cpp
    nupic/utils/LoggingException.cpp
    nupic/utils/LogItem.cpp
    nupic/utils/MovingAverage.cpp
    nupic/utils/Random.cpp
    nupic/utils/StringUtils.cpp
    nupic/utils/TRandom.cpp
    nupic/utils/Watcher.cpp)

set(src_lib_static_nupiccore_srcs
    ${src_capnp_srcs}
    ${src_py_support_files}
    ${src_nupiccore_srcs})

set(src_external_static_libs
    ${LIB_STATIC_YAML_CPP_LOC}
    ${LIB_STATIC_YAML_LOC}
    ${LIB_STATIC_APR1_LOC}
    ${LIB_STATIC_APRUTIL1_LOC}
    ${LIB_STATIC_Z_LOC})

set(src_lib_static_nupiccore_solo_link_libs
    ${src_external_static_libs}
    ${src_common_os_libs})

set(src_lib_static_nupiccore_solo_deps
    Apr1StaticLib
    AprUtil1StaticLib
    ZStaticLib
    YamlStaticLib
    YamlCppStaticLib)

# Add capnproto static libs when building nupic.core static lib without python
# extensions
if (NOT ${NUPIC_BUILD_PYEXT_MODULES})
  list(APPEND src_lib_static_nupiccore_solo_deps CapnProto capnp_specs)
  list(APPEND src_lib_static_nupiccore_solo_link_libs ${CAPNP_LINK_LIBRARIES})
endif() # ${NUPIC_BUILD_PYEXT_MODULES}

set(src_lib_static_nupiccore_compile_flags
    "${src_compile_flags} -I${src_numpy_core}/include")
# TODO Does the static library nupiccore_solo need the flags from
# INTERNAL_LINKER_FLAGS_OPTIMIZED? Static libraries don't get linked, do they?
#set(src_lib_static_nupiccore_link_flags
#    "${INTERNAL_LINKER_FLAGS_OPTIMIZED} -static")

message(STATUS "src_compile_flags = ${src_compile_flags}")
message(STATUS "src_lib_static_nupiccore_compile_flags = ${src_lib_static_nupiccore_compile_flags}")
#message(STATUS "src_lib_static_nupiccore_link_flags = ${src_lib_static_nupiccore_link_flags}")
message(STATUS "src_lib_static_nupiccore_solo_link_libs = ${src_lib_static_nupiccore_solo_link_libs}")

add_library(${src_lib_static_nupiccore_solo} STATIC
            ${src_lib_static_nupiccore_srcs})
add_dependencies(${src_lib_static_nupiccore_solo}
                 src_lib_static_nupiccore_solo_deps)
target_link_libraries(${src_lib_static_nupiccore_solo}
                      ${src_lib_static_nupiccore_solo_link_libs})
set_target_properties(${src_lib_static_nupiccore_solo} PROPERTIES COMPILE_FLAGS
                      ${src_lib_static_nupiccore_compile_flags})
#set_target_properties(${src_lib_static_nupiccore_solo} PROPERTIES LINK_FLAGS
#                      ${src_lib_static_nupiccore_link_flags})
if(${NUPIC_IWYU})
  # TODO: Create a target that doesn't include the generated capnp schema files
  # since we don't want to run iwyu on them and iwyu can only be applied to
  # CMake targets (not source files directly).
  set_target_properties(${src_lib_static_nupiccore_solo}
    PROPERTIES CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()

# Common libs for test executables
set(src_common_test_exe_libs
    ${src_lib_static_nupiccore_solo}
    ${PYTHON_LIBRARIES}
)

# Common dependencies for test executables
set(src_common_test_exe_deps)

# Add capnproto static libs to our C/C++ test apps when building python
# extensions, since capnproto is excluded from src_lib_static_nupiccore_solo to
# avoid conflict with capnproto methods compiled into the pycapnp extension.
if (${NUPIC_BUILD_PYEXT_MODULES})
  list(APPEND src_common_test_exe_deps CapnProto)
  list(APPEND src_common_test_exe_libs ${CAPNP_LINK_LIBRARIES})
endif() # ${NUPIC_BUILD_PYEXT_MODULES}


#
# Setup test_cpp_region
#
set(src_executable_cppregiontest cpp_region_test)
add_executable(${src_executable_cppregiontest} test/integration/CppRegionTest.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_cppregiontest} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_cppregiontest} ${src_common_test_exe_libs})
set_target_properties(${src_executable_cppregiontest} PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_cppregiontest} PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_cpp_region
                  COMMAND ${src_executable_cppregiontest}
                  DEPENDS ${src_executable_cppregiontest}
                  COMMENT "Executing test ${src_executable_cppregiontest}"
                  VERBATIM)

#
# Setup test_py_region
#
set(src_executable_pyregiontest py_region_test)
add_executable(${src_executable_pyregiontest} test/integration/PyRegionTest.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_pyregiontest} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_pyregiontest} ${src_common_test_exe_libs})
set_target_properties(${src_executable_pyregiontest}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_pyregiontest}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_py_region
                  COMMAND ${src_executable_pyregiontest}
                  DEPENDS ${src_executable_pyregiontest}
                  COMMENT "Executing test ${src_executable_pyregiontest}"
                  VERBATIM)

#
# Setup test_connections_performance
#
set(src_executable_connectionsperformancetest connections_performance_test)
add_executable(${src_executable_connectionsperformancetest}
               test/integration/ConnectionsPerformanceTest.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_connectionsperformancetest}
                   ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_connectionsperformancetest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_connectionsperformancetest}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_connectionsperformancetest}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_connections_performance
                  COMMAND ${src_executable_connectionsperformancetest}
                  DEPENDS ${src_executable_connectionsperformancetest}
                  COMMENT "Executing test ${src_executable_connectionsperformancetest}"
                  VERBATIM)

#
# Setup helloregion example
#
set(src_executable_helloregion helloregion)
add_executable(${src_executable_helloregion} examples/regions/HelloRegions.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_helloregion} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_helloregion} ${src_common_test_exe_libs})
set_target_properties(${src_executable_helloregion}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_helloregion}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Setup prototest example
#
set(src_executable_prototest prototest)
add_executable(${src_executable_prototest} examples/prototest.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_prototest} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_prototest} ${src_common_test_exe_libs})
set_target_properties(${src_executable_prototest}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_prototest}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")

#
# Setup HelloSP_TP example
#
set(src_executable_hellosptp hello_sp_tp)
add_executable(${src_executable_hellosptp} examples/algorithms/HelloSP_TP.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_hellosptp} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_hellosptp} ${src_common_test_exe_libs})
set_target_properties(${src_executable_hellosptp}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags})
set_target_properties(${src_executable_hellosptp}
                      PROPERTIES LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")


#
# Setup gtests
#
set(src_executable_gtests unit_tests)
add_executable(${src_executable_gtests}
               test/unit/algorithms/AnomalyTest.cpp
               test/unit/algorithms/CondProbTableTest.cpp
               test/unit/algorithms/ConnectionsTest.cpp
               test/unit/algorithms/FastCLAClassifierTest.cpp
               test/unit/algorithms/NearestNeighborUnitTest.cpp
               test/unit/algorithms/SDRClassifierTest.cpp
               test/unit/algorithms/SegmentTest.cpp
               test/unit/algorithms/SpatialPoolerTest.cpp
               test/unit/algorithms/TemporalMemoryTest.cpp
               test/unit/encoders/ScalarEncoderTest.cpp
               test/unit/engine/InputTest.cpp
               test/unit/engine/NetworkTest.cpp
               test/unit/engine/NetworkFactoryTest.cpp
               test/unit/engine/UniformLinkPolicyTest.cpp
               test/unit/engine/YAMLUtilsTest.cpp
               test/unit/experimental/ExtendedTemporalMemoryTest.cpp
               test/unit/math/DenseTensorUnitTest.cpp
               test/unit/math/DomainUnitTest.cpp
               test/unit/math/IndexUnitTest.cpp
               test/unit/math/MathsTest.cpp
               test/unit/math/SparseBinaryMatrixTest.cpp
               test/unit/math/SparseMatrix01UnitTest.cpp
               test/unit/math/SparseMatrixTest.cpp
               test/unit/math/SparseMatrixUnitTest.cpp
               test/unit/math/SparseTensorUnitTest.cpp
               test/unit/ntypes/ArrayTest.cpp
               test/unit/ntypes/BufferTest.cpp
               test/unit/ntypes/CollectionTest.cpp
               test/unit/ntypes/DimensionsTest.cpp
               test/unit/ntypes/MemParserTest.cpp
               test/unit/ntypes/MemStreamTest.cpp
               test/unit/ntypes/NodeSetTest.cpp
               test/unit/ntypes/ScalarTest.cpp
               test/unit/ntypes/ValueTest.cpp
               test/unit/os/DirectoryTest.cpp
               test/unit/os/EnvTest.cpp
               test/unit/os/OSTest.cpp
               test/unit/os/PathTest.cpp
               test/unit/os/RegexTest.cpp
               test/unit/os/TimerTest.cpp
               test/unit/py_support/PyHelpersTest.cpp
               test/unit/types/BasicTypeTest.cpp
               test/unit/types/ExceptionTest.cpp
               test/unit/types/FractionTest.cpp
               test/unit/UnitTestMain.cpp
               test/unit/utils/GroupByTest.cpp
               test/unit/utils/MovingAverageTest.cpp
               test/unit/utils/RandomTest.cpp
               test/unit/utils/WatcherTest.cpp)
if (src_common_test_exe_deps)
  add_dependencies(${src_executable_gtests} ${src_common_test_exe_deps})
endif()
target_link_libraries(${src_executable_gtests}
                      ${src_lib_static_gtest}
                      ${src_common_test_exe_libs})
set_target_properties(${src_executable_gtests}
                      PROPERTIES COMPILE_FLAGS ${src_compile_flags}
                                 LINK_FLAGS "${INTERNAL_LINKER_FLAGS_OPTIMIZED}")
add_custom_target(tests_unit
                  COMMAND ${src_executable_gtests}
                  DEPENDS ${src_executable_gtests}
                  COMMENT "Executing test ${src_executable_gtests}"
                  VERBATIM)

#
# tests_all just calls other targets
#
# TODO This doesn't seem to have any effect; it's probably because the DEPENDS
# of add_custom_target must be files, not other high-level targets. If really
# need to run these tests during build, then either the individual
# add_custom_target of the individual test runners should be declared with the
# ALL option, or tests_all target whould be declared without DEPENDS, and
# add_dependencies should be used to set it's dependencies on the custom targets
# of the inidividual test runners.
add_custom_target(tests_all
                  DEPENDS tests_cpp_region
                  DEPENDS tests_unit
                  COMMENT "Running all tests"
                  VERBATIM)


if (OFF)
  # Dependencies of the combined nupic.core static lib
  set(src_nupiccore_combined_deps
      ${src_lib_static_nupiccore_solo}
      Apr1StaticLib
      AprUtil1StaticLib
      ZStaticLib
      YamlStaticLib
      YamlCppStaticLib)

  if(UNIX OR MSYS OR MINGW)
    set(src_lib_static_nupiccore_combined "${PROJECT_BINARY_DIR}/libnupic_core.a")
    if(MSYS OR MINGW)
      add_custom_command(OUTPUT ${src_lib_static_nupiccore_combined}
                         COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${src_lib_static_nupiccore_solo}>
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_CPP_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APRUTIL1_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APR1_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_Z_LOC}
                         COMMAND ${CMAKE_AR} rcs ${src_lib_static_nupiccore_combined} *.obj
                         DEPENDS ${src_nupiccore_combined_deps})
    else()
      add_custom_command(OUTPUT ${src_lib_static_nupiccore_combined}
                         COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${src_lib_static_nupiccore_solo}>
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_CPP_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APRUTIL1_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APR1_LOC}
                         COMMAND ${CMAKE_AR} -x ${LIB_STATIC_Z_LOC}
                         COMMAND ${CMAKE_AR} rcs ${src_lib_static_nupiccore_combined} *.o
                         DEPENDS ${src_nupiccore_combined_deps})
    endif()
  else()
    set(src_lib_static_nupiccore_combined "${PROJECT_BINARY_DIR}/nupic_core.lib")
    add_custom_command(OUTPUT ${src_lib_static_nupiccore_combined}
                       COMMAND lib.exe /OUT:${src_lib_static_nupiccore_combined}
                               $<TARGET_FILE:${src_lib_static_nupiccore_solo}>
                               ${LIB_STATIC_YAML_CPP_LOC}
                               ${LIB_STATIC_YAML_LOC}
                               ${LIB_STATIC_APRUTIL1_LOC}
                               ${LIB_STATIC_APR1_LOC}
                               ${LIB_STATIC_Z_LOC}
                       DEPENDS ${src_nupiccore_combined_deps}
                       VERBATIM)
  endif()


  # Wrap nupic.core combined lib output in custom target to avoid conflict from
  # other dependent targets that may execute in parallel (per add_custom_command
  # documentation). NOTE Other targets MUST use LibStaticNupicCoreCombinedTarget
  # as a dependency instead of ${src_lib_static_nupiccore_combined}
  add_custom_target(LibStaticNupicCoreCombinedTarget
                    DEPENDS ${src_lib_static_nupiccore_combined})
else()
  # TODO Not really clear why the "combined" library is being built in the first
  # place, since it doesn't seem to add anything new to "solo"!
#  set(src_lib_static_nupiccore_combined "${src_lib_static_nupiccore_solo}")
#  message(STATUS "ZZZ Aliasing nupic core combined to solo")

#  set(src_lib_static_nupiccore_combined nupic_core)
#  message(STATUS "ZZZ Repackaging nupic core 'solo' static lib as 'combined'")
#  add_library(${src_lib_static_nupiccore_combined} STATIC
#              ${src_py_support_files})
#  target_link_libraries(${src_lib_static_nupiccore_combined}
#                        PRIVATE ${src_lib_static_nupiccore_solo})

  include(src/NupicLibraryUtils) # for MERGE_STATIC_LIBRARIES

  set(src_lib_static_nupiccore_combined nupic_core)
  set(src_combined_nupiccore_source_archives
      ${src_lib_static_nupiccore_solo}
      ${src_external_static_libs})
  merge_static_libraries(${src_lib_static_nupiccore_combined}
                         "${src_combined_nupiccore_source_archives}")
endif()


#
# Use SWIG to generate Python extensions.
#
if (${NUPIC_BUILD_PYEXT_MODULES})
  include(UseSWIG)

  message(STATUS "CMAKE_CXX_COMPILER_ID        = ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "PY_EXTENSIONS_DIR            = ${PY_EXTENSIONS_DIR}")

  # Set the output location for the language modules that are created.
  set(CMAKE_SWIG_OUTDIR ${PROJECT_BINARY_DIR})

  # Make sure the directory exists for the generated C++ files.
  file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/nupic/bindings)

  # TODO ZZZ set COMPILE_FLAGS on swig targets

  # SWIG options from:
  # https://github.com/swig/swig/blob/master/Source/Modules/python.cxx#L111
  set(src_swig_flags
      -c++
      -features
      autodoc=0,directors=0
      -noproxyimport
      -keyword
      -modern
      -modernargs
      -noproxydel
      -fvirtual
      -fastunpack
      -nofastproxy
      -fastquery
      -outputtuple
      -castmode
      -nosafecstrings
      -w402
      -w503
      -w511
      -w302
      -w362
      -w312
      -w389
      -DSWIG_PYTHON_LEGACY_BOOL
      -I${SWIG_DIR}/python
      -I${SWIG_DIR}
      ${src_compiler_definitions}
  )

  message(STATUS "src_swig_flags = ${src_swig_flags}")

  # Tell swig which command-line options to use, allowing user to override
  set(CMAKE_SWIG_FLAGS ${src_swig_flags} ${CMAKE_SWIG_FLAGS})

#  set(src_swig_support_files
#      nupic/py_support/NumpyVector.cpp
#      nupic/py_support/PyArray.cpp
#      nupic/py_support/PyHelpers.cpp
#      nupic/py_support/PythonStream.cpp
#      nupic/bindings/PySparseTensor.cpp
#  )
  # TODO It's not clear why these files need to be added again here, since these
  # files were already included in the nupic core "solo" archive!
  set(src_swig_support_files ${src_py_support_files})

  # NOTE Python extensions shouldn't be linking agains libpython; symbols should
  # be available automatically when python loads the extension.
  #
  # NOTE We don't link our non-windows python extensions with capnproto static libs to force
  # dynamic linkage to pycapnp's capnp symbols at nupic.bindings load-time and
  # avoid the conflict of executing the methods in our own compilation of
  # capnproto (different compiler/STL/version, flags, etc.) on objects created by
  # pycapnp's. Instead, we force dynamic linkage to pycapnp's capnproto symbols
  # by preloading the pycapnp extension in our extension python proxy modules, thus avoiding the conflict from
  # executing the methods in our own compilation of capnproto (different compiler,
  # flags, etc.) on objects created by pycapnp's.
  set(src_swig_link_libraries
      ${src_common_os_libs}
      ${src_lib_static_nupiccore_combined})

  # Set up linker flags for python extension shared libraries
  set(src_swig_extension_link_flags "${PYEXT_LINKER_FLAGS_OPTIMIZED}")

  # Common dependencies for our python extensions for use with
  # SWIG_MODULE_name_EXTRA_DEPS
  set(src_swig_extra_deps Swig)

  # NOTE Windows DLLs are shared executables with their own main; they require
  # all symbols to resolve at link time, so we have to add libpython for this
  # platform
  #
  # NOTE On Windows nupic.bindings builds, we include capnproto because we
  # presently build self-contained CAPNP_LITE on Windows, and Windows
  # nupic/nupic.bindings presently have no dependency on pycapnp, so all symbols
  # have to resolve locally within each extension DLL.
  if("${PLATFORM}" STREQUAL "windows")
    list(APPEND src_swig_link_libraries
         ${PYTHON_LIBRARIES}
         ${CAPNP_LINK_LIBRARIES})

    list(APPEND src_swig_extra_deps CapnProto)
  endif()

  message(STATUS "src_swig_extra_deps     = ${src_swig_extra_deps}")
  message(STATUS "src_swig_link_libraries = ${src_swig_link_libraries}")
  message(STATUS "src_swig_extension_link_flags= ${src_swig_extension_link_flags}")
  message(STATUS "CMAKE_SWIG_FLAGS        = ${CMAKE_SWIG_FLAGS}")


  function(PREPEND_BOILERPLATE_TO_PYTHON_PROXY_MODULE
           PARENT_SWIG_TARGET)
    # Adds a custom command to the given swig target to prepend boilerplate to
    # the swig-generated python proxy module
    # ${CMAKE_SWIG_OUTDIR}/${PARENT_SWIG_TARGET}.py
    #
    # :param PARENT_SWIG_TARGET: the custom command will be added to this
    #   swig_add_module target
    set(preamble_filepath "${CMAKE_SOURCE_DIR}/src/nupic/bindings/swig_proxy_preamble.py")
    set(module_filepath "${CMAKE_SWIG_OUTDIR}/${PARENT_SWIG_TARGET}.py")

    add_custom_command(
      TARGET ${SWIG_MODULE_${PARENT_SWIG_TARGET}_REAL_NAME}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND}
            -DSRC_FILE_1=${preamble_filepath}
            -DSRC_FILE_2=${module_filepath}
            -DTARGET_FILE=${module_filepath}
            -P ${CMAKE_SOURCE_DIR}/src/ConcatTwoFiles.cmake
      COMMENT "Prepending ${preamble_filepath} to swig-generated module ${module_filepath}"
    )
  endfunction(PREPEND_BOILERPLATE_TO_PYTHON_PROXY_MODULE)


  # Algorithms
  set(src_swig_algorithms_files nupic/bindings/algorithms.i)
  set_source_files_properties(${src_swig_algorithms_files} PROPERTIES
                              CPLUSPLUS ON
                              SWIG_MODULE_NAME algorithms)
  set(src_swig_algorithms algorithms)
  # Make sure we don't attempt to execute the swig executable until it is built.
  #set(SWIG_MODULE_${src_swig_algorithms}_EXTRA_DEPS Swig LibStaticNupicCoreCombinedTarget)
  set(SWIG_MODULE_${src_swig_algorithms}_EXTRA_DEPS ${src_swig_extra_deps})
  # Create custom command for generating files from SWIG.
  swig_add_module(${src_swig_algorithms} python
                  ${src_swig_support_files}
                  ${src_swig_algorithms_files})
  list(APPEND src_swig_generated_files ${swig_generated_file_fullname})
  swig_link_libraries(${src_swig_algorithms}
                      ${src_swig_link_libraries})
  set_target_properties(${SWIG_MODULE_${src_swig_algorithms}_REAL_NAME} PROPERTIES
                        LINK_FLAGS "${src_swig_extension_link_flags}")
  prepend_boilerplate_to_python_proxy_module(${src_swig_algorithms})

  # Engine
  set(src_swig_engine_files nupic/bindings/engine_internal.i)
  set_source_files_properties(${src_swig_engine_files} PROPERTIES
                              CPLUSPLUS ON
                              SWIG_MODULE_NAME engine_internal)
  set(src_swig_engine engine_internal)
  # Make sure we don't attempt to execute the swig executable until it is built.
  #set(SWIG_MODULE_${src_swig_engine}_EXTRA_DEPS Swig LibStaticNupicCoreCombinedTarget)
  set(SWIG_MODULE_${src_swig_engine}_EXTRA_DEPS ${src_swig_extra_deps})
  # Create custom command for generating files from SWIG.
  swig_add_module(${src_swig_engine} python
                  ${src_swig_support_files}
                  ${src_swig_engine_files})
  list(APPEND src_swig_generated_files ${swig_generated_file_fullname})
  swig_link_libraries(${src_swig_engine}
                      ${src_swig_link_libraries})
  set_target_properties(${SWIG_MODULE_${src_swig_engine}_REAL_NAME} PROPERTIES
                        LINK_FLAGS "${src_swig_extension_link_flags}")
  prepend_boilerplate_to_python_proxy_module(${src_swig_engine})

  # Experimental
  set(src_swig_experimental_files nupic/bindings/experimental.i)
  set_source_files_properties(${src_swig_experimental_files} PROPERTIES
                              CPLUSPLUS ON
                              SWIG_MODULE_NAME experimental)
  set(src_swig_experimental experimental)
  # Make sure we don't attempt to execute the swig executable until it is built.
  #set(SWIG_MODULE_${src_swig_experimental}_EXTRA_DEPS Swig LibStaticNupicCoreCombinedTarget)
  set(SWIG_MODULE_${src_swig_experimental}_EXTRA_DEPS ${src_swig_extra_deps})
  # Create custom command for generating files from SWIG.
  swig_add_module(${src_swig_experimental} python
                  ${src_swig_support_files}
                  ${src_swig_experimental_files})
  list(APPEND src_swig_generated_files ${swig_generated_file_fullname})
  swig_link_libraries(${src_swig_experimental}
                      ${src_swig_link_libraries})
  set_target_properties(${SWIG_MODULE_${src_swig_experimental}_REAL_NAME} PROPERTIES
                        LINK_FLAGS "${src_swig_extension_link_flags}")
  prepend_boilerplate_to_python_proxy_module(${src_swig_experimental})

  # Math
  set(src_swig_math_files nupic/bindings/math.i)
  set_source_files_properties(${src_swig_math_files} PROPERTIES
                              CPLUSPLUS ON
                              SWIG_MODULE_NAME math)
  set(src_swig_math math)
  # Make sure we don't attempt to execute the swig executable until it is built.
  #set(SWIG_MODULE_${src_swig_math}_EXTRA_DEPS Swig LibStaticNupicCoreCombinedTarget)
  set(SWIG_MODULE_${src_swig_math}_EXTRA_DEPS ${src_swig_extra_deps})
  # Create custom command for generating files from SWIG.
  swig_add_module(${src_swig_math} python
                  ${src_swig_support_files}
                  ${src_swig_math_files})
  list(APPEND src_swig_generated_files ${swig_generated_file_fullname})
  swig_link_libraries(${src_swig_math}
                      ${src_swig_link_libraries})
  set_target_properties(${SWIG_MODULE_${src_swig_math}_REAL_NAME} PROPERTIES
                        LINK_FLAGS "${src_swig_extension_link_flags}")
  prepend_boilerplate_to_python_proxy_module(${src_swig_math})


  # Set properties on swig-generated and support files
  set_source_files_properties(${src_swig_generated_files} PROPERTIES
                              GENERATED TRUE)
  set_source_files_properties(${src_swig_generated_files}
                              PROPERTIES
                              COMPILE_FLAGS ${swig_generated_file_compile_flags})
  set_source_files_properties(${src_swig_support_files}
                              PROPERTIES
                              COMPILE_FLAGS ${src_compile_flags})

  # If a path is specified, copy extensions files to proper location.
  if (PY_EXTENSIONS_DIR)
    install(TARGETS
            ${SWIG_MODULE_${src_swig_algorithms}_REAL_NAME}
            ${SWIG_MODULE_${src_swig_engine}_REAL_NAME}
            ${SWIG_MODULE_${src_swig_experimental}_REAL_NAME}
            ${SWIG_MODULE_${src_swig_math}_REAL_NAME}
            LIBRARY DESTINATION ${PY_EXTENSIONS_DIR})
    install(FILES
            ${PROJECT_BINARY_DIR}/algorithms.py
            ${PROJECT_BINARY_DIR}/engine_internal.py
            ${PROJECT_BINARY_DIR}/experimental.py
            ${PROJECT_BINARY_DIR}/math.py
            DESTINATION ${PY_EXTENSIONS_DIR})
  endif(PY_EXTENSIONS_DIR)

endif() # ${NUPIC_BUILD_PYEXT_MODULES}


#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${src_lib_static_nupiccore_solo}
        ${src_lib_static_nupiccore_combined}
        ${src_lib_static_gtest}
        ${src_executable_helloregion}
        ${src_executable_cppregiontest}
        ${src_executable_pyregiontest}
        ${src_executable_connectionsperformancetest}
        ${src_executable_hellosptp}
        ${src_executable_prototest}
        ${src_executable_gtests}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY nupic/ DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.h*"
        PATTERN "*.hpp.in" EXCLUDE)
install(DIRECTORY nupic/ DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.capnp")
install(DIRECTORY ${PROJECT_BINARY_DIR}/nupic/ DESTINATION include/nupic/
        FILES_MATCHING PATTERN "*.capnp.h")

install(DIRECTORY ${REPOSITORY_DIR}/external/common/include/gtest
        DESTINATION include/gtest
        FILES_MATCHING PATTERN "*.h*")

install(FILES ${PROJECT_BINARY_DIR}/Version.hpp
        DESTINATION include/nupic/)

install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
        DESTINATION include)

install(DIRECTORY "${src_numpy_core}/lib/"
        DESTINATION lib)
install(DIRECTORY "${src_numpy_core}/include/"
        DESTINATION include)

install(FILES ${CAPNP_LINK_LIBRARIES}
        DESTINATION lib)
foreach(directory ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY "${directory}/capnp"
          DESTINATION include)
  install(DIRECTORY "${directory}/kj"
          DESTINATION include)
endforeach()

#install(FILES ${src_lib_static_nupiccore_combined}
#        DESTINATION lib)

install(DIRECTORY nupic/py_support DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.c*")
install(DIRECTORY nupic DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.py")

#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}")
include(CPack)
