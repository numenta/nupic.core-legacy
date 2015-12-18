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

get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(STDLIB "-stdlib=libc++")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(STDLIB "")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(STDLIB "")
endif()

#
# Identify build type - local or deployment (Travis)
# the variable NUPIC_DEPLOYMENT_BUILD must be set in travis CI scripts
#
if($ENV{NUPIC_DEPLOYMENT_BUILD})
  message("Doing a deployment build in TravisCI")
else()
  message("Doing a local build")
endif()

# Compiler `-D*` definitions
set(COMPILER_DEFINITIONS
    -DNTA_OS_${PLATFORM_UPPERCASE}
    -DNTA_ARCH_${BITNESS}
    -DHAVE_CONFIG_H
    -DNTA_INTERNAL
    -DBOOST_NO_WREGEX
    -DNUPIC2
    -DNTA_ASSERTIONS_ON
    -DNTA_ASM)
if(UNIX) # or UNIX like (i.e. APPLE and CYGWIN)
  set(COMPILER_DEFINITIONS ${COMPILER_DEFINITIONS} -DHAVE_UNISTD_H)
elseif(MSVC OR MSYS OR MINGW)
  set(COMPILER_DEFINITIONS
      ${COMPILER_DEFINITIONS}
      -DPSAPI_VERSION=1
      -DAPR_DECLARE_STATIC
      -DAPU_DECLARE_STATIC
      -DZLIB_WINAPI
      -DWIN32
      -D_WINDOWS
      -D_MBCS
      -D_CRT_SECURE_NO_WARNINGS
      -DNDEBUG
      -D_VARIADIC_MAX=10
      -DNOMINMAX
      -DCAPNP_LITE=1)
  if(MSYS OR MINGW)
    set(COMPILER_DEFINITIONS
        ${COMPILER_DEFINITIONS}
        -DHAVE_UNISTD_H)
  endif()
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(COMPILER_DEFINITIONS
      ${COMPILER_DEFINITIONS}
      -DNTA_COMPILER_GNU)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(COMPILER_DEFINITIONS
      ${COMPILER_DEFINITIONS}
      -DNTA_COMPILER_CLANG)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(COMPILER_DEFINITIONS
      ${COMPILER_DEFINITIONS}
      -DNTA_COMPILER_MSVC)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MinGW")
  set(COMPILER_DEFINITIONS
      ${COMPILER_DEFINITIONS}
      -DNTA_COMPILER_GNU
      -D_hypot=hypot)
endif()

add_definitions(${COMPILER_DEFINITIONS})

#
# Find Python Interp, min. v2.7
# And NumPy core directory
#
find_package(PythonInterp 2.7 REQUIRED)
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                        "import numpy; import os; import sys; sys.stdout.write(os.path.dirname(numpy.get_include()))"
                         OUTPUT_VARIABLE NUMPY_CORE)

# Quick way to fixup directory paths to NumPy
get_filename_component(NUMPY_CORE ${NUMPY_CORE}/include/.. ABSOLUTE)

#
# Setup include paths
#
if("${PROJECT_BUILD_ARTIFACTS_DIR}" STREQUAL "")
  set(PROJECT_BUILD_ARTIFACTS_DIR "${PROJECT_BINARY_DIR}/artifacts")
endif()

include_directories(SYSTEM
                    ${NUMPY_CORE}/include
                    ${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR})

#
# Set linker (ld)
# use ld.gold if available
#
execute_process(COMMAND ld.gold --version RESULT_VARIABLE EXIT_CODE)
if(EXIT_CODE EQUAL 0)
  message("Using ld.gold as LINKER.")
  set(CMAKE_LINKER "ld.gold")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
   set(STDLIB "${STDLIB} -static-libgcc -static-libstdc++")
endif()

#
# Enable Optimization flags here
# These are quite aggresive flags, if your code misbehaves
# for strange reasons, try compiling without them.
#
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -mtune=generic -O2")
  set(OPTIMIZATION_FLAGS_CC "-pipe ${OPTIMIZATION_FLAGS_CC}") #TODO use -Ofast instead of -O3
  set(OPTIMIZATION_FLAGS_LT "-O2 ${OPTIMIZATION_FLAGS_LT}")

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -fuse-linker-plugin -flto-report")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -flto -fuse-ld=gold") #TODO fix LTO for clang
    set(OPTIMIZATION_FLAGS_LT "${OPTIMIZATION_FLAGS_LT} -flto") #TODO LTO for clang too
  endif()
endif()


#
# compiler specific settings here
#
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  # MS Visual C
  set(COMMON_COMPILE_FLAGS "/TP /Zc:wchar_t /Gm- /fp:precise /errorReport:prompt /W1 /WX- /GR /Gd /GS /Oy- /EHs /analyze- /nologo")
  set(COMMON_LINK_FLAGS "/NOLOGO /SAFESEH:NO /NODEFAULTLIB:LIBCMT")
  if("${BITNESS}" STREQUAL "32")
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X86")
  else()
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X${BITNESS}")
  endif()

else()
  # LLVM Clang / Gnu GCC
  set(COMMON_COMPILE_FLAGS "-std=c++11 -m${BITNESS} ${STDLIB} -Wextra -Wreturn-type -Wunused -Wno-unused-variable -Wno-unused-parameter -Wno-missing-field-initializers")
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(COMMON_COMPILE_FLAGS "-Werror -Wall -fPIC ${COMMON_COMPILE_FLAGS} -Wno-deprecated-register")
  endif()
  set(COMMON_LINK_FLAGS "-m${BITNESS} ${STDLIB}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} -Og -g")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -O0")
else()
  set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} ${OPTIMIZATION_FLAGS_CC}")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} ${OPTIMIZATION_FLAGS_LT}")
endif()

if(MINGW)
  # This is for GCC 4.8.x
  # http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
  set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} -include cmath")
endif()

# Set compiler-specific options.
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -Wl,--no-undefined")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -Wl,-undefined,error")
endif()

#
# Let CMake know where all of the external files are.
#
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
                      ${REPOSITORY_DIR}/external/common/
                      ${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/
                      ${NUMPY_CORE}/)
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
set(LIB_STATIC_GTEST gtest)
add_library(${LIB_STATIC_GTEST}
            STATIC
            ${REPOSITORY_DIR}/external/common/src/gtest/gtest-all.cpp)
set_target_properties(${LIB_STATIC_GTEST}
                      PROPERTIES COMPILE_FLAGS "${COMMON_COMPILE_FLAGS}"
                      LINK_FLAGS "${COMMON_LINK_FLAGS}")


#
# Find and add libapr1
#
set(LIB_STATIC_APR1 apr-1)
find_library(LIB_STATIC_APR1_LOC
             ${LIB_STATIC_APR1}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_APR1} STATIC IMPORTED GLOBAL)

#
# Find and add libaprutil-1
#
set(LIB_STATIC_APRUTIL1 aprutil-1)
find_library(LIB_STATIC_APRUTIL1_LOC
             ${LIB_STATIC_APRUTIL1}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_APRUTIL1} STATIC IMPORTED GLOBAL)

#
# Find and add libyaml
#
set(LIB_STATIC_YAML yaml)
find_library(LIB_STATIC_YAML_LOC
             ${LIB_STATIC_YAML}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_YAML} STATIC IMPORTED GLOBAL)

#
# Find and add libyaml-cpp
#
set(LIB_STATIC_YAML_CPP yaml-cpp)
find_library(LIB_STATIC_YAML_CPP_LOC
             ${LIB_STATIC_YAML_CPP}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_YAML_CPP} STATIC IMPORTED GLOBAL)

#
# Find and add libz
#
set(LIB_STATIC_Z z)
find_library(LIB_STATIC_Z_LOC
             ${LIB_STATIC_Z}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_Z} STATIC IMPORTED GLOBAL)

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
set(CAPNP_SPECS_REL
  nupic/proto/BitHistory.capnp
  nupic/proto/ClaClassifier.capnp
  nupic/proto/ConnectionsProto.capnp
  nupic/proto/LinkProto.capnp
  nupic/proto/Map.capnp
  nupic/proto/NetworkProto.capnp
  nupic/proto/PyRegionProto.capnp
  nupic/proto/RandomProto.capnp
  nupic/proto/RegionProto.capnp
  nupic/proto/SparseBinaryMatrixProto.capnp
  nupic/proto/SparseMatrixProto.capnp
  nupic/proto/SpatialPoolerProto.capnp
  nupic/proto/TemporalMemoryProto.capnp
  nupic/proto/TestNodeProto.capnp
  nupic/proto/VectorFileSensorProto.capnp
)

foreach(spec ${CAPNP_SPECS_REL})
  list(APPEND CAPNP_SPECS ${PROJECT_SOURCE_DIR}/${spec})
  list(APPEND CAPNP_HDRS ${PROJECT_BINARY_DIR}/${spec}.h)
  list(APPEND CAPNP_SRCS ${PROJECT_BINARY_DIR}/${spec}.c++)
endforeach(spec)
set(CAPNP_ALL "${CAPNP_HDRS};${CAPNP_SRCS}")

# Create custom target for generating C++ code from .capnp schema files.
create_capnpc_command(capnp_specs
                      "${CAPNP_SPECS}"
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}
                      ${PROJECT_BINARY_DIR}
                      "${CAPNP_ALL}")

#
# Python support
#
set (PY_SUPPORT_FILES
     nupic/py_support/NumpyVector.cpp
     nupic/py_support/PyArray.cpp
     nupic/py_support/PyHelpers.cpp
     nupic/py_support/PythonStream.cpp
     nupic/py_support/PyCapnp.cpp
)
if(NOT "${PLATFORM}" STREQUAL "windows")
  list(APPEND PY_SUPPORT_FILES nupic/py_support/CapnpToPycapnp.cpp)
endif()

message(STATUS "CAPNP_EXECUTABLE   = ${CAPNP_EXECUTABLE}")
message(STATUS "CAPNP_DEFINITIONS   = ${CAPNP_DEFINITIONS}")
message(STATUS "CAPNP_INCLUDE_DIRS  = ${CAPNP_INCLUDE_DIRS}")
message(STATUS "CAPNP_LIBRARIES     = ${CAPNP_LIBRARIES}")
message(STATUS "CAPNP_LIBRARIES_LITE= ${CAPNP_LIBRARIES_LITE}")

#
# Add the pycapnp Python package to includes
if(NOT "${PLATFORM}" STREQUAL "windows")
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                          "import capnp; import os; import sys; sys.stdout.write(os.path.dirname(os.path.dirname(capnp.__file__)))"
                  OUTPUT_VARIABLE PYCAPNP)
  message(STATUS "PYCAPNP             = ${PYCAPNP}")
  include_directories(SYSTEM ${PYCAPNP})
endif()

#
# Setup libnupic_core
#
set(LIB_STATIC_NUPICCORE nupic_core_solo)

if(MSVC OR MSYS OR MINGW)
  set(COMMON_LIBS
      ${PYTHON_LIBRARIES}
      ${CAPNP_LIBRARIES_LITE})
else()
  set(COMMON_LIBS
      ${PYTHON_LIBRARIES}
      ${CAPNP_LIBRARIES})
endif()

if("${PLATFORM}" STREQUAL "linux")
  list(APPEND COMMON_OS_LIBS pthread dl)
elseif("${PLATFORM}" STREQUAL "darwin")
  list(APPEND COMMON_OS_LIBS c++abi)
elseif(MSYS OR MINGW)
  list(APPEND COMMON_OS_LIBS psapi ws2_32 wsock32 rpcrt4)
elseif("${PLATFORM}" STREQUAL "windows")
  list(APPEND COMMON_OS_LIBS oldnames.lib psapi.lib ws2_32.lib)
endif()

set(LIB_STATIC_NUPICCORE_SRCS
    ${CAPNP_SRCS}
    ${PY_SUPPORT_FILES}
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
    nupic/algorithms/Linear.cpp
    nupic/algorithms/OutSynapse.cpp
    nupic/algorithms/Segment.cpp
    nupic/algorithms/SegmentUpdate.cpp
    nupic/algorithms/SpatialPooler.cpp
    nupic/algorithms/TemporalMemory.cpp
    nupic/algorithms/Svm.cpp
    nupic/engine/Collections.cpp
    nupic/engine/Input.cpp
    nupic/engine/Link.cpp
    nupic/engine/LinkPolicyFactory.cpp
    nupic/engine/Network.cpp
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
    nupic/test/Tester.cpp
    nupic/types/BasicType.cpp
    nupic/types/Fraction.cpp
    nupic/utils/LoggingException.cpp
    nupic/utils/LogItem.cpp
    nupic/utils/Random.cpp
    nupic/utils/StringUtils.cpp
    nupic/utils/TRandom.cpp
    nupic/utils/Watcher.cpp)
set(LIB_STATIC_NUPICCORE_LINK_LIBS
    ${LIB_STATIC_YAML_CPP_LOC}
    ${LIB_STATIC_YAML_LOC}
    ${LIB_STATIC_APR1_LOC}
    ${LIB_STATIC_APRUTIL1_LOC}
    ${LIB_STATIC_Z_LOC}
    ${COMMON_LIBS}
    ${COMMON_OS_LIBS})
set(LIB_STATIC_NUPICCORE_COMPILE_FLAGS
    "${COMMON_COMPILE_FLAGS} -I${NUMPY_CORE}/include")
set(LIB_STATIC_NUPICCORE_LINK_FLAGS
    "${COMMON_LINK_FLAGS} -static")

add_library(${LIB_STATIC_NUPICCORE} STATIC ${LIB_STATIC_NUPICCORE_SRCS})
target_link_libraries(${LIB_STATIC_NUPICCORE}
                      ${LIB_STATIC_NUPICCORE_LINK_LIBS})
set_target_properties(${LIB_STATIC_NUPICCORE} PROPERTIES COMPILE_FLAGS
                      ${LIB_STATIC_NUPICCORE_COMPILE_FLAGS})
set_target_properties(${LIB_STATIC_NUPICCORE} PROPERTIES LINK_FLAGS
                      ${LIB_STATIC_NUPICCORE_LINK_FLAGS})

# Add new library to common libs for rest of targets.
set(COMMON_LIBS ${LIB_STATIC_NUPICCORE} ${COMMON_LIBS})

#
# Setup test_cpp_region
#
set(EXECUTABLE_CPPREGIONTEST cpp_region_test)
add_executable(${EXECUTABLE_CPPREGIONTEST} test/integration/CppRegionTest.cpp)
target_link_libraries(${EXECUTABLE_CPPREGIONTEST} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_CPPREGIONTEST} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_CPPREGIONTEST} PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_custom_target(tests_cpp_region
                  COMMAND ${EXECUTABLE_CPPREGIONTEST}
                  DEPENDS ${EXECUTABLE_CPPREGIONTEST}
                  VERBATIM)

#
# Setup test_py_region
#
set(EXECUTABLE_PYREGIONTEST py_region_test)
add_executable(${EXECUTABLE_PYREGIONTEST} test/integration/PyRegionTest.cpp)
target_link_libraries(${EXECUTABLE_PYREGIONTEST} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_PYREGIONTEST}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_PYREGIONTEST}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_custom_target(tests_py_region
                  COMMAND ${EXECUTABLE_PYREGIONTEST}
                  DEPENDS ${EXECUTABLE_PYREGIONTEST}
                  VERBATIM)


#
# Setup test_connections_performance
#
set(EXECUTABLE_CONNECTIONSPERFORMANCETEST connections_performance_test)
add_executable(${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
               test/integration/ConnectionsPerformanceTest.cpp)
target_link_libraries(${EXECUTABLE_CONNECTIONSPERFORMANCETEST} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_custom_target(tests_connections_performance
                  COMMAND ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                  DEPENDS ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                  VERBATIM)

#
# Setup helloregion example
#
set(EXECUTABLE_HELLOREGION helloregion)
add_executable(${EXECUTABLE_HELLOREGION} examples/regions/HelloRegions.cpp)
target_link_libraries(${EXECUTABLE_HELLOREGION} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_HELLOREGION}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_HELLOREGION}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_HELLOREGION} ${COMMON_LIBS})

#
# Setup prototest example
#
set(EXECUTABLE_PROTOTEST prototest)
add_executable(${EXECUTABLE_PROTOTEST} examples/prototest.cpp)
target_link_libraries(${EXECUTABLE_PROTOTEST} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_PROTOTEST}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_PROTOTEST}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_PROTOTEST} ${COMMON_LIBS})

#
# Setup HelloSP_TP example
#
set(EXECUTABLE_HELLOSPTP hello_sp_tp)
add_executable(${EXECUTABLE_HELLOSPTP} examples/algorithms/HelloSP_TP.cpp)
target_link_libraries(${EXECUTABLE_HELLOSPTP} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_HELLOSPTP}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_HELLOSPTP}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_HELLOSPTP} ${COMMON_LIBS})


#
# Setup gtests
#
set(EXECUTABLE_GTESTS unit_tests)
add_executable(${EXECUTABLE_GTESTS}
               test/unit/algorithms/CondProbTableTest.cpp
               test/unit/algorithms/ConnectionsTest.cpp
               test/unit/algorithms/FastCLAClassifierTest.cpp
               test/unit/algorithms/NearestNeighborUnitTest.cpp
               test/unit/algorithms/SpatialPoolerTest.cpp
               test/unit/algorithms/TemporalMemoryTest.cpp
               test/unit/engine/InputTest.cpp
               test/unit/engine/NetworkTest.cpp
               test/unit/engine/UniformLinkPolicyTest.cpp
               test/unit/engine/YAMLUtilsTest.cpp
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
               test/unit/test/TesterTest.cpp
               test/unit/types/BasicTypeTest.cpp
               test/unit/types/ExceptionTest.cpp
               test/unit/types/FractionTest.cpp
               test/unit/UnitTestMain.cpp
               test/unit/utils/RandomTest.cpp
               test/unit/utils/WatcherTest.cpp)
target_link_libraries(${EXECUTABLE_GTESTS}
                      ${LIB_STATIC_GTEST}
                      ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_GTESTS}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS}
                                 LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_GTESTS} ${LIB_STATIC_GTEST} ${COMMON_LIBS})
set(TEST_HEADERS
    test/unit/math/DenseTensorUnitTest.hpp
    test/unit/math/DomainUnitTest.hpp
    test/unit/math/IndexUnitTest.hpp
    test/unit/math/MathsTest.hpp
    test/unit/math/SparseMatrix01UnitTest.hpp
    test/unit/math/SparseMatrixUnitTest.hpp
    test/unit/math/SparseTensorUnitTest.hpp
    test/unit/ntypes/ArrayTest.hpp
    test/unit/ntypes/BufferTest.hpp
    test/unit/ntypes/CollectionTest.hpp
    test/unit/ntypes/DimensionsTest.hpp
    test/unit/ntypes/MemParserTest.hpp
    test/unit/ntypes/MemStreamTest.hpp
    test/unit/ntypes/NodeSetTest.hpp
    test/unit/ntypes/ScalarTest.hpp
    test/unit/ntypes/ValueTest.hpp
    test/unit/os/DirectoryTest.hpp
    test/unit/os/EnvTest.hpp
    test/unit/os/OSTest.hpp
    test/unit/os/PathTest.hpp
    test/unit/os/RegexTest.hpp
    test/unit/os/TimerTest.hpp
    test/unit/py_support/PyHelpersTest.hpp
    test/unit/test/TesterTest.hpp
    test/unit/types/BasicTypeTest.hpp
    test/unit/types/ExceptionTest.hpp
    test/unit/types/FractionTest.hpp
    test/unit/utils/RandomTest.hpp
    test/unit/utils/WatcherTest.hpp)
set(tests_file_content "")
set(headers_file_content "")
foreach(file ${TEST_HEADERS})
  get_filename_component(test ${file} NAME_WE)
  set(tests_file_content "${tests_file_content}ADD_TEST(${test});\n")
  set(headers_file_content "${headers_file_content}#include \"${file}\"\n")
endforeach()
file(WRITE "${PROJECT_BINARY_DIR}/AddTests.hpp" "${tests_file_content}")
file(WRITE "${PROJECT_BINARY_DIR}/AddTestHeaders.hpp" "${headers_file_content}")

#
# tests_all just calls other targets
#
add_custom_target(tests_all
                  DEPENDS tests_cpp_region
                  DEPENDS tests_unit
                  COMMENT "Running all tests"
                  VERBATIM)

if(UNIX OR MSYS OR MINGW)
  if(MSYS OR MINGW)
    add_custom_target(combined ALL
                      COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${LIB_STATIC_NUPICCORE}>
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_CPP_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APRUTIL1_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APR1_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_Z_LOC}
                      COMMAND ${CMAKE_AR} rcs libnupic_core.a *.o *.obj
                      DEPENDS ${LIB_STATIC_NUPICCORE}
                              ${LIB_STATIC_GTEST}
                              ${EXECUTABLE_HELLOREGION}
                              ${EXECUTABLE_CPPREGIONTEST}
                              ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                              ${EXECUTABLE_GTESTS}
                              ${EXECUTABLE_HELLOSPTP})
  else()
    add_custom_target(combined ALL
                      COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${LIB_STATIC_NUPICCORE}>
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_CPP_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APRUTIL1_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APR1_LOC}
                      COMMAND ${CMAKE_AR} -x ${LIB_STATIC_Z_LOC}
                      COMMAND ${CMAKE_AR} rcs libnupic_core.a *.o
                      DEPENDS ${LIB_STATIC_NUPICCORE}
                              ${LIB_STATIC_GTEST}
                              ${EXECUTABLE_HELLOREGION}
                              ${EXECUTABLE_CPPREGIONTEST}
                              ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                              ${EXECUTABLE_GTESTS}
                              ${EXECUTABLE_HELLOSPTP})
  endif()
else()
  add_custom_target(combined ALL
                    COMMAND lib.exe /OUT:nupic_core.lib
                            $<TARGET_FILE:${LIB_STATIC_NUPICCORE}>
                            ${LIB_STATIC_YAML_CPP_LOC}
                            ${LIB_STATIC_YAML_LOC}
                            ${LIB_STATIC_APRUTIL1_LOC}
                            ${LIB_STATIC_APR1_LOC}
                            ${LIB_STATIC_Z_LOC}
                    DEPENDS ${LIB_STATIC_NUPICCORE}
                            ${LIB_STATIC_GTEST}
                            ${EXECUTABLE_HELLOREGION}
                            ${EXECUTABLE_CPPREGIONTEST}
                            ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
                            ${EXECUTABLE_GTESTS}
                            ${EXECUTABLE_HELLOSPTP}
                    VERBATIM)
endif()

#
# Call SWIG to generate wrapper code for Python.
#
include(UseSWIG)

# Set the output location for the language modules that are created.
set(CMAKE_SWIG_OUTDIR ${PROJECT_BINARY_DIR})

# Make sure the directory exists for the generated C++ files.
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/nupic/bindings)

# SWIG options from:
# https://github.com/swig/swig/blob/master/Source/Modules/python.cxx#L111
set(SWIG_FLAGS
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
    ${COMPILER_DEFINITIONS}
)

set(SWIG_SUPPORT_FILES
    nupic/py_support/NumpyVector.cpp
    nupic/py_support/PyArray.cpp
    nupic/py_support/PyHelpers.cpp
    nupic/py_support/PythonStream.cpp
    nupic/bindings/PySparseTensor.cpp
)

set(SWIG_LINK_LIBRARIES
    ${PROJECT_BINARY_DIR}/libnupic_core.a
    ${PYTHON_LIBRARIES}
    ${COMMON_OS_LIBS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(SWIG_LINK_LIBRARIES
      ${SWIG_LINK_LIBRARIES}
      ${CAPNP_LIBRARIES_LITE})
else()
  set(SWIG_LINK_LIBRARIES
      ${SWIG_LINK_LIBRARIES}
      ${CAPNP_LIBRARIES})
endif()

# Algorithms
set(SWIG_ALGORITHMS_FILES nupic/bindings/algorithms.i)
set_source_files_properties(${SWIG_ALGORITHMS_FILES} PROPERTIES
                            CPLUSPLUS ON
                            SWIG_MODULE_NAME algorithms
                            SWIG_FLAGS "${SWIG_FLAGS}")
set(SWIG_ALGORITHMS algorithms)
# Make sure we don't attempt to execute the swig executable until it is built.
set(SWIG_MODULE_${SWIG_ALGORITHMS}_EXTRA_DEPS Swig combined)
# Create custom command for generating files from SWIG.
swig_add_module(${SWIG_ALGORITHMS} python
                ${SWIG_SUPPORT_FILES}
                ${SWIG_ALGORITHMS_FILES})
list(APPEND SWIG_GENERATED_FILES ${swig_generated_file_fullname})
#TODO add_dependencies(${SWIG_MODULE_${SWIG_ALGORITHMS}_REAL_NAME}
swig_link_libraries(${SWIG_ALGORITHMS}
                    ${SWIG_LINK_LIBRARIES})

# Engine
set(SWIG_ENGINE_FILES nupic/bindings/engine_internal.i)
set_source_files_properties(${SWIG_ENGINE_FILES} PROPERTIES
                            CPLUSPLUS ON
                            SWIG_MODULE_NAME engine_internal
                            SWIG_FLAGS "${SWIG_FLAGS}")
set(SWIG_ENGINE engine_internal)
# Make sure we don't attempt to execute the swig executable until it is built.
set(SWIG_MODULE_${SWIG_ENGINE}_EXTRA_DEPS Swig combined)
# Create custom command for generating files from SWIG.
swig_add_module(${SWIG_ENGINE} python
                ${SWIG_SUPPORT_FILES}
                ${SWIG_ENGINE_FILES})
list(APPEND SWIG_GENERATED_FILES ${swig_generated_file_fullname})
swig_link_libraries(${SWIG_ENGINE}
                    ${SWIG_LINK_LIBRARIES})

# Math
set(SWIG_MATH_FILES nupic/bindings/math.i)
set_source_files_properties(${SWIG_MATH_FILES} PROPERTIES
                            CPLUSPLUS ON
                            SWIG_MODULE_NAME math
                            SWIG_FLAGS "${SWIG_FLAGS}")
set(SWIG_MATH math)
# Make sure we don't attempt to execute the swig executable until it is built.
set(SWIG_MODULE_${SWIG_MATH}_EXTRA_DEPS Swig combined)
# Create custom command for generating files from SWIG.
swig_add_module(${SWIG_MATH} python
                ${SWIG_SUPPORT_FILES}
                ${SWIG_MATH_FILES})
list(APPEND SWIG_GENERATED_FILES ${swig_generated_file_fullname})
swig_link_libraries(${SWIG_MATH}
                    ${SWIG_LINK_LIBRARIES})

set_source_files_properties(${SWIG_GENERATED_FILES} PROPERTIES
                            GENERATED TRUE)
set_source_files_properties(${SWIG_GENERATED_FILES} ${SWIG_SUPPORT_FILES}
                            PROPERTIES
                            COMPILE_FLAGS ${COMMON_COMPILE_FLAGS}
                            LINK_FLAGS ${COMMON_LINK_FLAGS})

# If a path is specified, copy extensions files to proper location.
if (PY_EXTENSIONS_DIR)
  install(TARGETS
          ${SWIG_MODULE_${SWIG_ALGORITHMS}_REAL_NAME}
          ${SWIG_MODULE_${SWIG_ENGINE}_REAL_NAME}
          ${SWIG_MODULE_${SWIG_MATH}_REAL_NAME}
          LIBRARY DESTINATION ${PY_EXTENSIONS_DIR})
  install(FILES
          ${PROJECT_BINARY_DIR}/algorithms.py
          ${PROJECT_BINARY_DIR}/engine_internal.py
          ${PROJECT_BINARY_DIR}/math.py
          DESTINATION ${PY_EXTENSIONS_DIR})
endif(PY_EXTENSIONS_DIR)


#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${LIB_STATIC_NUPICCORE}
        ${LIB_STATIC_GTEST}
        ${EXECUTABLE_HELLOREGION}
        ${EXECUTABLE_CPPREGIONTEST}
        ${EXECUTABLE_PYREGIONTEST}
        ${EXECUTABLE_CONNECTIONSPERFORMANCETEST}
        ${EXECUTABLE_HELLOSPTP}
        ${EXECUTABLE_PROTOTEST}
        ${EXECUTABLE_GTESTS}
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
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/lib/"
        DESTINATION lib)
install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
        DESTINATION include)
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}${PLATFORM_SUFFIX}/include/"
        DESTINATION include)
install(DIRECTORY "${NUMPY_CORE}/lib/"
        DESTINATION lib)
install(DIRECTORY "${NUMPY_CORE}/include/"
        DESTINATION include)
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  install(FILES ${CAPNP_LIBRARIES_LITE}
          DESTINATION lib)
else()
  install(FILES ${CAPNP_LIBRARIES}
          DESTINATION lib)
endif()
foreach(directory ${CAPNP_INCLUDE_DIRS})
  install(DIRECTORY "${directory}/capnp"
          DESTINATION include)
  install(DIRECTORY "${directory}/kj"
          DESTINATION include)
endforeach()
if(UNIX OR MSYS OR MINGW)
  install(FILES "${PROJECT_BINARY_DIR}/libnupic_core.a"
          DESTINATION lib)
else()
  install(FILES "${PROJECT_BINARY_DIR}/nupic_core.lib"
          DESTINATION lib)
endif()

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
