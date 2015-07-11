# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2014, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
#
# Contributor(s):
#   David Ragazzi (@DavidRagazzi): Full conversion and adaptation from
#     Autotools scripts
#   Jim Bridgewater (jwbwater@gmail.com): Updated option help messages
#   Austin Marshall (amarshall@numenta.com): On-going maintenance
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)
project(nupic_core CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF) # toggle for cmake debug

#
# Include extra cmake files
#
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

#
# Identify platform
#
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(PLATFORM "darwin")
  set(STDLIB "-stdlib=libc++")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(PLATFORM "linux")
  set(STDLIB "")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(PLATFORM "windows")
  set(STDLIB "")
endif()
string(TOUPPER ${PLATFORM} PLATFORM_UPPERCASE)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(STDLIB "${STDLIB} -static-libgcc -static-libstdc++")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()

# Compiler `-D*` definitions
add_definitions(-DNTA_OS_${PLATFORM_UPPERCASE}
                -DNTA_ARCH_${BITNESS}
                -DHAVE_CONFIG_H
                -DNTA_INTERNAL
                -DBOOST_NO_WREGEX
                -DNUPIC2
                -DNTA_ASSERTIONS_ON
                -DNTA_ASM)
if(UNIX) # or UNIX like (i.e. APPLE and CYGWIN)
  add_definitions(-DHAVE_UNISTD_H)
elseif(WIN32) # incl. win64
  add_definitions(-DPSAPI_VERSION=1
                  -DAPR_DECLARE_STATIC
                  -DAPU_DECLARE_STATIC
                  -DZLIB_WINAPI
                  -DWIN32
                  -D_WINDOWS
                  -D_MBCS
                  -D_CRT_SECURE_NO_WARNINGS
                  -DNDEBUG)
endif()
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  add_definitions(-DNTA_COMPILER_GNU)
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  add_definitions(-DNTA_COMPILER_CLANG)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  add_definitions(-DNTA_COMPILER_MSVC
                  -DCAPNP_LITE=1
                  -D_VARIADIC_MAX=10
                  -DNOMINMAX)
endif()


#
# Identify build type - local or deployment (Travis)
# the variable NUPIC_DEPLOYMENT_BUILD must be set in travis CI scripts
#
if($ENV{NUPIC_DEPLOYMENT_BUILD})
  message("doing deployment build in TravisCI")
else()
  message("doing local build")
endif()




#
# Setup include paths
#
get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR}/.. ABSOLUTE)

if("${PROJECT_BUILD_ARTIFACTS_DIR}" STREQUAL "")
  set(PROJECT_BUILD_ARTIFACTS_DIR "${REPOSITORY_DIR}/build/artifacts")
endif()

include_directories(SYSTEM
                    "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/include"
                    "${REPOSITORY_DIR}/external/common/include"
                    "${PROJECT_SOURCE_DIR}"
                    "${PROJECT_BINARY_DIR}")

#
# Set linker (ld)
# use ld.gold if available
#
execute_process(COMMAND ld.gold --version
                RESULT_VARIABLE EXIT_CODE)
if(EXIT_CODE EQUAL 0)
  message("Using ld.gold as LINKER.")
  set(CMAKE_LINKER "ld.gold")
endif()


#
# Identify build type - local or deployment (Travis)
# the variable NUPIC_DEPLOYMENT_BUILD must be set in travis CI scripts
#
if($ENV{NUPIC_DEPLOYMENT_BUILD})
  message("doing deployment build in TravisCI")
  if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -mtune=generic -O2")
    set(OPTIMIZATION_FLAGS_LT "-O2 ${OPTIMIZATION_FLAGS_LT}")
  endif()
else()
  message("doing local build")
  if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -march=native -O3")
    set(OPTIMIZATION_FLAGS_LT "-O3 ${OPTIMIZATION_FLAGS_LT}")
  endif()
endif()


#
# Enable Optimization flags here
# these are quite aggresive flags, if your code misbehaves for strange reasons, try
# compiling without them.
#
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(OPTIMIZATION_FLAGS_CC "-pipe ${OPTIMIZATION_FLAGS_CC}") #TODO use -Ofast instead of -O3
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -fuse-linker-plugin -flto-report")
    set(OPTIMIZATION_FLAGS_CC "${OPTIMIZATION_FLAGS_CC} -flto -fuse-ld=gold") #TODO fix LTO for clang
    set(OPTIMIZATION_FLAGS_LT "${OPTIMIZATION_FLAGS_LT} -flto") #TODO LTO for clang too
  endif()
endif()


#
# compiler specific settings here
#
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(COMMON_COMPILE_FLAGS "/TP /Zc:wchar_t /Gm- /fp:precise /errorReport:prompt /W1 /WX- /GR /Gd /GS /Oy- /EHs /analyze- /nologo")
  set(COMMON_LINK_FLAGS "/NOLOGO /SAFESEH:NO /NODEFAULTLIB:LIBCMT")
  if("${BITNESS}" STREQUAL "32")
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X86")
  else()
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} /MACHINE:X${BITNESS}")
  endif()
else()
  # clang/gcc
  set(COMMON_COMPILE_FLAGS "-fPIC -std=c++11 -m${BITNESS} ${STDLIB} -fvisibility=hidden -Werror -Wall -Wextra -Wreturn-type -Wunused -Wno-unused-parameter -Wno-missing-field-initializers")
  set(COMMON_LINK_FLAGS "-m${BITNESS} ${STDLIB}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -static-libstdc++")
  endif()
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} -Og -g")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} -O0")
else()
  set(COMMON_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} ${OPTIMIZATION_FLAGS_CC}")
  set(COMMON_LINK_FLAGS "${COMMON_LINK_FLAGS} ${OPTIMIZATION_FLAGS_LT}")
endif()

#
# Let CMake know where all of the external files are.
#
set(CMAKE_PREFIX_PATH ${REPOSITORY_DIR}/external/common/
                      ${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/)
#
# Make sure the Cap'n Proto generation scripts are on the PATH.
#
set(ENV{PATH} "$ENV{PATH}:${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/bin")

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
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_APR1} STATIC IMPORTED GLOBAL)

#
# Find and add libaprutil-1
#
set(LIB_STATIC_APRUTIL1 aprutil-1)
find_library(LIB_STATIC_APRUTIL1_LOC
             ${LIB_STATIC_APRUTIL1}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_APRUTIL1} STATIC IMPORTED GLOBAL)

#
# Find and add libyaml
#
set(LIB_STATIC_YAML yaml)
find_library(LIB_STATIC_YAML_LOC
             ${LIB_STATIC_YAML}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_YAML} STATIC IMPORTED GLOBAL)

#
# Find and add libyaml-cpp
#
set(LIB_STATIC_YAML_CPP yaml-cpp)
find_library(LIB_STATIC_YAML_CPP_LOC
             ${LIB_STATIC_YAML_CPP}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_YAML_CPP} STATIC IMPORTED GLOBAL)

#
# Find and add libz
#
set(LIB_STATIC_Z z)
find_library(LIB_STATIC_Z_LOC
             ${LIB_STATIC_Z}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_Z} STATIC IMPORTED GLOBAL)

#
# Find and add libkj
#
set(LIB_STATIC_KJ kj)
find_library(LIB_STATIC_KJ_LOC
             ${LIB_STATIC_KJ}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_KJ} STATIC IMPORTED GLOBAL)

#
# Find and add libcapnp
#
set(LIB_STATIC_CAPNP capnp)
find_library(LIB_STATIC_CAPNP_LOC
             ${LIB_STATIC_CAPNP}
             "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
             NO_DEFAULT_PATH)
add_library(${LIB_STATIC_CAPNP} STATIC IMPORTED GLOBAL)

#
# Generate Cap'n Proto C++ files
#
set(CAPNPC_OUTPUT_DIR ${CMAKE_BINARY_DIR})
find_package(CapnProto REQUIRED)
include_directories(${CAPNP_INCLUDE_DIRS})
# List all .capnp files here. The C++ files will be generated and included
# when compiling later on.
set(CAPNP_SPECS
    nupic/proto/BitHistory.capnp
    nupic/proto/ClaClassifier.capnp
    nupic/proto/ConnectionsProto.capnp
    nupic/proto/LinkProto.capnp
    nupic/proto/Map.capnp
    nupic/proto/NetworkProto.capnp
    nupic/proto/RandomProto.capnp
    nupic/proto/RegionProto.capnp
    nupic/proto/SparseBinaryMatrixProto.capnp
    nupic/proto/SparseMatrixProto.capnp
    nupic/proto/SpatialPoolerProto.capnp
    nupic/proto/TemporalMemoryProto.capnp
    nupic/proto/TestNodeProto.capnp
    nupic/proto/VectorFileSensorProto.capnp
)
capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS ${CAPNP_SPECS})

#
# Setup libnupic_core
#
set(LIB_STATIC_NUPICCORE nupic_core_solo)

set(COMMON_LIBS ${LIB_STATIC_NUPICCORE}
                ${LIB_STATIC_KJ_LOC}
                ${LIB_STATIC_CAPNP_LOC})


if("${PLATFORM}" STREQUAL "linux")
  list(APPEND COMMON_LIBS pthread dl)
elseif("${PLATFORM}" STREQUAL "darwin")
  list(APPEND COMMON_LIBS c++abi)
elseif("${PLATFORM}" STREQUAL "windows")
  list(APPEND COMMON_LIBS oldnames.lib psapi.lib ws2_32.lib)
endif()

ADD_LIBRARY(${LIB_STATIC_NUPICCORE}
            STATIC
            ${CAPNP_SRCS}
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
target_link_libraries(${LIB_STATIC_NUPICCORE}
                      ${LIB_STATIC_YAML_CPP_LOC}
                      ${LIB_STATIC_YAML_LOC}
                      ${LIB_STATIC_APR1_LOC}
                      ${LIB_STATIC_APRUTIL1_LOC}
                      ${LIB_STATIC_Z_LOC}
                      ${LIB_STATIC_KJ_LOC}
                      ${LIB_STATIC_CAPNP_LOC})
set_target_properties(${LIB_STATIC_NUPICCORE} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${LIB_STATIC_NUPICCORE} PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS} -static")

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

set_target_properties(${LIB_STATIC_NUPICCORE} PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS} -static")

#
# Setup test_py_region
#
set(EXECUTABLE_PYREGIONTEST py_region_test)
add_executable(${EXECUTABLE_PYREGIONTEST} test/integration/PyRegionTest.cpp)
target_link_libraries(${EXECUTABLE_PYREGIONTEST} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_PYREGIONTEST} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_PYREGIONTEST}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")

#
# Setup helloregion example
#
set(EXECUTABLE_HELLOREGION helloregion)
add_executable(${EXECUTABLE_HELLOREGION} examples/regions/HelloRegions.cpp)
target_link_libraries(${EXECUTABLE_HELLOREGION} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_HELLOREGION} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_HELLOREGION}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_HELLOREGION} ${COMMON_LIBS})

#
# Setup prototest example
#
set(EXECUTABLE_PROTOTEST prototest)
add_executable(${EXECUTABLE_PROTOTEST} examples/prototest.cpp)
target_link_libraries(${EXECUTABLE_PROTOTEST}
                      ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_PROTOTEST} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_PROTOTEST}
                      PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_PROTOTEST} ${COMMON_LIBS})

#
# Setup HelloSP_TP example
#
set(EXECUTABLE_HELLOSPTP hello_sp_tp)
add_executable(${EXECUTABLE_HELLOSPTP} examples/algorithms/HelloSP_TP.cpp)
target_link_libraries(${EXECUTABLE_HELLOSPTP} ${COMMON_LIBS})
set_target_properties(${EXECUTABLE_HELLOSPTP} PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS})
set_target_properties(${EXECUTABLE_HELLOSPTP} PROPERTIES LINK_FLAGS "${COMMON_LINK_FLAGS}")
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
               test/unit/test/TesterTest.cpp
               test/unit/types/BasicTypeTest.cpp
               test/unit/types/ExceptionTest.cpp
               test/unit/types/FractionTest.cpp
               test/unit/UnitTestMain.cpp
               test/unit/utils/RandomTest.cpp
               test/unit/utils/WatcherTest.cpp)
target_link_libraries(${EXECUTABLE_GTESTS}
                      ${COMMON_LIBS}
                      ${LIB_STATIC_GTEST})
set_target_properties(${EXECUTABLE_GTESTS}
                      PROPERTIES COMPILE_FLAGS ${COMMON_COMPILE_FLAGS}
                                 LINK_FLAGS "${COMMON_LINK_FLAGS}")
add_dependencies(${EXECUTABLE_GTESTS} ${LIB_STATIC_GTEST} ${COMMON_LIBS})
add_custom_target(tests_unit
                  COMMAND ${EXECUTABLE_GTESTS} "--gtest_output=xml:${PROJECT_BUILD_ARTIFACTS_DIR}/unit_tests_report.xml"
                  DEPENDS ${EXECUTABLE_GTESTS}
                  VERBATIM)
set(TEST_HEADERS
    test/unit/algorithms/CondProbTableTest.hpp
    test/unit/algorithms/ConnectionsTest.hpp
    test/unit/algorithms/NearestNeighborUnitTest.hpp
    test/unit/algorithms/SpatialPoolerTest.hpp
    test/unit/engine/InputTest.hpp
    test/unit/engine/NetworkTest.hpp
    test/unit/engine/UniformLinkPolicyTest.hpp
    test/unit/engine/YAMLUtilsTest.hpp
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

if(UNIX)
  add_custom_target(combined ALL
                    COMMAND ${CMAKE_AR} -x $<TARGET_FILE:${LIB_STATIC_NUPICCORE}>
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_CPP_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_YAML_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APRUTIL1_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_APR1_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_Z_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_KJ_LOC}
                    COMMAND ${CMAKE_AR} -x ${LIB_STATIC_CAPNP_LOC}
                    COMMAND ${CMAKE_AR} rcs libnupic_core.a *.o
                    DEPENDS ${LIB_STATIC_NUPICCORE}
                            ${LIB_STATIC_GTEST}
                            ${EXECUTABLE_HELLOREGION}
                            ${EXECUTABLE_CPPREGIONTEST}
                            ${EXECUTABLE_GTESTS}
                            ${EXECUTABLE_HELLOSPTP})
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
                            ${EXECUTABLE_GTESTS}
                            ${EXECUTABLE_HELLOSPTP}
                    VERBATIM)
endif()

#
# Install targets into CMAKE_INSTALL_PREFIX
#
install(TARGETS
        ${LIB_STATIC_NUPICCORE}
        ${LIB_STATIC_GTEST}
        ${EXECUTABLE_HELLOREGION}
        ${EXECUTABLE_CPPREGIONTEST}
        ${EXECUTABLE_PYREGIONTEST}
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
install(DIRECTORY ${CMAKE_BINARY_DIR}/nupic/ DESTINATION include/nupic
        FILES_MATCHING PATTERN "*.capnp.h")
install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/gtest"
        DESTINATION include/gtest
        FILES_MATCHING PATTERN "*.h*")
install(FILES "${PROJECT_BINARY_DIR}/Version.hpp"
        DESTINATION include/nupic/)
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/bin/"
        DESTINATION bin USE_SOURCE_PERMISSIONS)
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/lib/"
        DESTINATION lib)
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/bin/"
        DESTINATION bin USE_SOURCE_PERMISSIONS)
install(DIRECTORY "${REPOSITORY_DIR}/external/common/include/"
        DESTINATION include)
install(DIRECTORY "${REPOSITORY_DIR}/external/${PLATFORM}${BITNESS}/include/"
        DESTINATION include)
if(UNIX)
  install(FILES "${PROJECT_BINARY_DIR}/libnupic_core.a"
          DESTINATION lib)
else()
  install(FILES "${PROJECT_BINARY_DIR}/nupic_core.lib"
          DESTINATION lib)
endif()

#
# `make package` results in
# nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}.tar.gz binary release
#

set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_FILE_NAME "nupic_core-${NUPIC_CORE_VERSION}-${PLATFORM}${BITNESS}")
include(CPack)

