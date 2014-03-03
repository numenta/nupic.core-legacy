# --------------------------------------------------------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have purchased from
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
#   David Ragazzi (@DavidRagazzi): Full conversion and adaptation from Autotools scripts
# --------------------------------------------------------------------------------------------------------------------------
 

############################################################################################################################
###                                                                                                                      ###
###  Macros                                                                                                              ###
###                                                                                                                      ###
############################################################################################################################

# This macro set environment variables of the operating system
macro (show_environment_variable variable value)
  message(STATUS "  ${variable} = ${value}")
endmacro()

# This macro set environment variables of the operating system
macro (set_environment_variable variable value incremental)
  set(current_value "$ENV{${variable}}")
  set(new_value "${value}")

  # Check if variable already is set
  set(is_set OFF)
  string(REGEX MATCH "${new_value}" found_value "${current_value}")
  if("${found_value}" STREQUAL "${new_value}")
    set(is_set ON)
  endif()

  if(NOT is_set)
    # 'incremental' option means that new value wont replace current value, but increments it (ex: PATH variable)
    if("${incremental}" STREQUAL "ON")
      set(new_value "${new_value}:${current_value}")
    endif()

    if(LINUX)
      execute_process(COMMAND export ${variable}=${new_value})
    elseif(OSX)
      execute_process(COMMAND launchctl setenv ${variable} ${new_value})
    elseif(WINDOWS)
      execute_process(COMMAND setx ${variable} ${new_value})
    endif()
  endif()

  show_environment_variable(${variable} ${new_value})
endmacro()

# This macro get all files from a directory and its subdirectories.
macro (get_files_list current_dir dir_mask file_mask)
  file(GLOB children RELATIVE ${current_dir} ${current_dir}/*)
  foreach(child ${children})
    if(IS_DIRECTORY ${current_dir}/${child})
      get_files_list(${current_dir}/${child} ${dir_mask} ${file_mask})
    else()
      set(can_return_file ON)
      if(NOT "${dir_mask}" STREQUAL " ")
        if(NOT ${current_dir} MATCHES ${dir_mask})
          set(can_return_file OFF)
        endif()
      endif()
      if(NOT "${file_mask}" STREQUAL " ")
        if(NOT ${child} MATCHES ${file_mask})
          set(can_return_file OFF)
        endif()
      endif()
      if(can_return_file)
        set(FILES_RETURNED ${FILES_RETURNED} ${current_dir}/${child})
      endif()
    endif()
  endforeach()
endmacro()

# This macro generates a executable
macro(generate_executable name cxx_flags source_files link_flags libraries)
  add_executable(${name}
    ${source_files}
  )
  target_link_libraries(${name}
    ${libraries}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  set_target_properties(${name} PROPERTIES LINK_FLAGS "${link_flags}")
endmacro()

# This macro generates a static library
macro(generate_static_library name cxx_flags source_files)
  add_library(${name}
    ${source_files}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
endmacro()

# This macro generates a object library
macro(generate_object_library name cxx_flags source_files)
  add_library(${name}
    OBJECT ${source_files}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
endmacro()

# This macro generates a dynamic library
macro(generate_dynamic_library name cxx_flags source_files link_flags libraries)
  add_library(${name} SHARED
    ${source_files}
  )
  target_link_libraries(${name}
    ${libraries}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  set_target_properties(${name} PROPERTIES LINK_FLAGS "${link_flags}")
endmacro()


############################################################################################################################
###                                                                                                                      ###
###  1st Part:                                                                                                           ###
###  Configuration of the environment                                                                                    ###
###                                                                                                                      ###
############################################################################################################################

#
# Set C++ compiler.
# According to CMake documentation, this must be done before any language is set (ie before any project() or enable_language() command).
#
set(USER_CXX_COMPILER "" CACHE STRING "Default C++ compiler to be used (GCC / CLANG) [default=system compiler]")
if(NOT "${USER_CXX_COMPILER}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "${USER_CXX_COMPILER}")
endif()

#
# Project configuration
#
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0002 NEW)
project(nupic.core CXX)

#
# Sets default locations.
# 
# Default directories structure is:
#
# ~/../repository (root directory with repository downloaded from internet)
#         /docs (subdirectory with support documentation)
#         /src (subdirectory with all source code)
#         /build/scripts ([suggested] subdirectory with build process [makefiles or IDE solution] generated by CMake)
#         /build/release (subdirectory with executables and libraries generated by build proccess)
#         /build/temp (directory with temporary files generated by build proccess)
#         Readme.md
#         Licence.txt
#
# Repository root is the parent directory which this CMake file is located
# This CMake file must be always on /repository/src subdirectory.
#get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} PATH)
# TODO: This is palliative, it should be removed after repository restructuring
set(REPOSITORY_DIR ${PROJECT_SOURCE_DIR})
# 'release' is a subdirectory created on /build root.
if("${PROJECT_BUILD_RELEASE_DIR}" STREQUAL "")
  set(PROJECT_BUILD_RELEASE_DIR "${PROJECT_SOURCE_DIR}/build/release")
endif()
# 'temp' is a subdirectory created on /build root.
if("${PROJECT_BUILD_TEMP_DIR}" STREQUAL "")
  set(PROJECT_BUILD_TEMP_DIR "${PROJECT_SOURCE_DIR}/build/temp")
endif()
# CMAKE_CURRENT_BINARY_DIR is used by swig module for put wrappers on, so don't erase this line!
set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_BUILD_TEMP_DIR})

# TODO: This is palliative, it should be removed after repository restructuring
# Makes a copy to simulate /nta folder
execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/nta)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/nta)

#
# Clean directories before generate files
#
# Clean '/build/release'
execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_BUILD_RELEASE_DIR})
# Clean '/build/temp'
execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_BUILD_TEMP_DIR})

#
# Set OS flags
#
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OSX ON)
  set(STATIC_LIB_EXTENSION "a")
  set(DYNAMIC_LIB_EXTENSION "dylib")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINUX ON)
  set(STATIC_LIB_EXTENSION "a")
  set(DYNAMIC_LIB_EXTENSION "so")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(WINDOWS ON)
  set(STATIC_LIB_EXTENSION "lib")
  set(DYNAMIC_LIB_EXTENSION "dll")
endif()

#
# Remove libraries prefixes (ex: libsome_thing => some_thing)
#
set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

#
# Determine Nupic.Core platform type
#
message(STATUS "Checking platform...")
if(${CMAKE_SIZEOF_VOID_P} MATCHES "8")
  set(COMPILER_PLATFORM_ARCH "64")
else()
  set(COMPILER_PLATFORM_ARCH "32")
endif()
if(OSX)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.7")
  set(COMPILER_PLATFORM_OS "darwin64")
  set(COMPILER_PLATFORM_CXXFLAGS "-fPIC -DPIC -m64")
  set(COMPILER_PLATFORM_DEBUGFLAGS "-gfull")
  set(COMPILER_PLATFORM_LINKFLAGS "-Wl,-u,_munmap")
  set(COMPILER_PLATFORM_LIBS "-stdlib=libstdc++ -std=c++11 -lc++abi -liconv -lsqlite3 -framework CoreServices -framework Accelerate")
elseif(LINUX)
  set(COMPILER_PLATFORM_CXXFLAGS "-fPIC -DPIC")
  set(COMPILER_PLATFORM_DEBUGFLAGS "-g")
  set(COMPILER_PLATFORM_LINKFLAGS "-Wl,--no-as-needed -static-libgcc")
  set(COMPILER_PLATFORM_LIBS "-std=c++11 -lm -lpthread -ldl -lutil")
  if(${COMPILER_PLATFORM_ARCH} MATCHES "64")
    set(COMPILER_PLATFORM_OS "linux64")
    set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -m64")
  else()    
    set(COMPILER_PLATFORM_OS "linux32")
    set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -ffloat-store")
  endif()
elseif(WINDOWS)
  set(COMPILER_PLATFORM_OS "win32")
  set(COMPILER_PLATFORM_CXXFLAGS "-DWIN32")
  set(COMPILER_PLATFORM_DEBUGFLAGS "")
  set(COMPILER_PLATFORM_LINKFLAGS "")
  set(COMPILER_PLATFORM_LIBS "")
else()
  message(FATAL_ERROR "${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE}: error: ${CMAKE_SYSTEM_NAME} not supported yet.")
endif()

message(STATUS "Platform: ${COMPILER_PLATFORM_OS}")
message(STATUS "Platform-specific flags:")
message(STATUS "  Libraries: ${COMPILER_PLATFORM_LIBS}")
message(STATUS "  CXX flags: ${COMPILER_PLATFORM_CXXFLAGS}")
message(STATUS "  Debug flags: ${COMPILER_PLATFORM_DEBUGFLAGS}")
message(STATUS "  LD flags: ${COMPILER_PLATFORM_LINKFLAGS}")

#
# Shows components details.
#
message(STATUS "C++ details:")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")

#
# Config output paths.
# First config output paths for the generic no-config case.
# Second, for multi-config builds (DEBUG or RELEASE, for example).
#
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BUILD_RELEASE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BUILD_RELEASE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BUILD_RELEASE_DIR}/bin)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()

#
# Set environment variables of the operating system
#
message(STATUS "Checking environment...")
message(STATUS "Environment variables:")
set_environment_variable(PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" ON)
set_environment_variable(LD_LIBRARY_PATH "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}" ON)
set_environment_variable(DYLD_LIBRARY_PATH "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" ON)
if(OSX)
  set_environment_variable(MACOSX_DEPLOYMENT_TARGET "${CMAKE_OSX_DEPLOYMENT_TARGET}" OFF)
  message(STATUS "  (some variables will be updated only after login.)")
endif()

#
# Project details
#
message(STATUS "Checking project...")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Locations:")
message(STATUS "  Source: ${PROJECT_SOURCE_DIR}")
message(STATUS "  Build System: ${PROJECT_BINARY_DIR}")
message(STATUS "  Build Directory: ${PROJECT_BUILD_TEMP_DIR}")
message(STATUS "  Release: ${PROJECT_BUILD_RELEASE_DIR}")


############################################################################################################################
###                                                                                                                      ###
###  2nd Part:                                                                                                           ###
###  Configuration of compiler flags                                                                                     ###
###                                                                                                                      ###
############################################################################################################################

#
# Include directories of headers
# Allows us to find includes for external libraries and enables
# #include <common/...>
#
set(COMPILER_INCLUDEFLAGS "-I${REPOSITORY_DIR} -I${REPOSITORY_DIR}/external/common/include -I${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/include")

#
# NTA_INTERNAL tells us that the code is being built under the build system
# and not as a separate program. Used for cppvision example.
#
set(COMPILER_CXXFLAGS_BASE "${COMPILER_INCLUDEFLAGS} ${COMPILER_PLATFORM_CXXFLAGS} -DHAVE_CONFIG_H -DNTA_INTERNAL -DNTA_PLATFORM_${COMPILER_PLATFORM_OS} -DBOOST_NO_WREGEX -DNUPIC2 -fvisibility=hidden -Wall -Wreturn-type -Wunused-variable -Wno-deprecated")

#
# All executables and plugins are linked with these flags
#
set(COMPILER_LINKFLAGS "${COMPILER_PLATFORM_LINKFLAGS} ${COMPILER_PLATFORM_LIBS} -L${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib")


message(STATUS "Options:")

#
# Optimization can be set at configure time
#
option(COMPILER_OPTIMIZATION_ENABLED "--optimization=[ON/OFF] turn on optimization [default=ON]" ON)
if(COMPILER_OPTIMIZATION_ENABLED)
  # -ftree-vectorize conflicts with -no-tree-aliasing on leopard, so removed
  # flag NTA_ASM enables/disables our hand tuned assbembly code (SSE), but it's available only for some platform
  set(COMPILER_CXXFLAGS_OPTIMIZATION "-O3 -pipe -DNTA_ASM")
else()
  set(COMPILER_CXXFLAGS_OPTIMIZATION "-O0 -fno-inline")
endif()
message(STATUS "  Optimization: ${COMPILER_OPTIMIZATION_ENABLED}")

#
# Enable/disable debugging symbols (-g). Enabled by default (stripped for release).
# This is orthogonal to optimization
#
option(COMPILER_DEBUGSYMBOLS_ENABLED "--debugsymbols=[ON/OFF] turn on debug symbols [default=ON]" ON)
if(COMPILER_DEBUGSYMBOLS_ENABLED)
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} ${COMPILER_PLATFORM_DEBUGFLAGS}")
endif()
message(STATUS "  Debug Symbols: ${COMPILER_DEBUGSYMBOLS_ENABLED}")

#
# Enable/disable NTA_ASSERT.
#
option(COMPILER_ASSERTIONS_ENABLED "--assertions=[ON/OFF] turn on assertions [default=ON]" ON)
if(COMPILER_ASSERTIONS_ENABLED)
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -DNTA_ASSERTIONS_ON")
else()
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -DNDEBUG")
endif()
message(STATUS "  Assertions: ${COMPILER_ASSERTIONS_ENABLED}")

#
# Enable/disable gprof profiling (this option is not frequently used/tested).
#
option(COMPILER_PROFILING_ENABLED "--profiling=[ON/OFF] turn on profiling [default=OFF]" OFF)
if(COMPILER_PROFILING_ENABLED)
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -pg")
  set(COMPILER_LINKFLAGS "${COMPILER_LINKFLAGS} -pg")
endif()
message(STATUS "  Profiling: ${COMPILER_PROFILING_ENABLED}")

#
# Build using MPI headers and library,
# and enable features that depend on MPI.
#
option(COMPILER_MPI_SUPPORT "--mpi=[ON/OFF] turn on MPI support [default=ON]" ON)
message(STATUS "  MPI Support: ${COMPILER_MPI_SUPPORT}")

#
# Set GNU c/c++ flags in case of CLang is off.
#
if(CMAKE_COMPILER_IS_GNUCXX)
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -Wno-sign-compare")
  set(COMPILER_CXXFLAGS_OPTIMIZATION "${COMPILER_CXXFLAGS_OPTIMIZATION} -falign-loops=16")
endif()

#
# COMPILER_CXXFLAGS are used everywhere.
#
set(COMPILER_CXXFLAGS "${COMPILER_CXXFLAGS_BASE} ${COMPILER_CXXFLAGS_OPTIMIZATION}")


############################################################################################################################
###                                                                                                                      ###
###  3rd Part:                                                                                                           ###
###  Generation of the project and its modules                                                                           ###
###                                                                                                                      ###
############################################################################################################################

#
# Set current source dir to source root
#
set(CURRENT_SOURCE_DIR ${PROJECT_SOURCE_DIR})

#
# Apache's external libraries.
#
set(LIB_STATIC_APR1 apr-1)
add_library(${LIB_STATIC_APR1} STATIC IMPORTED)
set_property(TARGET ${LIB_STATIC_APR1} PROPERTY IMPORTED_LOCATION "${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib/libapr-1.${STATIC_LIB_EXTENSION}")

set(LIB_STATIC_APRUTIL1 aprutil-1)
add_library(${LIB_STATIC_APRUTIL1} STATIC IMPORTED)
set_property(TARGET ${LIB_STATIC_APRUTIL1} PROPERTY IMPORTED_LOCATION "${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib/libaprutil-1.${STATIC_LIB_EXTENSION}")

set(LIB_STATIC_Z z)
add_library(${LIB_STATIC_Z} STATIC IMPORTED)
set_property(TARGET ${LIB_STATIC_Z} PROPERTY IMPORTED_LOCATION "${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib/libz.${STATIC_LIB_EXTENSION}")

set(LIB_STATIC_YAML yaml)
add_library(${LIB_STATIC_YAML} STATIC IMPORTED)
set_property(TARGET ${LIB_STATIC_YAML} PROPERTY IMPORTED_LOCATION "${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib/libyaml.${STATIC_LIB_EXTENSION}")

if(WINDOWS)
  set(LIB_STATIC_PCRE pcre)
  add_library(${LIB_STATIC_PCRE} STATIC IMPORTED)
  set_property(TARGET ${LIB_STATIC_PCRE} PROPERTY IMPORTED_LOCATION "${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/lib/libpcre.${STATIC_LIB_EXTENSION}")
endif()

#
# Set list of external libraries to this project.
#
set(COMPILER_STANDARD_LIBS 
  ${LIB_STATIC_YAML}
  ${LIB_STATIC_APRUTIL1}
  ${LIB_STATIC_APR1}
  ${LIB_STATIC_Z}
)

#
# LibAlgorithms
#
set(LIB_STATIC_ALGORITHMS algorithms)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/algorithms")
set(LIB_STATIC_ALGORITHMS_SOURCE
  ${CURRENT_SOURCE_DIR}/bit_history.cpp
  ${CURRENT_SOURCE_DIR}/Cell.cpp
  ${CURRENT_SOURCE_DIR}/Cells4.cpp
  ${CURRENT_SOURCE_DIR}/classifier_result.cpp
  ${CURRENT_SOURCE_DIR}/CondProbTable.cpp
  ${CURRENT_SOURCE_DIR}/fast_cla_classifier.cpp
  ${CURRENT_SOURCE_DIR}/flat_spatial_pooler.cpp
  ${CURRENT_SOURCE_DIR}/gaborNode.cpp
  ${CURRENT_SOURCE_DIR}/imageSensorLite.cpp
  ${CURRENT_SOURCE_DIR}/InSynapse.cpp
  ${CURRENT_SOURCE_DIR}/linear.cpp
  ${CURRENT_SOURCE_DIR}/OutSynapse.cpp
  ${CURRENT_SOURCE_DIR}/Segment.cpp
  ${CURRENT_SOURCE_DIR}/SegmentUpdate.cpp
  ${CURRENT_SOURCE_DIR}/spatial_pooler.cpp
  ${CURRENT_SOURCE_DIR}/svm.cpp
  ${CURRENT_SOURCE_DIR}/arrayBuffer.hpp
  ${CURRENT_SOURCE_DIR}/bit_history.hpp
  ${CURRENT_SOURCE_DIR}/Cell.hpp
  ${CURRENT_SOURCE_DIR}/Cells4.hpp
  ${CURRENT_SOURCE_DIR}/classifier_result.hpp
  ${CURRENT_SOURCE_DIR}/CondProbTable.hpp
  ${CURRENT_SOURCE_DIR}/fast_cla_classifier.hpp
  ${CURRENT_SOURCE_DIR}/FDRCSpatial.hpp
  ${CURRENT_SOURCE_DIR}/FDRSpatial.hpp
  ${CURRENT_SOURCE_DIR}/flat_spatial_pooler.hpp
  ${CURRENT_SOURCE_DIR}/gaborNode.hpp
  ${CURRENT_SOURCE_DIR}/imageSensorLite.hpp
  ${CURRENT_SOURCE_DIR}/InSynapse.hpp
  ${CURRENT_SOURCE_DIR}/linear.hpp
  ${CURRENT_SOURCE_DIR}/OutSynapse.hpp
  ${CURRENT_SOURCE_DIR}/scanning.hpp
  ${CURRENT_SOURCE_DIR}/Segment.hpp
  ${CURRENT_SOURCE_DIR}/SegmentUpdate.hpp
  ${CURRENT_SOURCE_DIR}/spatial_pooler.hpp
  ${CURRENT_SOURCE_DIR}/svm.hpp
  ${CURRENT_SOURCE_DIR}/svm_t.hpp
)
generate_object_library(${LIB_STATIC_ALGORITHMS} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_ALGORITHMS_SOURCE}")

#
# LibMath
#
set(LIB_STATIC_MATH math)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/math")
set(LIB_STATIC_MATH_SOURCE
  ${CURRENT_SOURCE_DIR}/SparseMatrixAlgorithms.cpp
  ${CURRENT_SOURCE_DIR}/stl_io.cpp
  ${CURRENT_SOURCE_DIR}/array2D.hpp
  ${CURRENT_SOURCE_DIR}/array_algo.hpp
  ${CURRENT_SOURCE_DIR}/convolution.hpp
  ${CURRENT_SOURCE_DIR}/DenseMatrix.hpp
  ${CURRENT_SOURCE_DIR}/Domain.hpp
  ${CURRENT_SOURCE_DIR}/erosion.hpp
  ${CURRENT_SOURCE_DIR}/functions.hpp
  ${CURRENT_SOURCE_DIR}/GraphAlgorithms.hpp
  ${CURRENT_SOURCE_DIR}/Index.hpp
  ${CURRENT_SOURCE_DIR}/math.hpp
  ${CURRENT_SOURCE_DIR}/NearestNeighbor.hpp
  ${CURRENT_SOURCE_DIR}/rotation.hpp
  ${CURRENT_SOURCE_DIR}/Set.hpp
  ${CURRENT_SOURCE_DIR}/SparseBinaryMatrix.hpp
  ${CURRENT_SOURCE_DIR}/SparseMatrix.hpp
  ${CURRENT_SOURCE_DIR}/SparseMatrix01.hpp
  ${CURRENT_SOURCE_DIR}/SparseMatrixAlgorithms.hpp
  ${CURRENT_SOURCE_DIR}/SparseRLEMatrix.hpp
  ${CURRENT_SOURCE_DIR}/SparseTensor.hpp
  ${CURRENT_SOURCE_DIR}/stl_io.hpp
  ${CURRENT_SOURCE_DIR}/types.hpp
  ${CURRENT_SOURCE_DIR}/utils.hpp
)
generate_object_library(${LIB_STATIC_MATH} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_MATH_SOURCE}")

#
# LibTypes
#
set(LIB_STATIC_TYPES types)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/types")
set(LIB_STATIC_TYPES_SOURCE
  ${CURRENT_SOURCE_DIR}/BasicType.cpp
  ${CURRENT_SOURCE_DIR}/Fraction.cpp
  ${CURRENT_SOURCE_DIR}/BasicType.hpp
  ${CURRENT_SOURCE_DIR}/Exception.hpp
  ${CURRENT_SOURCE_DIR}/Fraction.hpp
  ${CURRENT_SOURCE_DIR}/types.hpp
  ${CURRENT_SOURCE_DIR}/types.h
)
generate_object_library(${LIB_STATIC_TYPES} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_TYPES_SOURCE}")

#
# LibTest
#
set(LIB_STATIC_TEST test)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/test")
set(LIB_STATIC_TEST_SOURCE
  ${CURRENT_SOURCE_DIR}/Tester.cpp
  ${CURRENT_SOURCE_DIR}/Tester.hpp
)
generate_object_library(${LIB_STATIC_TEST} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_TEST_SOURCE}")

#
# LibUtils
#
set(LIB_STATIC_UTILS utils)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/utils")
set(LIB_STATIC_UTILS_SOURCE
  ${CURRENT_SOURCE_DIR}/LoggingException.cpp
  ${CURRENT_SOURCE_DIR}/LogItem.cpp
  ${CURRENT_SOURCE_DIR}/Random.cpp
  ${CURRENT_SOURCE_DIR}/StringUtils.cpp
  ${CURRENT_SOURCE_DIR}/TRandom.cpp
  ${CURRENT_SOURCE_DIR}/Watcher.cpp
  ${CURRENT_SOURCE_DIR}/Log.hpp
  ${CURRENT_SOURCE_DIR}/LoggingException.hpp
  ${CURRENT_SOURCE_DIR}/LogItem.hpp
  ${CURRENT_SOURCE_DIR}/Random.hpp
  ${CURRENT_SOURCE_DIR}/StringUtils.hpp
  ${CURRENT_SOURCE_DIR}/TRandom.hpp
  ${CURRENT_SOURCE_DIR}/utils.hpp
  ${CURRENT_SOURCE_DIR}/Watcher.hpp
)
generate_object_library(${LIB_STATIC_UTILS} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_UTILS_SOURCE}")

#
# LibNTypes
#
set(LIB_STATIC_NTYPES ntypes)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ntypes")
set(LIB_STATIC_NTYPES_SOURCE
  ${CURRENT_SOURCE_DIR}/ArrayBase.cpp
  ${CURRENT_SOURCE_DIR}/Buffer.cpp
  ${CURRENT_SOURCE_DIR}/BundleIO.cpp
  ${CURRENT_SOURCE_DIR}/Collection.cpp
  ${CURRENT_SOURCE_DIR}/Dimensions.cpp
  ${CURRENT_SOURCE_DIR}/MemParser.cpp
  ${CURRENT_SOURCE_DIR}/Scalar.cpp
  ${CURRENT_SOURCE_DIR}/Value.cpp
  ${CURRENT_SOURCE_DIR}/Array.hpp
  ${CURRENT_SOURCE_DIR}/ArrayBase.hpp
  ${CURRENT_SOURCE_DIR}/ArrayRef.hpp
  ${CURRENT_SOURCE_DIR}/Buffer.hpp
  ${CURRENT_SOURCE_DIR}/BundleIO.hpp
  ${CURRENT_SOURCE_DIR}/Collection.hpp
  ${CURRENT_SOURCE_DIR}/Dimensions.hpp
  ${CURRENT_SOURCE_DIR}/MemParser.hpp
  ${CURRENT_SOURCE_DIR}/MemStream.hpp
  ${CURRENT_SOURCE_DIR}/NodeSet.hpp
  ${CURRENT_SOURCE_DIR}/object_model.hpp
  ${CURRENT_SOURCE_DIR}/Scalar.hpp
  ${CURRENT_SOURCE_DIR}/Value.hpp
  ${CURRENT_SOURCE_DIR}/object_model.h
)
generate_object_library(${LIB_STATIC_NTYPES} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_NTYPES_SOURCE}")

#
# LibEngine
#
set(LIB_STATIC_ENGINE engine)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/engine")
set(LIB_STATIC_ENGINE_SOURCE
  ${CURRENT_SOURCE_DIR}/Collections.cpp
  ${CURRENT_SOURCE_DIR}/Input.cpp
  ${CURRENT_SOURCE_DIR}/Link.cpp
  ${CURRENT_SOURCE_DIR}/LinkPolicyFactory.cpp
  ${CURRENT_SOURCE_DIR}/Network.cpp
  ${CURRENT_SOURCE_DIR}/NuPIC.cpp
  ${CURRENT_SOURCE_DIR}/Output.cpp
  ${CURRENT_SOURCE_DIR}/Region.cpp
  ${CURRENT_SOURCE_DIR}/Region_io.cpp
  ${CURRENT_SOURCE_DIR}/Region_parameters.cpp
  ${CURRENT_SOURCE_DIR}/RegionImpl.cpp
  ${CURRENT_SOURCE_DIR}/RegionImplFactory.cpp
  ${CURRENT_SOURCE_DIR}/Spec.cpp
  ${CURRENT_SOURCE_DIR}/TestFanIn2LinkPolicy.cpp
  ${CURRENT_SOURCE_DIR}/TestNode.cpp
  ${CURRENT_SOURCE_DIR}/UniformLinkPolicy.cpp
  ${CURRENT_SOURCE_DIR}/YAMLUtils.cpp
  ${CURRENT_SOURCE_DIR}/Input.hpp
  ${CURRENT_SOURCE_DIR}/Link.hpp
  ${CURRENT_SOURCE_DIR}/LinkPolicy.hpp
  ${CURRENT_SOURCE_DIR}/LinkPolicyFactory.hpp
  ${CURRENT_SOURCE_DIR}/Network.hpp
  ${CURRENT_SOURCE_DIR}/NuPIC.hpp
  ${CURRENT_SOURCE_DIR}/Output.hpp
  ${CURRENT_SOURCE_DIR}/Region.hpp
  ${CURRENT_SOURCE_DIR}/RegionImpl.hpp
  ${CURRENT_SOURCE_DIR}/RegionImplFactory.hpp
  ${CURRENT_SOURCE_DIR}/Spec.hpp
  ${CURRENT_SOURCE_DIR}/TestFanIn2LinkPolicy.hpp
  ${CURRENT_SOURCE_DIR}/TestNode.hpp
  ${CURRENT_SOURCE_DIR}/UniformLinkPolicy.hpp
  ${CURRENT_SOURCE_DIR}/YAMLUtils.hpp
)
generate_object_library(${LIB_STATIC_ENGINE} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_ENGINE_SOURCE}")

#
# LibOS
#
set(LIB_STATIC_OS os)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/os")
set(LIB_STATIC_OS_SOURCE
  ${CURRENT_SOURCE_DIR}/Directory.cpp
  ${CURRENT_SOURCE_DIR}/DynamicLibrary.cpp
  ${CURRENT_SOURCE_DIR}/Env.cpp
  ${CURRENT_SOURCE_DIR}/FStream.cpp
  ${CURRENT_SOURCE_DIR}/OS.cpp
  ${CURRENT_SOURCE_DIR}/OSUnix.cpp
  ${CURRENT_SOURCE_DIR}/OSWin.cpp
  ${CURRENT_SOURCE_DIR}/Path.cpp
  ${CURRENT_SOURCE_DIR}/regex.cpp
  ${CURRENT_SOURCE_DIR}/Timer.cpp
  ${CURRENT_SOURCE_DIR}/Directory.hpp
  ${CURRENT_SOURCE_DIR}/DynamicLibrary.hpp
  ${CURRENT_SOURCE_DIR}/Env.hpp
  ${CURRENT_SOURCE_DIR}/FStream.hpp
  ${CURRENT_SOURCE_DIR}/OS.hpp
  ${CURRENT_SOURCE_DIR}/Path.hpp
  ${CURRENT_SOURCE_DIR}/regex.hpp
  ${CURRENT_SOURCE_DIR}/Timer.hpp
)
generate_object_library(${LIB_STATIC_OS} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_OS_SOURCE}")

#
# LibRegions
#
set(LIB_STATIC_REGIONS regions)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/regions")
set(LIB_STATIC_REGIONS_SOURCE
  ${CURRENT_SOURCE_DIR}/VectorFile.cpp
  ${CURRENT_SOURCE_DIR}/VectorFileEffector.cpp
  ${CURRENT_SOURCE_DIR}/VectorFileSensor.cpp
  ${CURRENT_SOURCE_DIR}/VectorFile.hpp
  ${CURRENT_SOURCE_DIR}/VectorFileEffector.hpp
  ${CURRENT_SOURCE_DIR}/VectorFileSensor.hpp
)
generate_object_library(${LIB_STATIC_REGIONS} "${COMPILER_CXXFLAGS}" "${LIB_STATIC_REGIONS_SOURCE}")

#
# LibNupicCore
#
# This library is installed in <build/release/lib> and
# loaded dynamically or statically.
#
set(LIB_NUPICCORE nupic.core)
set(CURRENT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(SOURCE
  ${LIB_STATIC_ENGINE_SOURCE}
  ${LIB_STATIC_REGIONS_SOURCE}
  ${LIB_STATIC_ALGORITHMS_SOURCE}
  ${LIB_STATIC_MATH_SOURCE}
  ${LIB_STATIC_NTYPES_SOURCE}
  ${LIB_STATIC_SUPPORT_SOURCE}
  ${LIB_STATIC_OS_SOURCE}
  ${LIB_STATIC_UTILS_SOURCE}
  ${LIB_STATIC_TEST_SOURCE}
  ${LIB_STATIC_TYPES_SOURCE}
)
set(LIBRARIES
  ${COMPILER_STANDARD_LIBS}
)
generate_static_library(${LIB_NUPICCORE} "${COMPILER_CXXFLAGS}" "${SOURCE}")
# TODO: Generate a dynamic library too: generate_dynamic_library(${LIB_NUPICCORE} "${COMPILER_CXXFLAGS}" "${SOURCE}" "${COMPILER_LINKFLAGS}" "${LIBRARIES}")


############################################################################################################################
###                                                                                                                      ###
###  4th Part:                                                                                                           ###
###  Post configuration operations                                                                                       ###
###                                                                                                                      ###
############################################################################################################################


# Just to separate from result message
message(STATUS "")
