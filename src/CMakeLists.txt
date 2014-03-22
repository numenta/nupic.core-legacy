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

# These macros copy all source directories after the configuration is done
macro (copy suffix src dst)  
  message(STATUS "Copying from '${src}' to '${dst}'")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy${suffix} ${src} ${dst})
endmacro()
macro (copy_file src dst)
  copy("" ${src} ${dst})
endmacro()
macro (copy_directory src dst)
  copy("_directory" ${src} ${dst})
endmacro()

# This macro get all subdirectories from a directory.
macro (get_directories_list current_dir dir_mask)
  file(GLOB children RELATIVE ${current_dir} ${current_dir}/*)
  foreach(child ${children})
    if(IS_DIRECTORY ${current_dir}/${child})
      get_directories_list(${current_dir}/${child} ${dir_mask})
      set(can_return_item ON)
      if(NOT "${dir_mask}" STREQUAL " ")
        if(NOT ${current_dir} MATCHES ${dir_mask})
          set(can_return_item OFF)
        endif()
      endif()
      if(can_return_item)
        set(ITEMS_RETURNED ${ITEMS_RETURNED} ${current_dir}/${child})
      endif()
    endif()
  endforeach()
endmacro()

# This macro get all files from a directory and its subdirectories.
macro (get_files_list current_dir dir_mask file_mask)
  file(GLOB children RELATIVE ${current_dir} ${current_dir}/*)
  foreach(child ${children})
    if(IS_DIRECTORY ${current_dir}/${child})
      get_files_list(${current_dir}/${child} ${dir_mask} ${file_mask})
    else()
      set(can_return_item ON)
      if(NOT "${dir_mask}" STREQUAL " ")
        if(NOT ${current_dir} MATCHES ${dir_mask})
          set(can_return_item OFF)
        endif()
      endif()
      if(NOT "${file_mask}" STREQUAL " ")
        if(NOT ${child} MATCHES ${file_mask})
          set(can_return_item OFF)
        endif()
      endif()
      if(can_return_item)
        set(ITEMS_RETURNED ${ITEMS_RETURNED} ${current_dir}/${child})
      endif()
    endif()
  endforeach()
endmacro()

# This macro get all source files from a directory and its subdirectories.
macro(get_source_groups src_dir)
  set(SOURCE_RETURNED)
  set(ITEMS_RETURNED)

  # Add root files as a blank source group
  file(GLOB src_files
    ${src_dir}/*.cpp
    ${src_dir}/*.hpp
    ${src_dir}/*.h
  )
  if(src_files)
    set(SOURCE_RETURNED ${SOURCE_RETURNED} ${src_files})
    source_group("" FILES ${src_files})
  endif()

  # Add every subdir as a source group
  get_directories_list(${src_dir} " ")
  set(src_sub_dirs ${ITEMS_RETURNED})
  if(src_sub_dirs)
    foreach(dir ${src_sub_dirs})
      file(GLOB src_files
        ${dir}/*.cpp
        ${dir}/*.hpp
        ${dir}/*.h
      )
      get_filename_component(src_group ${dir} NAME)
      set(SOURCE_RETURNED ${SOURCE_RETURNED} ${src_files})
      source_group(${src_group} FILES ${src_files})
    endforeach()
  endif()
endmacro()

# This macro generates a executable
macro(generate_executable name src_dir output_dir cxx_flags link_flags libraries)
  get_source_groups(${src_dir})
  add_executable(${name}
    ${SOURCE_RETURNED}
  )
  target_link_libraries(${name}
    ${libraries}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  set_target_properties(${name} PROPERTIES LINK_FLAGS "${link_flags}")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_dir})
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
  endforeach()
endmacro()

# This macro generates a static library
macro(generate_static_library name src_dir output_dir cxx_flags)
  get_source_groups(${src_dir})
  add_library(${name}
    ${SOURCE_RETURNED}
  )
  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  set_target_properties(${name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${output_dir})
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set_target_properties(${name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
  endforeach()
endmacro()

# This macro generates a dynamic library
macro(generate_dynamic_library name src_dir output_dir cxx_flags link_flags project_libraries external_libraries)
  get_source_groups(${src_dir})
  add_library(${name} SHARED
    ${SOURCE_RETURNED}
  )

  # All project libraries should included as whole into dynamic library
  foreach(library ${project_libraries})
    target_link_libraries(${name}
      ${COMPILER_LINKFLAGS_ARCHIVES} ${library}
    )
  endforeach()
  target_link_libraries(${name}
    ${external_libraries}
  )

  set_target_properties(${name} PROPERTIES COMPILE_FLAGS "${cxx_flags}")
  set_target_properties(${name} PROPERTIES LINKER_LANGUAGE "CXX")
  set_target_properties(${name} PROPERTIES LINK_FLAGS "${link_flags}")
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${output_dir})
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
  endforeach()
endmacro()


############################################################################################################################
###                                                                                                                      ###
###  1st Part:                                                                                                           ###
###  Configuration of the environment                                                                                    ###
###                                                                                                                      ###
############################################################################################################################

#
# Project configuration
#
cmake_minimum_required(VERSION 2.8)
project(nupic.core CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#
# Set C++ compiler.
# According to CMake documentation, this must be done before any language is set (ie before any project() or enable_language() command).
#
set(USER_CXX_COMPILER "" CACHE STRING "Default C++ compiler to be used (GCC / CLANG) [default=system compiler]")
if(NOT "${USER_CXX_COMPILER}" STREQUAL "")
  set(CMAKE_CXX_COMPILER "${USER_CXX_COMPILER}")
endif()

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
get_filename_component(REPOSITORY_DIR ${PROJECT_SOURCE_DIR} PATH)
# 'release' is a subdirectory created on /build root.
if("${PROJECT_BUILD_RELEASE_DIR}" STREQUAL "")
  set(PROJECT_BUILD_RELEASE_DIR "${REPOSITORY_DIR}/build/release")
endif()
# 'temp' is a subdirectory created on /build root.
if("${PROJECT_BUILD_TEMP_DIR}" STREQUAL "")
  set(PROJECT_BUILD_TEMP_DIR "${REPOSITORY_DIR}/build/temp")
endif()
# CMAKE_CURRENT_BINARY_DIR is used by swig module for put wrappers on, so don't erase this line!
set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_BUILD_TEMP_DIR})

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
#set(CMAKE_STATIC_LIBRARY_PREFIX "")
#set(CMAKE_SHARED_LIBRARY_PREFIX "")

#
# Determine Nupic.Core platform type
#
message(STATUS "Checking platform...")
if(${CMAKE_SIZEOF_VOID_P} MATCHES "8")
  set(COMPILER_PLATFORM_ARCH "64")
else()
  set(COMPILER_PLATFORM_ARCH "32")
endif()
set(COMPILER_PLATFORM_CXXFLAGS "-fPIC -DPIC")
set(COMPILER_PLATFORM_DEBUGFLAGS "-gfull")
set(COMPILER_PLATFORM_LINKFLAGS "")
set(COMPILER_PLATFORM_LIBS "-stdlib=libstdc++ -std=c++98 -lc++abi")
if(OSX)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "10.7")
  set(COMPILER_PLATFORM_OS "darwin64")
  set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -m64")
  set(COMPILER_PLATFORM_LINKFLAGS "${COMPILER_PLATFORM_LINKFLAGS} -Wl,-u,_munmap")
  set(COMPILER_PLATFORM_LIBS "${COMPILER_PLATFORM_LIBS} -liconv -lsqlite3 -framework CoreServices -framework Accelerate")
elseif(LINUX)
  set(COMPILER_PLATFORM_LINKFLAGS "${COMPILER_PLATFORM_LINKFLAGS} -Wl,--no-as-needed -static-libgcc")
  set(COMPILER_PLATFORM_LIBS "${COMPILER_PLATFORM_LIBS} -lm -lpthread -ldl -lutil")
  if(${COMPILER_PLATFORM_ARCH} MATCHES "64")
    set(COMPILER_PLATFORM_OS "linux64")
    set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -m64")
  else()    
    set(COMPILER_PLATFORM_OS "linux32")
    set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -ffloat-store")
  endif()
elseif(WINDOWS)
  set(COMPILER_PLATFORM_OS "win32")
  set(COMPILER_PLATFORM_CXXFLAGS "${COMPILER_PLATFORM_CXXFLAGS} -DWIN32")
else()
  message(FATAL_ERROR "${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE}: error: ${CMAKE_SYSTEM_NAME} not supported yet.")
endif()

message(STATUS "Platform: ${COMPILER_PLATFORM_OS}")
message(STATUS "Platform-specific flags:")
message(STATUS "  Libraries: ${COMPILER_PLATFORM_LIBS}")
message(STATUS "  CXX flags: ${COMPILER_PLATFORM_CXXFLAGS}")
message(STATUS "  Debug flags: ${COMPILER_PLATFORM_DEBUGFLAGS}")
message(STATUS "  LD flags: ${COMPILER_PLATFORM_LINKFLAGS}")

message(STATUS "Checking software tools...")

#
# Gives the version of C++ compiler
#
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)

#
# Shows components details.
#
message(STATUS "C++ details:")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Version: ${CXX_VERSION}")

#
# Set environment variables of the operating system
#
message(STATUS "Checking environment...")
message(STATUS "Environment variables:")
set_environment_variable(PATH "${PROJECT_BUILD_RELEASE_DIR}/bin" ON)
set_environment_variable(LD_LIBRARY_PATH "${PROJECT_BUILD_RELEASE_DIR}/lib" ON)
set_environment_variable(DYLD_LIBRARY_PATH "${PROJECT_BUILD_RELEASE_DIR}/lib" ON)
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
# Define custom preprocessor variables (#define XX in C++ code)
# The reason to use add_definitions() instead of appending flag -std=c++0x directly to CXXBASE_FLAGS is that
# cmake deploys "internal smarts" for compatibility with different compilers (clang, gcc 4.6 vs 4.8)
add_definitions(-std=c++98) # smart flag setting for gcc/clang

#
# Include directories of headers
# Allows us to find includes for external libraries and enables
# #include <common/...>
#
set(COMPILER_INCLUDEFLAGS "-I${REPOSITORY_DIR} -I${PROJECT_BUILD_RELEASE_DIR}/include -isystem${REPOSITORY_DIR}/external/common/include -isystem${REPOSITORY_DIR}/external/${COMPILER_PLATFORM_OS}/include")

#
# NTA_INTERNAL tells us that the code is being built under the build system
# and not as a separate program. Used for cppvision example.
#
set(COMPILER_CXXFLAGS_BASE "${COMPILER_INCLUDEFLAGS} ${COMPILER_PLATFORM_CXXFLAGS} -DHAVE_CONFIG_H -DNTA_INTERNAL -DNTA_PLATFORM_${COMPILER_PLATFORM_OS} -DBOOST_NO_WREGEX -DNUPIC2 -fvisibility=hidden -Wall -Wreturn-type -Wunused -Wno-unused-parameter")

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
# Compiler specific (clang, gcc, ..) flags can be set here.
#
if(CMAKE_COMPILER_IS_GNUCXX)
  set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -Wno-sign-compare -Wno-error=format=")
  set(COMPILER_CXXFLAGS_OPTIMIZATION "${COMPILER_CXXFLAGS_OPTIMIZATION} -falign-loops=16")
  # Workaround for gcc 4.4 in Grok
  if (CXX_VERSION VERSION_GREATER 4.6 OR CXX_VERSION VERSION_EQUAL 4.6)
    set(COMPILER_CXXFLAGS_BASE "${COMPILER_CXXFLAGS_BASE} -Wno-error=cpp -Werror")
  endif()
  set(COMPILER_LINKFLAGS_ARCHIVES "-Wl,-whole-archive")
else()
  set(COMPILER_LINKFLAGS_ARCHIVES "-Wl,-force_load")
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

message(STATUS "Configuring subprojects...")

#
# LibNupicCore
#
set(LIB_STATIC_NUPICCORE nupic.core)
generate_static_library(${LIB_STATIC_NUPICCORE} "${PROJECT_SOURCE_DIR}/main" "${PROJECT_BUILD_RELEASE_DIR}/lib" "${COMPILER_CXXFLAGS}")

#
# HtmTest
#
set(EXECUTABLE_HTMTEST htmtest)
set(LIBRARIES
  ${LIB_STATIC_NUPICCORE}
  ${COMPILER_STANDARD_LIBS}
)
generate_executable(${EXECUTABLE_HTMTEST} "${PROJECT_SOURCE_DIR}/test/htmtest" "${PROJECT_BUILD_RELEASE_DIR}/bin" "${COMPILER_CXXFLAGS}" "${COMPILER_LINKFLAGS}" "${LIBRARIES}")

#
# TestEverything
#
set(EXECUTABLE_TESTEVERYTHING testeverything)
set(LIBRARIES
  ${LIB_STATIC_NUPICCORE}
  ${COMPILER_STANDARD_LIBS}
)
generate_executable(${EXECUTABLE_TESTEVERYTHING} "${PROJECT_SOURCE_DIR}/test/testeverything" "${PROJECT_BUILD_RELEASE_DIR}/bin" "${COMPILER_CXXFLAGS}" "${COMPILER_LINKFLAGS} ${NTA_LINKFLAGS_PYEMBED}" "${LIBRARIES}")


############################################################################################################################
###                                                                                                                      ###
###  4th Part:                                                                                                           ###
###  Post configuration operations                                                                                       ###
###                                                                                                                      ###
############################################################################################################################

#
# Copy all headers files to 'build/release/include'
#
set(ITEMS_RETURNED)
get_files_list(${PROJECT_SOURCE_DIR}/main " " ".h")
set(files ${ITEMS_RETURNED})
set(headers_file_content "")
foreach(src_file ${files})
  get_filename_component(src_dir ${src_file} ABSOLUTE)
  string(REPLACE "${PROJECT_SOURCE_DIR}/main" "${PROJECT_BUILD_RELEASE_DIR}/include/nta" dst_file ${src_file})
  copy_file(${src_file} ${dst_file})
endforeach()
copy_file(${PROJECT_SOURCE_DIR}/main/ntypes/Collection.cpp ${PROJECT_BUILD_RELEASE_DIR}/include/nta/ntypes/Collection.cpp)

#
# Create file with unit tests used by TestEverything
#
set(ITEMS_RETURNED)
get_files_list(${PROJECT_SOURCE_DIR}/test/testeverything " " "Test.hpp")
set(files ${ITEMS_RETURNED})
set(tests_file_content "")
set(headers_file_content "")
foreach(file ${files})
  get_filename_component(test ${file} NAME_WE)
  if(NOT ${file} MATCHES "TesterTest")
    set(tests_file_content "${tests_file_content}ADD_TEST(${test});\n")
    set(headers_file_content "${headers_file_content}#include \"${file}\"\n")
  endif()
endforeach()
file(WRITE "${PROJECT_SOURCE_DIR}/test/testeverything/everything_addtests.hpp" "${tests_file_content}")
file(WRITE "${PROJECT_SOURCE_DIR}/test/testeverything/everything_headers.hpp" "${headers_file_content}")


# Just to separate from result message
message(STATUS "")
