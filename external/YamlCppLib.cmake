# -----------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

# Creates ExternalProject for building the yaml-cpp static library
#
# OUTPUT VARIABLES:
#
#   YAML_CPP_STATIC_LIB_TARGET: name of static library target that contains all
#                               of yaml-cpp library objects.
#   YAML_CPP_STATIC_LIB_INC_DIR: directory of installed yaml-cpp lib headers

include(${REPOSITORY_DIR}/src/NupicLibraryUtils.cmake) # for MERGE_STATIC_LIBRARIES


# Output static library target for linking and dependencies
set(YAML_CPP_STATIC_LIB_TARGET yaml-cpp-bundle)


set(yamlcpplib_url "${REPOSITORY_DIR}/external/common/share/yaml-cpp/yaml-cpp-release-0.6.2.tar.gz")
set(yamlcpplib_install_prefix "${EP_BASE}/Install/YamlCppStaticLib")
set(yamlcpplib_install_lib_dir "${yamlcpplib_install_prefix}/lib")

#TODO: How to get CMake to figure out this path for MSVC
# Path to static yaml-cpp installed by external project
if(MSVC)
   set(flags "/W3 /Gm- /EHsc /FC /nologo /Zc:__cplusplus /std:${INTERNAL_CPP_STANDARD} /MT") 
   if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
       set(yamlcpplib_built_archive_file "${yamlcpplib_install_lib_dir}/libyaml-cppmd.lib")
   else()
       set(yamlcpplib_built_archive_file "${yamlcpplib_install_lib_dir}/libyaml-cppmdd.lib")
   endif()
else()
   # Provide a string variant of the INTERNAL_CXX_FLAGS list
   set(flags)
   foreach(flag_item ${INTERNAL_CXX_FLAGS})
     set(flags "${flags} ${flag_item}")
   endforeach()

   set(yamlcpplib_built_archive_file
      "${yamlcpplib_install_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}yaml-cpp${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

message(STATUS "FLAGS in yaml: ${flags}")

ExternalProject_Add(YamlCppStaticLib

    URL ${yamlcpplib_url}

    UPDATE_COMMAND ""

    CMAKE_GENERATOR ${CMAKE_GENERATOR}

    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DYAML_CPP_BUILD_TOOLS=OFF
	-DYAML_CPP_BUILD_TESTS=OFF # causes build errors with gtest (as of YamlCpp 0.6.2)
        -DYAML_CPP_BUILD_CONTRIB=OFF
        -DCMAKE_CXX_FLAGS=${flags}
        -DCMAKE_INSTALL_PREFIX=${yamlcpplib_install_prefix}
)


# Wrap external project-generated static library in an `add_library` target.
merge_static_libraries(${YAML_CPP_STATIC_LIB_TARGET}
                       "${yamlcpplib_built_archive_file}")
add_dependencies(${YAML_CPP_STATIC_LIB_TARGET} YamlCppStaticLib)

# Export directory of installed yaml-cpp lib headers to parent
set(yaml-cpp_INCLUDE_DIRS "${yamlcpplib_install_prefix}/include" PARENT_SCOPE)
set(yaml-cpp_LIBRARIES ${yamlcpplib_built_archive_file} PARENT_SCOPE)
