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

# Utilities for manipulating libraries

cmake_minimum_required(VERSION 2.8)
project(nupic_core_library_utils CXX)


# function MERGE_STATIC_LIBRARIES
#
# Generate a new static library target that will merge the objects
# of the given static libraries.
#
# :param LIB_TARGET: the name to use for the new library target to be
#   passed verbatim as the target arg to `ADD_LIBRARY`
#
# :param STATIC_LIBS: a list of static libraries to be merge. The
#   elements of this list may be a combination of archive file paths and
#   static library target names (i.e., those defined here via
#   add_library(xyz STATIC ...)).

function(MERGE_STATIC_LIBRARIES LIB_TARGET STATIC_LIBS)

  message(STATUS "MERGE_STATIC_LIBRARIES "
          "LIB_TARGET=${LIB_TARGET}, "
          "STATIC_LIBS = ${STATIC_LIBS}")

  # We need at least one source file for ADD_LIBRARY
  set(dummy_source_file "${CMAKE_CURRENT_BINARY_DIR}/${LIB_TARGET}_dummy.c++")

  # Define a static lib containing the dummy source file; we will subsequently
  # add a post-build custom step that will add the objects from the given static
  # libraries
  add_library(${LIB_TARGET} STATIC ${dummy_source_file})
  set_target_properties(${LIB_TARGET} PROPERTIES COMPILE_FLAGS
                        ${INTERNAL_CXX_FLAGS_OPTIMIZED})

  set(static_lib_locations)
  set(dummy_dependencies)
  set(link_libs)

  foreach(lib ${STATIC_LIBS})
    list(APPEND dummy_dependencies ${lib})

    if (NOT TARGET ${lib})
      # Assuming a path of an externally-generated static library
      list(APPEND static_lib_locations ${lib})
    else()
      # Assuming a cmake static library target
      get_target_property(lib_type ${lib} TYPE)
      if(NOT ${lib_type} STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "Expected static lib source object ${lib}, but got type=${lib_type}!")
      endif()

      list(APPEND static_lib_locations "$<TARGET_FILE:${lib}>")
      add_dependencies(${LIB_TARGET} ${lib})

      # Collect its link interface
      get_target_property(link_iface ${lib} INTERFACE_LINK_LIBRARIES)
      if (link_iface)
        list(APPEND link_libs ${link_iface})
        message(STATUS "MERGE_STATIC_LIBRARIES: "
                "INTERFACE_LINK_LIBRARIES[${lib}] = ${link_iface}.")
      else()
        message(STATUS "MERGE_STATIC_LIBRARIES: "
                "Link interface not specified in source lib ${lib}.")
      endif()
    endif()
  endforeach()

  # Transfer link interface of source libraries to target
  if (link_libs)
    list(REMOVE_DUPLICATES link_libs)
    target_link_libraries(${LIB_TARGET} ${link_libs})
  endif()

  # Force relink whenever any of the source libraries change
  add_custom_command(OUTPUT ${dummy_source_file}
                     COMMAND ${CMAKE_COMMAND} -E touch ${dummy_source_file}
                     DEPENDS ${dummy_dependencies})

  # Merge the archives
  if(MSVC)
    # pass source libs to lib.exe via STATIC_LIBRARY_FLAGS target property

    set(msvc_library_flags "")
    foreach(lib_location ${static_lib_locations})
      set(msvc_library_flags "${msvc_library_flags} ${lib_location}")
    endforeach()
    set_target_properties(${LIB_TARGET} PROPERTIES STATIC_LIBRARY_FLAGS
                          "${msvc_library_flags}")
  else()
    # UNIX OR MSYS OR MINGW: use post-build command to extract objects from
    # source libs and repack them for the target library

    set(target_location_gen "$<TARGET_FILE:${LIB_TARGET}>")

    # NOTE With cmake 2.8.11+, we could use "$<SEMICOLON>", but default Travis
    # environment is configured with cmake 2.8.7
    set(lib_locations_separator "++++")
    string(REPLACE ";" ${lib_locations_separator}
           escaped_lib_locations_arg "${static_lib_locations}")

    add_custom_command(
      TARGET ${LIB_TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove "${target_location_gen}"
      COMMAND
         ${CMAKE_COMMAND}
             -DLIB_TARGET="${LIB_TARGET}"
             -DTARGET_LOCATION="${target_location_gen}"
             -DSRC_LIB_LOCATIONS="${escaped_lib_locations_arg}"
             -DLIST_SEPARATOR=${lib_locations_separator}
             -DBINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
             -DCMAKE_AR="${CMAKE_AR}"
             -P ${CMAKE_SOURCE_DIR}/src/CombineUnixArchives.cmake
      COMMENT "Combining ${target_location_gen} for target ${LIB_TARGET} from ${static_lib_locations}."
    )

  endif()

endfunction(MERGE_STATIC_LIBRARIES)
