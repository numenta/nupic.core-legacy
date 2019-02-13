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

# Expand a static library into an object library
# NOTE: for now this only works with Unix like OS's (Linux, OSx) using AR.
#
# This script is intended to be invoked via `${CMAKE_COMMAND} -DLIB_TARGET= ...`.
#
# ARGS:
#
#   LIB_TARGET: target name of resulting object library (unique identifier for call)
#   SRC_LIB_LOCATIONS: List of source static library paths
#   LIST_SEPARATOR: separator string that separates paths in
#     SRC_LIB_LOCATIONS; NOTE with cmake 2.8.11+, caller could use the generator
#     "$<SEMICOLON>" in SRC_LIB_LOCATIONS, and this arg would be unnecessary.
#   BINARY_DIR: The value of ${CMAKE_CURRENT_BINARY_DIR} from caller
#   CMAKE_AR: The value of ${CMAKE_AR} from caller

function(ExpandStaticLib
         LIB_TARGET
         SRC_LIB_LOCATIONS
         LIST_SEPARATOR
         BINARY_DIR
         CMAKE_AR)

 # message(STATUS
 #         "ExpandStaticLib("
 #         "  LIB_TARGET=${LIB_TARGET}, "
 #         "  SRC_LIB_LOCATIONS=${SRC_LIB_LOCATIONS}, "
 #         "  LIST_SEPARATOR=${LIST_SEPARATOR}, "
 #         "  BINARY_DIR=${BINARY_DIR} ")

  string(REPLACE ${LIST_SEPARATOR} ";"
         SRC_LIB_LOCATIONS "${SRC_LIB_LOCATIONS}")
  if(MSVC)
    message(FATAL_ERROR "ExpandStaticLib does not work with Windows.")
  endif()

  set(scratch_dir ${BINARY_DIR}/ExpandStaticLib_${LIB_TARGET})
  if(EXISTS "${scratch_dir}")
    file(REMOVE_RECURSE "${scratch_dir}")
  endif()
  file(MAKE_DIRECTORY ${scratch_dir})

  # Extract archives into individual directories to avoid object file collisions
  set(all_object_locations)
  foreach(lib ${SRC_LIB_LOCATIONS})
    message(STATUS "ExpandStaticLib: LIB_TARGET=${LIB_TARGET}, src-lib=${lib}")
    # Create working directory for current source lib
    get_filename_component(basename ${lib} NAME)
    set(working_dir ${scratch_dir}/${basename}.dir)
    file(MAKE_DIRECTORY ${working_dir})

    # Extract objects from current source lib 
    execute_process(COMMAND ${CMAKE_AR} -x ${lib}
                    WORKING_DIRECTORY ${working_dir}
                    RESULT_VARIABLE exe_result)
    if(NOT "${exe_result}" STREQUAL "0")
      message(FATAL_ERROR "ExpandStaticLib: obj extraction process failed exe_result='${exe_result}'")
    endif()
    
    set(globbing_ext "o")

    file(GLOB_RECURSE objects "${working_dir}/*.${globbing_ext}")
    if (NOT objects)
      file(GLOB_RECURSE working_dir_listing "${working_dir}/*")
      message(FATAL_ERROR
              "ExpandStaticLib: no extracted object files from ${lib} "
              "found in ${working_dir} using globbing_ext=${globbing_ext}, "
              "but the following entries were found: ${working_dir_listing}.")
    endif()

    # Prepend source lib name to object name to help prevent collisions during
    # subsequent archive extractions. This helps guard against obj file
    # overwrites during object extraction in cases where the same object name
    # existed in two different source archives, but does not prevent the issue
    # if same-named objects exist in one archive.
    foreach(old_obj_file_path ${objects})
      get_filename_component(old_obj_file_name ${old_obj_file_path} NAME)
      set(new_obj_file_path "${working_dir}/${basename}-${old_obj_file_name}")
      file(RENAME ${old_obj_file_path} ${new_obj_file_path})
# message(STATUS "${new_obj_file_path}")
      list(APPEND all_object_locations ${new_obj_file_path})
    endforeach()
  endforeach()
  
  set(all_object_locations ${all_object_locations} PARENT_SCOPE)

  # Note: do not remove object directory
endfunction(ExpandStaticLib)



