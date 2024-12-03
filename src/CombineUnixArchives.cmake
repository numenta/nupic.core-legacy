# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Combine multiple Unix static libraries into a single static library.
#
# This script is intended to be invoked via `${CMAKE_COMMAND} -DLIB_TARGET= ...`.
#
# ARGS:
#
#   LIB_TARGET: target name of resulting static library (passed to add_library)
#   TARGET_LOCATION: Full path to the target library
#   SRC_LIB_LOCATIONS: List of source static library paths
#   LIST_SEPARATOR: separator string that separates paths in
#     SRC_LIB_LOCATIONS; NOTE with cmake 2.8.11+, caller could use the generator
#     "$<SEMICOLON>" in SRC_LIB_LOCATIONS, and this arg would be unnecessary.
#   BINARY_DIR: The value of ${CMAKE_CURRENT_BINARY_DIR} from caller
#   CMAKE_AR: The value of ${CMAKE_AR} from caller

function(COMBINE_UNIX_ARCHIVES
         LIB_TARGET
         TARGET_LOCATION
         SRC_LIB_LOCATIONS
         LIST_SEPARATOR
         BINARY_DIR
         CMAKE_AR)

  message(STATUS
          "COMBINE_UNIX_ARCHIVES("
          "  LIB_TARGET=${LIB_TARGET}, "
          "  TARGET_LOCATION=${TARGET_LOCATION}, "
          "  SRC_LIB_LOCATIONS=${SRC_LIB_LOCATIONS}, "
          "  LIST_SEPARATOR=${LIST_SEPARATOR}, "
          "  BINARY_DIR=${BINARY_DIR}, "
          "  CMAKE_AR=${CMAKE_AR})")

  string(REPLACE ${LIST_SEPARATOR} ";"
         SRC_LIB_LOCATIONS "${SRC_LIB_LOCATIONS}")

  set(scratch_dir ${BINARY_DIR}/combine_unix_archives_${LIB_TARGET})
  file(MAKE_DIRECTORY ${scratch_dir})

  # Extract archives into individual directories to avoid object file collisions
  set(all_object_locations)
  foreach(lib ${SRC_LIB_LOCATIONS})
    message(STATUS "COMBINE_UNIX_ARCHIVES: LIB_TARGET=${LIB_TARGET}, src-lib=${lib}")
    # Create working directory for current source lib
    get_filename_component(basename ${lib} NAME)
    set(working_dir ${scratch_dir}/${basename}.dir)
    file(MAKE_DIRECTORY ${working_dir})

    # Extract objects from current source lib
    execute_process(COMMAND ${CMAKE_AR} -x ${lib}
                    WORKING_DIRECTORY ${working_dir}
                    RESULT_VARIABLE exe_result)
    if(NOT "${exe_result}" STREQUAL "0")
      message(FATAL_ERROR "COMBINE_UNIX_ARCHIVES: obj extraction process failed exe_result='${exe_result}'")
    endif()

    # Accumulate objects
    if(UNIX)
      # Linux or OS X
      set(globbing_ext "o")
    else()
      # i.e., Windows with MINGW toolchain
      set(globbing_ext "obj")
    endif()

    file(GLOB_RECURSE objects "${working_dir}/*.${globbing_ext}")
    if (NOT objects)
      file(GLOB_RECURSE working_dir_listing "${working_dir}/*")
      message(FATAL_ERROR
              "COMBINE_UNIX_ARCHIVES: no extracted obj files from ${lib} "
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
      # Use relative paths to work around too-long command failure when building
      # on Windows in AppVeyor
      file(RELATIVE_PATH new_obj_file_path ${scratch_dir} ${new_obj_file_path})
      list(APPEND all_object_locations ${new_obj_file_path})
    endforeach()
  endforeach()

  # Generate the target static library from all source objects
  file(TO_NATIVE_PATH ${TARGET_LOCATION} TARGET_LOCATION)
  execute_process(COMMAND ${CMAKE_AR} rcs ${TARGET_LOCATION} ${all_object_locations}
                  WORKING_DIRECTORY ${scratch_dir} RESULT_VARIABLE exe_result)
  if(NOT "${exe_result}" STREQUAL "0")
    message(FATAL_ERROR "COMBINE_UNIX_ARCHIVES: archive construction process failed exe_result='${exe_result}'")
  endif()

  # Remove scratch directory
  file(REMOVE_RECURSE ${scratch_dir})
endfunction(COMBINE_UNIX_ARCHIVES)


combine_unix_archives(${LIB_TARGET}
                      ${TARGET_LOCATION}
                      "${SRC_LIB_LOCATIONS}"
                      ${LIST_SEPARATOR}
                      ${BINARY_DIR}
                      ${CMAKE_AR})
