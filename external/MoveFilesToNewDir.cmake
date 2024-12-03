# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Moves files to destination directory, creating the destination directory if
# needed.

# ARGS:
#
#  GLOBBING_EXPR
#  DEST_DIR_PATH


function(move_files_to_new_dir GLOBBING_EXPR DEST_DIR_PATH)
    file(GLOB FILE_PATHS ${GLOBBING_EXPR})
    file(MAKE_DIRECTORY ${DEST_DIR_PATH})

    foreach(FILEPATH ${FILE_PATHS})
            file(COPY ${FILEPATH} DESTINATION ${DEST_DIR_PATH})
    endforeach()

    foreach(FILEPATH ${FILE_PATHS})
        file(REMOVE ${FILEPATH})
    endforeach()
endfunction(move_files_to_new_dir)

move_files_to_new_dir(${GLOBBING_EXPR} ${DEST_DIR_PATH})
