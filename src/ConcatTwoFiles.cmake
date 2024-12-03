# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Concatenates two files into a target file, overwriting target file if already
# exists; either of the source files may be the same path as the target file.

# This script is intended to be invoked via `${CMAKE_COMMAND} -DSRC_FILE_1= ...`.

# ARGS:
#
#  SRC_FILE_1: path of first source file; may be same as TARGET_FILE.
#  SRC_FILE_2: path of second source file; may be same as TARGET_FILE.
#  TARGET_FILE: path of target file.


function(CONCAT_TWO_FILES SRC_FILE_1 SRC_FILE_2 TARGET_FILE)
    file(READ ${SRC_FILE_1} src_1_content)
    file(READ ${SRC_FILE_2} src_2_content)
    file(WRITE ${TARGET_FILE} "${src_1_content}${src_2_content}")
endfunction(CONCAT_TWO_FILES)

concat_two_files(${SRC_FILE_1} ${SRC_FILE_2} ${TARGET_FILE})
