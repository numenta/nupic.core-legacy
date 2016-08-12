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
