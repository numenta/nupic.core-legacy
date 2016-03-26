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

# Moves files to destination directory, creating the destination directory if
# needed.

# ARGS:
#
#  SRC_DIR_PATH
#  GLOBBING_EXPR
#  DEST_DIR_PATH


function(move_files_to_new_dir SRC_DIR_PATH GLOBBING_EXPR DEST_DIR_PATH)
    file(GLOB FILE_PATHS ${SRC_DIR_PATH} ${GLOBBING_EXPR})
    file(MAKE_DIRECTORY ${DEST_DIR_PATH})
    file(COPY ${FILE_PATHS} DESTINATION ${DEST_DIR_PATH})
    file(REMOVE ${FILE_PATHS})
endfunction(move_files_to_new_dir)


move_files_to_new_dir(${SRC_DIR_PATH} ${GLOBBING_EXPR} ${DEST_DIR_PATH})