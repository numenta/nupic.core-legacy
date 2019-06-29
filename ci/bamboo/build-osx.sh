#!/bin/bash
# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
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
# ----------------------------------------------------------------------

set -o errexit
set -o xtrace

# Fixup $PATH for --user installation
export PATH=${HOME}/Library/Python/2.7/bin:${PATH}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build and test the manylinux wheel; see build-and-test-nupic-bindings.sh for
# destination wheelhouse
BUILD_TYPE="Release" \
  WHEEL_PLAT="macosx_10_9_intel" \
  ARCHFLAGS="-arch x86_64" \
  PIP_USER=1 \
  ${DIR}/../build-and-test-nupic-bindings.sh
