#!/bin/bash
# ----------------------------------------------------------------------
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
# ----------------------------------------------------------------------

set -o errexit
set -o xtrace

export ARCHFLAGS="-arch x86_64"

# Fixup $PATH for --user installation
export PATH=${HOME}/Library/Python/2.7/bin:${PATH}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configure environment for manylinux build
source ${DIR}/manylinux-build-env.rc

# Build and test the manylinux wheel; see build-and-test-nupic-bindings.sh for
# destination wheelhouse
BUILD_TYPE="Release" \
WHEEL_PLAT="manylinux1_x86_64" \
  ${DIR}/../build-and-test-nupic-bindings.sh "$@"
