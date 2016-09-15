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

# Build a non-debug nupic.bindings wheel for OS X
#
# ASSUMPTIONS: Expects a pristine nupic.core source tree without any remnant
#              build artifacts from prior build attempts. Otherwise, behavior is
#              undefined.
#
#              The desired python is reflected in PATH
#
# OUTPUTS: see nupic.core/ci/build-and-test-nupic-bindings.sh

set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

## Use virtualenv to allow pip installs without --user in
## build-and-test-nupic-bindings.sh
#pip install --user virtualenv
#virtualenv ./venv
#source ./venv/bin/activate

# Build and test
ARCHFLAGS="-arch x86_64" \
BUILD_TYPE="Release" \
RESULT_KEY="${bamboo_buildResultKey}" \
  ${DIR}/../build-and-test-nupic-bindings.sh "$@"
