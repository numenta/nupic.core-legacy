#!/bin/bash
# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# ASUMPTIONS: Expects a pristine nupic.core source tree without any remnant
#             build artifacts from prior build attempts. Otherwise, behavior is
#             undefined.
#
# OUTPUTS: see nupic.core/ci/build-and-test-nupic-bindings.sh


set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the common setup
${DIR}/setup-dependencies-linux.sh

# Install the Include What You Use tool used by debug build
apt-get install -y iwyu

# Build and test
PIP_USER=1 \
BUILD_TYPE="Debug" \
  ${DIR}/../build-and-test-nupic-bindings.sh "$@"
