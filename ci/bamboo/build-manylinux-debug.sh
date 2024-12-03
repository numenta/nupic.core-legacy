#!/bin/bash
# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# This script runs inside numenta's custom manylinux docker image
# quay.io/numenta/manylinux1_x86_64_centos6 and builds the debug manylinux
# x86_64 wide-unicode nupic.bindings wheel per PEP-513. See
# https://github.com/numenta/manylinux.
#
# ASUMPTIONS: Expects a pristine nupic.core source tree without any remnant
#             build artifacts from prior build attempts. Otherwise, behavior is
#             undefined.
#
# OUTPUTS: see nupic.core/ci/build-and-test-nupic-bindings.sh


set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configure environment for manylinux build
source ${DIR}/manylinux-build-env.rc

# Install the Include What You Use tool used by debug build
yum install -y iwyu

# Build and test the manylinux wheel; see build-and-test-nupic-bindings.sh for
# destination wheelhouse
BUILD_TYPE="Debug" \
WHEEL_PLAT="manylinux1_x86_64" \
  ${DIR}/../build-and-test-nupic-bindings.sh "$@"
