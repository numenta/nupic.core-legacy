#!/bin/bash
# Copyright 2016 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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
