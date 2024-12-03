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

if git clang-format --diff master | grep -q '^diff'
then
    echo "ERROR: Code changes do not comply with numenta's code style rules."
    echo "Please run 'git fetch upstream && git clang-format upstream/master' before commit."
    echo "See 'githooks/README.md' for instructions on how to install clang-format on your platform."
    git clang-format --diff master
    exit 1
fi
exit 0

