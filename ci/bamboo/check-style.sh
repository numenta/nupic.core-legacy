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

if git clang-format --diff master | grep -q '^diff'
then
    echo "ERROR: Code changes do not comply with numenta's code style rules."
    echo "Please run 'git fetch upstream && git clang-format upstream/master' before commit."
    echo "See 'githooks/README.md' for instructions on how to install clang-format on your platform."
    git clang-format --diff master
    exit 1
fi
exit 0

