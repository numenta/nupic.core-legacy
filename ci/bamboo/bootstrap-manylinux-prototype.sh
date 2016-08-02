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

# Build nupic.bindings python extension using manylinux docker image.

set -o errexit
set -o xtrace

NUPIC_CORE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

#echo "NUPIC_CORE_DIR=${NUPIC_CORE_DIR}"

DOCKER_IMAGE="quay.io/numenta/manylinux1_x86_64_centos6"

docker run -ti --rm -v ${NUPIC_CORE_DIR}:/nupic.core ${DOCKER_IMAGE} \
  /nupic.core/ci/bamboo/build-manylinux-prototype.sh

