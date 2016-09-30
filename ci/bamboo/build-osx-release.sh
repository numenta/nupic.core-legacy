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

# Build and test the manylinux wheel; see build-and-test-nupic-bindings.sh for
# destination wheelhouse
BUILD_TYPE="Release" \
WHEEL_PLAT="maxosx_10_9_intel"

if [[ $1 == --help ]]; then
  echo "${USAGE}"
  exit 0
fi

if [[ $# > 0 ]]; then
  echo "ERROR Unexpected arguments: ${@}" >&2
  echo "${USAGE}" >&2
  exit 1
fi


set -o xtrace


# Apply defaults
BUILD_TYPE=${BUILD_TYPE-"Release"}


NUPIC_CORE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

DEST_WHEELHOUSE="${NUPIC_CORE_ROOT}/nupic_bindings_wheelhouse"

TEST_RESULTS_DIR="${NUPIC_CORE_ROOT}/test_results"

echo "RUNNING NUPIC BINDINGS BUILD: BUILD_TYPE=${BUILD_TYPE}, " \
     "DEST_WHEELHOUSE=${DEST_WHEELHOUSE}" >&2

# Install pycapnp to get the matching capnproto headers for nupic.core build
# NOTE Conditional pycapnp dependency should be incorporated into
# bindings/py/requirements.txt to abstract it from upstream scripts.
pip install --user pycapnp==0.5.8

# Install nupic.bindings dependencies; the nupic.core cmake build depends on
# some of them (e.g., numpy).
pip install --user -r ${NUPIC_CORE_ROOT}/bindings/py/requirements.txt


#
# Build nupic.bindings
#

# NOTE without -p to force build failure upon pre-existing build side-effects
mkdir ${NUPIC_CORE_ROOT}/build
mkdir ${NUPIC_CORE_ROOT}/build/scripts

cd ${NUPIC_CORE_ROOT}/build/scripts

# Configure nupic.core build
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  EXTRA_CMAKE_DEFINITIONS="-DNUPIC_IWYU=ON -DNTA_COV_ENABLED=ON"
fi

cmake ${NUPIC_CORE_ROOT} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ${EXTRA_CMAKE_DEFINITIONS} \
    -DCMAKE_INSTALL_PREFIX=${NUPIC_CORE_ROOT}/build/release \
    -DPY_EXTENSIONS_DIR=${NUPIC_CORE_ROOT}/bindings/py/nupic/bindings

# Build nupic.core
make install

# Build nupic.bindings python extensions from nupic.core build artifacts
if [[ $WHEEL_PLAT ]]; then
  EXTRA_WHEEL_OPTIONS="--plat-name ${WHEEL_PLAT}"
fi

cd ${NUPIC_CORE_ROOT}
python setup.py bdist_wheel --dist-dir ${DEST_WHEELHOUSE} ${EXTRA_WHEEL_OPTIONS}


#
# Test
#

# Install nupic.bindings before running c++ tests; py_region_test depends on it
pip install --ignore-installed --user ${DEST_WHEELHOUSE}/nupic.bindings-*.whl

# Run the nupic.core c++ tests
cd ${NUPIC_CORE_ROOT}/build/release/bin
./cpp_region_test
./py_region_test
./unit_tests

# Run nupic.bindings python tests

mkdir ${TEST_RESULTS_DIR}

cd ${TEST_RESULTS_DIR}    # so that py.test will deposit its artifacts here

# Run tests with pytest options per nupic.core/setup.cfg
py.test ${NUPIC_CORE_ROOT}/bindings/py/tests
