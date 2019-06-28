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


USAGE="Usage:

[BUILD_TYPE=Release | Debug] [WHEEL_PLAT=platform] $( basename ${0} )

This script builds and tests the nupic.bindings Python extension.

In Debug builds, also
  - Turns on the Include What You Use check if clang is being used (assumes
    iwyu is installed)

ASUMPTION: Expects a pristine nupic.core source tree without any remnant build
   artifacts from prior build attempts. Otherwise, behavior is undefined.


INPUT ENVIRONMENT VARIABLES:

  BUILD_TYPE : Specifies build type, which may be either Release or Debug;
               defaults to Release. [OPTIONAL]
  WHEEL_PLAT : Wheel platform name; pass manylinux1_x86_64 for manylinux build;
               leave undefined for all other builds.

OUTPUTS:
  nupic.bindings wheel: On success, the resulting wheel will be located in the
                        subdirectory nupic_bindings_wheelhouse of the source
                        tree's root directory.

  test results: nupic.bindings test results will be located in the subdirectory
                test_results of the source tree's root directory with the
                the following content:

                junit-test-results.xml
                htmlcov/

"

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


HTM_CORE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

DEST_WHEELHOUSE="${HTM_CORE_ROOT}/nupic_bindings_wheelhouse"

TEST_RESULTS_DIR="${HTM_CORE_ROOT}/test_results"

echo "RUNNING HTM BINDINGS BUILD: BUILD_TYPE=${BUILD_TYPE}, " \
     "DEST_WHEELHOUSE=${DEST_WHEELHOUSE}" >&2

# Install nupic.bindings dependencies; the nupic.core cmake build depends on
# some of them (e.g., numpy).
pip install \
    --ignore-installed \
    -r ${HTM_CORE_ROOT}/bindings/py/requirements.txt

#
# Build nupic.bindings
#

# NOTE without -p to force build failure upon pre-existing build side-effects
mkdir ${HTM_CORE_ROOT}/build
mkdir ${HTM_CORE_ROOT}/build/scripts

cd ${HTM_CORE_ROOT}/build/scripts

# Configure nupic.core build
if [[ "$BUILD_TYPE" == "Debug" ]]; then
  EXTRA_CMAKE_DEFINITIONS="-DNTA_COV_ENABLED=ON"

  # Only add iwyu for clang builds
  if [[ $CC == *"clang"* ]]; then
    EXTRA_CMAKE_DEFINITIONS="-DHTM_IWYU=ON ${EXTRA_CMAKE_DEFINITIONS}"
  fi
fi

cmake ${HTM_CORE_ROOT} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ${EXTRA_CMAKE_DEFINITIONS} \
    -DCMAKE_INSTALL_PREFIX=${HTM_CORE_ROOT}/build/release \
    -DPY_EXTENSIONS_DIR=${HTM_CORE_ROOT}/bindings/py/src/nupic/bindings

# Build nupic.core
make install

# Build nupic.bindings python extensions from nupic.core build artifacts
if [[ $WHEEL_PLAT ]]; then
  EXTRA_WHEEL_OPTIONS="--plat-name ${WHEEL_PLAT}"
fi

cd ${HTM_CORE_ROOT}
python setup.py bdist_wheel --dist-dir ${DEST_WHEELHOUSE} ${EXTRA_WHEEL_OPTIONS}


#
# Test
#

# Install nupic.bindings before running c++ tests; py_region_test depends on it
pip install \
    --ignore-installed \
    ${DEST_WHEELHOUSE}/nupic.bindings-*.whl

# Run the nupic.core c++ tests
cd ${HTM_CORE_ROOT}/build/release/bin
./cpp_region_test
./py_region_test
./unit_tests

# These are utilities or demonstration executables so leave out of main build
# to keep build times down.
#./connections_performance_test
#./hello_sp_tp
#./helloregion
#./prototest


# Run nupic.bindings python tests

mkdir ${TEST_RESULTS_DIR}

cd ${TEST_RESULTS_DIR}    # so that py.test will deposit its artifacts here

# Run tests with pytest options per nupic.core/setup.cfg
py.test ${HTM_CORE_ROOT}/bindings/py/tests
