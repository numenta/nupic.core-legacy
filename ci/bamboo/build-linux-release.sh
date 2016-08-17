#!/bin/bash
set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the common setup
${DIR}/setup-dependencies-linux.sh

# Install nupic.core dependencies
BUILD_TYPE="Release" \
RESULT_KEY="${bamboo_buildResultKey}" \
  ${DIR}/../build-and-test-nupic-bindings.sh "$@"
