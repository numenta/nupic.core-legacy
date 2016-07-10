#!/bin/bash
set -o errexit
set -o xtrace

NUPIC_CORE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

#echo "NUPIC_CORE_DIR=${NUPIC_CORE_DIR}"

DOCKER_IMAGE="quay.io/numenta/manylinux1_x86_64_centos6"

docker run -ti --rm -v ${NUPIC_CORE_DIR}:/nupic.core ${DOCKER_IMAGE} /nupic.core/ci/bamboo/build-manylinux-prototype.sh

