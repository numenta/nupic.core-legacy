#!/bin/bash
set -o errexit
set -o xtrace

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Environment defaults
if [ -z "${USER}" ]; then
    USER="docker"
fi
export USER
