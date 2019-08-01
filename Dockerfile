# Default arch. Pass in like "--build-arg arch=arm64".
#   Supports Debian arches: amd64, arm64, etc.
#   Our circleci arm64 build uses this specifically.
#   https://docs.docker.com/engine/reference/commandline/build/
ARG arch=amd64

# Multiarch Debian 10 Buster (Jun 2019) (amd64, arm64, etc).
#   https://hub.docker.com/r/multiarch/debian-debootstrap
FROM multiarch/debian-debootstrap:$arch-buster

RUN apt-get update
RUN apt-get install -y \
    cmake \
    g++ \
    git-core \
    libyaml-dev \
    python \
    python-dev \
    python-numpy \
    python-pip

ADD . /usr/local/src/htm.core
WORKDIR /usr/local/src/htm.core

# Install
RUN pip install --upgrade setuptools
RUN pip install wheel
RUN pip install \
# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/htm.core/bindings/py/dist
#        --cache-dir /usr/local/src/htm.core/pip-cache \
#        --build /usr/local/src/htm.core/pip-build \
#        --no-clean \
        -r bindings/py/packaging/requirements.txt
RUN python setup.py install

# Test
RUN ./build/Release/bin/unit_tests
RUN python setup.py test

