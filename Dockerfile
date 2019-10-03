# Default arch. Pass in like "--build-arg arch=arm64".
#   Supports Debian arches: amd64, arm64, etc.
#   Our circleci arm64 build uses this specifically.
#   https://docs.docker.com/engine/reference/commandline/build/
ARG arch=amd64

# Multiarch Debian 10 Buster (amd64, arm64, etc).
#   https://hub.docker.com/r/multiarch/debian-debootstrap
FROM multiarch/debian-debootstrap:${arch}-buster

RUN apt-get update
RUN apt-get install -y --no-install-suggests \
    cmake \
    g++-8 \
    git-core \
    libyaml-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-venv

ADD . /usr/local/src/htm.core
WORKDIR /usr/local/src/htm.core

# Setup py env
#! RUN python3 -m venv pyenv && . pyenv/bin/activate && python --version

RUN ln -s /usr/bin/python3 /usr/local/bin/python && python --version 

RUN python -m pip install --upgrade setuptools pip wheel

# Install
RUN pip uninstall -y htm.core
RUN python -m pip install \
# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/htm.core/bindings/py/dist
#        --cache-dir /usr/local/src/htm.core/pip-cache \
#        --build /usr/local/src/htm.core/pip-build \
#        --no-clean \
        -r requirements.txt
RUN python setup.py install --force

# Test
RUN python setup.py test #Note, if you get weird import errors here, 
# do `git clean -xdf` in your host system, and rerun the docker

# build wheel and release package
RUN python setup.py bdist_wheel
RUN cd build/scripts && \
    make install && \
    make package && \
    ls * && \ 
    ls ../Release/distr/dist/* 

