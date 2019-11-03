## Default arch. Pass in like "--build-arg arch=arm64".
#   Supports Debian arches: amd64, arm64, etc.
#   Our circleci arm64 build uses this specifically.
#   https://docs.docker.com/engine/reference/commandline/build/
## To run a build using this file locally, do: 
# docker run --privileged --rm -it multiarch/qemu-user-static:register
# docker build -t htm-arm64-docker --build-arg arch=arm64 .
# docker run -it htm-arm64-docker

#target compile arch
ARG arch=arm64
#host HW arch
ARG host=amd64

## Stage 0: deboostrap: setup cross-compile env 
FROM multiarch/qemu-user-static as bootstrap
ARG arch
ARG host
RUN echo "Switching from $host to $arch" && uname -a

## Stage 1: build of htm.core on the target platform
# Multiarch Debian 10 Buster (amd64, arm64, etc).
#   https://hub.docker.com/r/multiarch/debian-debootstrap
FROM multiarch/alpine:${arch}-latest-stable as build
ARG arch
#copy value of ARG arch from above 
RUN echo "Building HTM for${arch}" && uname -a

RUN apk add --update  \
    cmake \
    make \
    g++ \
    git \
    python3-dev \
    py3-numpy 

ADD . /usr/local/src/htm.core
WORKDIR /usr/local/src/htm.core

# Setup py env
#! RUN python3 -m venv pyenv && . pyenv/bin/activate && python --version

RUN ln -s /usr/bin/python3 /usr/local/bin/python && python --version 

RUN python -m pip install --upgrade setuptools pip wheel

# Install
RUN python -m pip uninstall -y htm.core
RUN python -m pip install \
# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/htm.core/bindings/py/dist
#        --cache-dir /usr/local/src/htm.core/pip-cache \
#        --build /usr/local/src/htm.core/pip-build \
#        --no-clean \
        -r requirements.txt
RUN mkdir -p build/scripts && \
    cd build/scripts && \
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DBINDING_BUILD=Python3 && \
    make -j4 && make install

RUN python setup.py install --force

# Test
#RUN python setup.py test #Note, if you get weird import errors here, 
# do `git clean -xdf` in your host system, and rerun the docker

## Stage 2: create release packages (for PyPI, GH Releases)
RUN python setup.py bdist_wheel
RUN cd build/scripts && \
    make install && \
    make package

RUN mkdir dist && \
    cp -a build/scripts/*.tar.gz dist && \
    cp -a build/Release/distr/dist/*.whl dist && \
    ls -l dist

