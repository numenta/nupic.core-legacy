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
    python3-minimal \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-venv

ADD . /usr/local/src/htm.core
WORKDIR /usr/local/src/htm.core

# Setup py env
#RUN python3 -m venv pyenv && . pyenv/bin/activate
RUN pip3 install --upgrade setuptools pip wheel
#RUN export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.7/dist-packages

# Install
RUN pip3 install \
# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/htm.core/bindings/py/dist
#        --cache-dir /usr/local/src/htm.core/pip-cache \
#        --build /usr/local/src/htm.core/pip-build \
#        --no-clean \
        -r requirements.txt
RUN python3 setup.py install --force

# Test
RUN python3 setup.py test

