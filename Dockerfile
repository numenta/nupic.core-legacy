# Default arch. Pass in like "--build-arg arch=arm64".
#   Our circleci arm64 build uses this specifically.
#   https://docs.docker.com/engine/reference/commandline/build/
ARG arch=x86_64

# Multiarch Ubuntu Bionic 18.04. arches: x86_64, arm64, etc.
#   https://hub.docker.com/r/multiarch/ubuntu-core/tags/
FROM multiarch/ubuntu-core:$arch-bionic

RUN apt-get install -y \
    curl \
    wget \
    git-core \
    g++ \
    cmake \
    python2.7 \
    python2.7-dev \
    zlib1g-dev \
    bzip2 \
    libyaml-dev \
    libyaml-0-2
RUN wget http://releases.numenta.org/pip/1ebd3cb7a5a3073058d0c9552ab074bd/get-pip.py -O - | python
RUN pip install --upgrade setuptools
RUN pip install wheel

ENV CXX g++

ADD . /usr/local/src/nupic.cpp
WORKDIR /usr/local/src/nupic.cpp

# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/nupic.cpp/bindings/py/dist
RUN pip install \
        --cache-dir /usr/local/src/nupic.cpp/pip-cache \
        --build /usr/local/src/nupic.cpp/pip-build \
        --no-clean \
        -r bindings/py/packaging/requirements.txt && \
    python setup.py bdist bdist_dumb bdist_wheel sdist
