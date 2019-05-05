# Default arch. Pass in like "--build-arg arch=arm64".
#   Our circleci arm64 build uses this specifically.
#   https://docs.docker.com/engine/reference/commandline/build/
ARG arch=x86_64

# Multiarch Ubuntu Bionic 18.04. arches: x86_64, arm64, etc.
#   https://hub.docker.com/r/multiarch/ubuntu-core/tags/
FROM multiarch/ubuntu-core:$arch-bionic

RUN apt-get update
RUN apt-get install -y \
    git-core \
    g++-8 \
    cmake \
    python \
    python2.7 \
    python2.7-dev \
    python-numpy \
    libyaml-dev \
    python-pip

RUN pip install --upgrade setuptools
RUN pip install wheel

ENV CC gcc-8
ENV CXX g++-8

ADD . /usr/local/src/nupic.cpp
WORKDIR /usr/local/src/nupic.cpp

# Explicitly specify --cache-dir, --build, and --no-clean so that build
# artifacts may be extracted from the container later.  Final built python
# packages can be found in /usr/local/src/nupic.cpp/bindings/py/dist
RUN pip install \
#        --cache-dir /usr/local/src/nupic.cpp/pip-cache \
#        --build /usr/local/src/nupic.cpp/pip-build \
#        --no-clean \
        -r bindings/py/packaging/requirements.txt
RUN python setup.py install

