FROM ubuntu:14.04
RUN apt-get update && \
    apt-get install -y \
    curl \
    wget \
    git-core \
    gcc \
    g++ \
    cmake \
    python \
    python2.7 \
    python2.7-dev \
    zlib1g-dev \
    bzip2 \
    libyaml-dev \
    libyaml-0-2
RUN wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py -O - | python
RUN pip install --upgrade setuptools
RUN pip install wheel
ENV CC gcc
ENV CXX g++
ENV USER docker
ADD . /usr/local/src/nupic.core
WORKDIR /usr/local/src/nupic.core
RUN pip install \
        --cache-dir /usr/local/src/nupic.core/pip-cache \
        --build /usr/local/src/nupic.core/pip-build \
        --no-clean \
        pycapnp==0.5.5 \
        -r bindings/py/requirements.txt && \
    python setup.py bdist bdist_dumb bdist_wheel sdist
