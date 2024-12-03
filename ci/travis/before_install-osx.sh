#!/bin/bash
# Copyright 2013-2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

echo
echo Running before_install-osx.sh...
echo

if [ $CC = 'gcc' ]; then
    export CC='gcc-4.8'
    export CXX='g++-4.8'
fi

if [ $CC = 'clang' ]; then
    export CC='clang'
    export CXX='clang++'
fi

export PATH=$HOME/Library/Python/2.7/bin:$PATH
export PYTHONPATH=$HOME/Library/Python/2.7/lib/python/site-packages:$PYTHONPATH

echo "Installing pip, setuptools, and wheel"
curl --silent --show-error --retry 5 -O http://releases.numenta.org/pip/1ebd3cb7a5a3073058d0c9552ab074bd/get-pip.py
python get-pip.py --user pip setuptools wheel

echo "Installing Python dependencies"
pip install --user -r bindings/py/requirements.txt --quiet || exit

pip install pycapnp==0.6.3 --user || exit
