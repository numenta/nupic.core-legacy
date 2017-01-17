#!/bin/bash
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-5, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

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
pip install --use-wheel --user -r bindings/py/requirements.txt --quiet || exit

pip install pycapnp==0.5.8 --user || exit
