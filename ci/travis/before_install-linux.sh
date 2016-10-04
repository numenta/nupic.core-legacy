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
echo Running before_install-linux.sh...
echo

# install gcc-4.8 for C++11 compatibility, #TODO remove when Travis has gcc>=4.8, (it's used for clang too, in coveralls)
alias gcc='gcc-4.8'
alias g++='g++-4.8'
if [ $CC == 'gcc' ]; then
    export CC='gcc-4.8'
    export CXX='g++-4.8'
fi

if [ $CC = 'clang' ]; then
    export CC='clang'
    export CXX='clang++'
fi

export PIP_USER=--user

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

echo "Installing latest pip"
pip install --upgrade --ignore-installed --user setuptools
pip install --upgrade --ignore-installed --user pip

# Print diagnostic info to help debug why we're not picking up the latest pip
pip --version
echo "WHICH PIP: $(which pip)"
echo "WHEREIS PIP: $(whereis pip)"
/usr/local/bin/pip --version
/usr/local/bin/pip2.7 --version
echo "PATH=$PATH"

echo "Installing wheel..."
pip install wheel==0.25.0 --user || exit
echo "Installing Python dependencies"
pip install --use-wheel --user -r bindings/py/requirements.txt --quiet || exit

pip install --user pycapnp==0.5.8 || exit
pip install cpp-coveralls --user
