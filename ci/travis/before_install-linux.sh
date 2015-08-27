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

echo "Installing Cap'n Proto..."
curl -O https://capnproto.org/capnproto-c++-0.5.2.tar.gz
tar zxf capnproto-c++-0.5.2.tar.gz
pushd capnproto-c++-0.5.2
./configure --prefix=${HOME}/.local
make
make install
popd

export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

echo "PATH"
echo $PATH
echo "LDFLAGS"
echo $LDFLAGS
echo "CPPFLAGS"
echo $CPPFLAGS
echo "LD_LIBRARY_PATH"
echo $LD_LIBRARY_PATH
echo "LD_RUN_PATH"
echo $LD_RUN_PATH
echo "which cc"
which cc
echo "CC"
echo $CC

echo "Installing latest pip"
pip install --ignore-installed --user setuptools
pip install --ignore-installed --user pip
echo "PYTHONPATH"
echo $PYTHONPATH
echo $HOME/.local/lib/python2.7/site-packages
ls $HOME/.local/lib/python2.7/site-packages/
echo "$HOME/.local/bin/pip --version"
$HOME/.local/bin/pip --version
echo "which pip"
which pip
python -c "import pip; print pip.__version__, pip.__file__"
pip --version
echo $HOME
ls $HOME/.local
echo "ls .local/bin:"
ls $HOME/.local/bin
echo "ls .local/lib/python2.7/site-packages:"
ls $HOME/.local/lib/python2.7/site-packages

echo "Installing wheel..."
pip install wheel --user || exit
echo "Installing Python dependencies"
pip install --user pycapnp==0.5.7 --install-option="--force-system-libcapnp"
pip install --use-wheel --user -r bindings/py/requirements.txt || exit

pip install cpp-coveralls --user
