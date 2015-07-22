#!/bin/bash
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

echo
echo Running before_install-linux.sh...
echo

echo ">>> Doing prep work..."
sudo add-apt-repository -y ppa:fkrull/deadsnakes
sudo apt-get update

# install gcc-4.8 for C++11 compatibility, #TODO remove when Travis has gcc>=4.8, (it's used for clang too, in coveralls)
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get -qq update
sudo apt-get -qq install g++-4.8
alias gcc='gcc-4.8'        
alias g++='g++-4.8'

if [ $CC == 'gcc' ]; then
    export CC='gcc-4.8'
    export CXX='g++-4.8'
fi

if [ $CC = 'clang' ]; then
    export CXX='clang++'
fi

echo ">>> Installing virtualenv..."
sudo apt-get install python$PY_VER python$PY_VER-dev python-virtualenv cmake-data
sudo ls -laFh /usr/lib/libpython$PY_VER.so

echo "Installing wheel..."
pip install wheel --user || exit
echo "Installing numpy..."
pip install --use-wheel numpy==1.9.2 --user || exit

sudo pip install cpp-coveralls
