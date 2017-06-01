#!/bin/bash
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-7, Numenta, Inc.  Unless you have an agreement
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
echo Running deploy_s3-osx.sh...
echo

#Organize files for upload
cp bindings/py/requirements.txt dist/
mkdir -p dist/include/nupic
mkdir release
cp build/release/include/nupic/Version.hpp dist/include/nupic/
tar -zcv -f release/nupic_core-${CIRCLE_SHA1}-darwin64.tar.gz dist


# awscli needs to be manually installed on Circle's OS X
pip install awscli --user

aws s3 cp release/nupic_core-${CIRCLE_SHA1}-darwin64.tar.gz s3://artifacts.numenta.org/numenta/nupic.core/circle/
