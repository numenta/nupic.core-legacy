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
echo Running before_deploy.sh...
echo

set -o xtrace

# If this branch is master, this is an iterative deployment, so we'll package
# wheels ourselves for deployment to S3. No need to build docs.
if [ "${TRAVIS_BRANCH}" = "master" ]; then

    # Upgrading pip
    pip install --upgrade pip
    # Assuming pip 1.5.X+ is installed.
    pip install wheel==0.25.0 --user

    cd ${TRAVIS_BUILD_DIR}/bindings/py

    # Build all NuPIC and all required python packages into dist/wheels as .whl
    # files.
    pip wheel --wheel-dir=dist/wheels -r requirements.txt
    python setup.py bdist_wheel -d dist/wheels
    python setup.py bdist_egg -d dist
    # The dist/wheels folder is expected to be deployed to S3.

# If this is a tag, we're doing a release deployment, so we want to build docs
# for pypi...
else

    # For docs, direct people to numenta.org/docs/nupic.
    mkdir ./build/docs
    echo "<html><body>See NuPIC docs at <a href='http://numenta.org/docs/nupic/'>http://numenta.org/docs/nupic/</a>.</body></html>" > build/docs/index.html

fi
