 
#!/bin/bash

# script to perform deployments for GitHub Release and PYPI
# It is used for both travis and circleci.
# First argument is the tag.
export TAG=$1

# Assumes pwd is at top of registry.
# assumes release was initiated in GitHub Releases. (it sets tag)
# assumes environment variable $GITHUB_TOKEN has the API Authentication token for GitHub

# Deploy to GitHub
python ci/deploy-artifact-to-GiHub.py $TAG "build/scripts/*.tar.gz" $GITHUB_TOKEN

# deploy to PYPI
pushd build/Release/distr
export TWINE_USERNAME="__token__"   # token for PYPI test account for David Keeney.  Not for production
export TWINE_PASSWORD="pypi-AgENdGVzdC5weXBpLm9yZwIkOTk0YmZjNGYtZTgxNS00Yjk2LTg5ZTAtODE1MGI4MjZhNGZlAAIleyJwZXJtaXNzaW9ucyI6ICJ1c2VyIiwgInZlcnNpb24iOiAxfQAABiDXJOuxvodsEDoD5dOH-e0td1DdUSwrl2NCl_lP_vy6RA"
export TWINE_REPOSITORY_URL="https://test.pypi.org/legacy/"
twine upload --skip-existing dist/*.whl
popd
