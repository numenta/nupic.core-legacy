#!/bin/bash

# script to fetch artifacts from Win,OSX CI to Travis,
# where all 3 files are deployed to GitHub for release. 
# The curent pwd should be build/scripts when running this script.


VERSION=`cat ../../VERSION`
echo "Released version: $VERSION"
echo

TRAVIS_FILE="htm_core-$VERSION-linux64.tar.gz"


echo
echo "Fetching Windows artifacts:"
APPVEYOR_FILE="htm_core-$VERSION-windows64.zip"
APPVEYOR_ARTIFACT_URL="https://ci.appveyor.com/api/projects/htm-community/htm.core/artifacts/build/scripts/$APPVEYOR_FILE?branch=master" 

echo "Fetching Windows artifacts:" 
# Perform authentication

for i in `seq 10`; do
  echo "  Downloading artifacts. Try number $i"
  # get the headers from the query for the latest master build.  
  # This is most likely a redirect to the real URL of the artifact.
  # So, Locate the line for "location:" and use the word in quotes as the real URL.
  # Then we can use the -O option for the filename of the file it found.
  page=`curl -v -X GET -I -H "Authorization: Bearer $TOKEN_APPVEYOR" $APPVEYOR_ARTIFACT_URL 2>/dev/null`
echo "  $page \n"
  newurl=`echo $page | grep Location: | sed -e 's/^.*: "//' -e 's/".*//'`
  if [[ ! -z "$newurl" ]]; then
    # We have one, fetch the file
    curl -O $newurl
    ex=$?
    if test "$ex" == "0"; then  
      echo "  happy"
      ls *.zip
      break
    fi
  else
    echo "URL empty."
  fi
  echo "  sleep for 5min is good"
  sleep 300 # wait 5min to retry. Other CI needs to finish its build
done


# same code, just for CircleCI
CIRCLE_FILE="htm_core-$VERSION-darwin64.tar.gz"
CIRCLE_URL="https://circleci.com/api/v1.1/project/github/htm-community/htm.core/latest/artifacts?circle-token=$TOKEN_CIRCLE&limit=1&branch=master&filter=successful"
echo
echo "Fetching OSX artifacts from CircleCI:  $CIRCLE_FILE \n"
# echo "URL: $CIRCLE_URL \n" #do not expose the TOKEN

for i in `seq 10`; do
  echo "  Get latest CircleCI artifact, try $i"
  page=`curl $CIRCLE_URL`
  ex=$?
  if test "$ex" == "0"; then
    echo "Page: $page"
    newurl=`echo $page | grep $'"url" :' | sed -e 's/^.*: "//' -e 's/".*//'`
    echo "Artifact URL: $newurl"
    curl -O $newurl
    ex=$?
    if test "$ex" == "0"; then
      echo "  happy"
      ls *.gz
      break
    fi
  fi
  echo "  the curl command failed with: $ex"
  echo "  sleep for 5min is good"
  sleep 300 # wait 5min to retry. Other CI needs to finish its build
done


# extract the wheel files for deploying to PYPI
mkdir artifacts
cd artifacts
unzip ../$APPVEYOR_FILE
tar -xf ../$TRAVIS_FILE
tar -xf ../$CIRCLE_FILE
cp */py/*.whl ..
cd ..
echo "Python wheels ready for deployment:"
ls *.whl
echo
#rm -r artifacts

