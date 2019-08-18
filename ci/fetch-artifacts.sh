#!/bin/bash

# script to fetch artifacts from Win,OSX CI to Travis,
# where all 3 files are deployed to GitHub for release. 
# The curent pwd should be build/scripts when running this script.


export VERSION=`cat ../../VERSION`
echo "Released version: $VERSION"
echo



#WINDOWS_ARTIFACT_URL="https://ci.appveyor.com/api/buildjobs/by4cpqmul79tp9ag/artifacts/build/scripts/htm_core-6866bd13ece35608313709e82c29b2260ee7be73-windows64.zip?branch=master" #FIXME make the URLs automated, needs API?
#OSX_ARTIFACT_URL="https://3238-118031881-gh.circle-artifacts.com/0/tmp/uploads/$CIRCLE_FILE"


echo
echo "Fetching Windows artifacts:"
export APPVEYOR_TOKEN="v2.5tido3qf80n33le5bi3e"  # token by David Keeney.  TODO: replace with project token.  Needs Admin permissions.
export APPVEYOR_FILE="htm_core-$VERSION-windows64.zip"
export APPVEYOR_ARTIFACT_URL="https://ci.appveyor.com/api/projects/htm-community/htm.core/artifacts/build/scripts/$APPVEYOR_FILE?branch=master" 
#export APPVEYOR_ARTIFACT_URL="https://ci.appveyor.com/api/projects/htm-community/htm.core/branch/master" 

echo "Fetching Windows artifacts:" 
# Perform authentication

if false; then
for i in `seq 10`; do
  echo "  Downloading artifacts. Try number $i"
  # get the headers from the query for the latest master build.  
  # This is most likely a redirect to the real URL of the artifact.
  # So, Locate the line for "location" and use the second word as the real URL.
  # Then we can use the -O option for the filename of the file it found.
  page=`curl -v -X GET -I -H "Authorization: Bearer $APPVEYOR_TOKEN" $APPVEYOR_ARTIFACT_URL 2>/dev/null`
echo "  $page \n"
  line=`echo $page | grep Location: | tr " " "\n"`
  NEWURL=`echo ${line[1]} | tr '' ' '`
  if [ -z "$NEWURL" ]; then
    echo "  Did not get a URL."
  else
    # We have one, remove windows line endings from the URL and fetch the file
    NEWURL=${NEWURL%$'\r'} 
    curl -v -O $NEWURL
    ex=$?
    if test "$ex" != "0"; then  
      echo "  happy"
      ls *.zip
      break
    fi
  fi
  echo "  sleep for 5min is good"
  sleep 300 # wait 5min to retry. Other CI needs to finish its build
done
fi

echo
echo "Fetching OSx artifacts from CircleCI:"
export CIRCLE_FILE="htm_core-$VERSON-darwin64.tar.gz"
export CIRCLE_TOKEN='?circle-token=e4ec350f46372b0f9e0dc0f35453ee8c55d98838'    #token by david keeney  TODO, replace with project token.   Needs Admin permissions.
export CIRCLE_URL="https://circleci.com/api/v1.1/project/github/htm-community/htm.core/latest/artifacts?limit=1&branch=master&filter=successful"

for i in `seq 10`; do
  echo "  Get latest CircleCI artifact, try $i"
  curl -v -J -o $CIRCLE_FILE -H "Authorization: Bearer $CIRCLE_TOKEN" $CIRCLE_URL 
  ex=$?
  if test "$ex" != "0"; then
    echo "  the curl command failed with: $ex"
    echo "  sleep for 5min is good"
    sleep 300 # wait 5min to retry. Other CI needs to finish its build
  else
    echo "  happy"
    ls *.gz
    break
  fi
done


# extract the wheel files for deploying to PYPI
mkdir artifacts
cd artifacts
unzip ../*windows64.zip
tar -xvf ../*linux64.tar.gz
tar -xvf ../*darwin64.tar.gz
cp */py/*.whl ..
cd ..
echo "Python wheels ready for deployment:"
ls *.whl
echo
#rm -r artifacts

