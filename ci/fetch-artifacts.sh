#!/bin/bash

# script to fetch artifacts from Win,OSX CI to Travis,
# where all 3 files are deployed to GitHub for release. 
# curent pwd should be build/scripts


echo "Released version: "
cat ../../VERSION
echo

WINDOWS_ARTIFACT_URL="https://ci.appveyor.com/api/buildjobs/by4cpqmul79tp9ag/artifacts/build/scripts/htm_core-6866bd13ece35608313709e82c29b2260ee7be73-windows64.zip" #FIXME make the URLs automated, needs API?
OSX_ARTIFACT_URL="https://3238-118031881-gh.circle-artifacts.com/0/tmp/uploads/htm_core-7b7b53c798b3795088ce772e96c6994068fc1302-darwin64.tar.gz"

echo "Fetching Windows artifacts:" 
retry=false
for i in `seq 10`; do
  echo "Downloading artifacts. Try $i"
  wget "$WINDOWS_ARTIFACT_URL" || retry=true
  if [[ x"$retry" == "xtrue" ]]; then
    echo "sleep for 5min is good"
    sleep 300 # wait 5min to retry. Other CI needs to finish its build
    retry=false
  else
    echo "happy"
    ls *.zip
    break
  fi
done

echo "Fetching OSX artifacts:"
retry=false
for i in `seq 10`; do
  echo "Downloading artifacts. Try $i"
  wget "$OSX_ARTIFACT_URL" || retry=true
  if [[ x"$retry" == "xtrue" ]]; then
    echo "sleep for 5min is good"
    sleep 300 # wait 5min to retry. Other CI needs to finish its build
    retry=false
  else
    echo "happy"
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
echo "wheels ready for deployment"
ls *.whl
#rm -r artifacts

