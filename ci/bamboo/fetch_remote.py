# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""Checks status and downloads artifacts from Travis/AppVeyor."""

import argparse
import os
import sys
import time

import requests

STATUS_URL = "http://nubot.numenta.org/hubot/ci-status?sha={sha}"

PLATFORMS = {
  "darwin": "continuous-integration/travis-ci/push",
  "windows": "continuous-integration/appveyor/branch",
}

# Time in seconds to check for completion before failing.
MAX_WAIT = 2*60*60

# Time to wait before retrying unfinished build.
RETRY_DELAY = 60



class _Status(object):
  SUCCESS = "success"
  FAILURE = "failure"
  ERROR = "error"
  PENDING = "pending"
  MISSING = "missing"



def _downloadArtifacts(artifacts, artifactDir):
  for artifactUrl in artifacts:
    artifactFilename = artifactUrl.split("/")[-1]
    outputPath = os.path.join(artifactDir, artifactFilename)

    print "Downloading artifact from URL: ", artifactUrl
    r = requests.get(artifactUrl, stream=True)
    r.raise_for_status()
    if r.status_code != 200:
      raise IOError("Received HTTP status code of {} for download: {}".format(r.status_code, artifact))

    with open(outputPath, "wb") as f:
      for chunk in r:
        f.write(chunk)



def _checkStatus(platform, sha):
  formattedUrl = STATUS_URL.format(sha=sha)
  print "Getting status from URL: {}".format(formattedUrl)
  response = requests.get(formattedUrl)
  nupicCoreStatus = response.json()["numenta/nupic.core"]

  found = nupicCoreStatus["shaMatch"]
  if not found or platform not in nupicCoreStatus["builds"]:
    status = _Status.MISSING
    artifacts = []
    return status, artifacts

  build = nupicCoreStatus["builds"][platform]
  status = build["state"]
  artifacts = build["artifacts"]
  return status, artifacts


def _parseArgs():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("--platform", required=True, choices=PLATFORMS.keys())
  parser.add_argument("--sha", required=True,
                      help="Commit SHA to fetch artifacts for.")
  parser.add_argument("--artifactDir", default="",
                      help="Directory to download artifacts to.")
  args = parser.parse_args()
  platform = PLATFORMS[args.platform]
  sha = args.sha
  artifactDir = os.path.join(os.getcwd(), args.artifactDir)
  return platform, sha, artifactDir



def main():
  platform, sha, artifactDir = _parseArgs()
  try:
    os.makedirs(artifactDir)
  except OSError:
    if not os.path.isdir(artifactDir):
      raise

  artifacts = None
  timeout = time.time() + MAX_WAIT
  while time.time() < timeout:
    print "Checking status... ",
    status, artifacts = _checkStatus(platform, sha)
    if status == _Status.SUCCESS:
      print "Build succeeded."
      break
    elif status == _Status.PENDING or status == _Status.MISSING:
      print "Build is pending or not started. Waiting for {} seconds...".format(RETRY_DELAY)
      time.sleep(RETRY_DELAY)
      continue
    elif status == _Status.FAILURE:
      print "Build failed."
      sys.exit(-1)
    elif status == _Status.ERROR:
      print "Build errored."
      sys.exit(-1)
    else:
      raise ValueError("Unknown build status: {}".format(status))

  _downloadArtifacts(artifacts, artifactDir)



if __name__ == "__main__":
  main()
