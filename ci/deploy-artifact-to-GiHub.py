# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2014-2016, Numenta, Inc.
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
# ----------------------------------------------------------------------

"""
This program is used to upload artifacts to 'GitHub Releases' as part of a distribution.

 - confirm that the GITHUB_TOKEN is valid.
 - lookup the <tag> to confirm that a release exists in 'GitHub Releases' and get its release id.
   If this is not found, it is an error.
 - lookup the <filename> of the artifact and get its asset id if it exists.
 - If it exists, delete it.
 - upload the artifact.

    
USAGE
=====
    python3 deploy-artifact-to-GiHub.py <tag> <filename> <token>
    
    - tag:  The version tag we are deploying. (Assumed that tag is created using GitHub Releases)
    - filename:  The full or reletive path to the artifact(s) to be uploaded. Wildcards ok.
    - token:  The $GITHUB_TOKEN needed for API validation.

"""

import requests
import glob
import os
import sys
import json
import re



def upload(tag, filename, token):
  url = "https://api.github.com/repos/htm-community/htm.core"
  headers = {"Authorization": "token "+token}
  
  # Verify token is usable
  res = requests.get(url, headers=headers)
  if not res.ok:    
    print("deploy-artifact-to-GitHub: token or network issue.")
    return 1
  
  # Get the release id
  res = requests.get(url+"/releases/tags/"+tag, headers=headers)
  if not res.ok:    
    print("deploy-artifact-to-GitHub: Cannot find GitHub release for tag: "+tag)
    return 1
  response = json.loads(res.content)
  #print("release: "+json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))
  release_id = response["id"];
  
  # expand the wildcards in filename
  asset_list = glob.glob(filename)
  for path in asset_list:
    name = os.path.basename(path)
      
    # look in response for this name and get its asset id
    #   if found, delete it
    for asset in response["assets"]:
      if asset["name"] == name:
        requests.delete(url+"/releases/assets/"+str(asset["id"]), headers=headers)
        print("deleted existing asset: "+name+" (id: "+str(asset["id"])+")")
         
    # Upload the artifact
    print("Uploading artifact: "+name)
    upload_url = re.sub(r"\{\?.*\}", "?name="+name, response["upload_url"])
    post_headers = {"Authorization": "token "+token,
                    "Content-Type": "application/octet-stream"}
    data = open(path, 'rb')
    res = requests.post(upload_url, data=data, headers=post_headers)
    if not ((res.status_code//100) == 2):    
      print("deploy-artifact-to-GitHub: Problem uploading artifact "+path+" in GitHub release for tag: "+tag+" status: "+str(res.status_code)+"\n")
      return 1
      
  print("deploy-artifact-to-GitHub: Success.\n")
  return 0





if __name__ == "__main__":
  tag = sys.argv[1]
  filename = sys.argv[2]
  token = sys.argv[3]
  exit (upload(tag, filename, token))
  