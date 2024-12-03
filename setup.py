# Copyright 2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""
Based on: http://stackoverflow.com/questions/5317672/pip-not-finding-setup-file
"""

import os

from setuptools.command import egg_info

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
filename = os.path.basename(__file__)

os.chdir(os.path.join(REPO_DIR, "bindings/py"))
setupdir = os.getcwd()

egginfo = "pip-egg-info"

if not os.path.exists(egginfo) and os.path.exists(os.path.join("../..", egginfo)):
  print "Symlinking pip-egg-info"
  os.symlink(os.path.join("../..", egginfo), os.path.join(REPO_DIR, "bindings/py", egginfo))

__file__ = os.path.join(setupdir, filename)

def replacement_run(self):
  self.mkpath(self.egg_info)

  installer = self.distribution.fetch_build_egg

  for ep in egg_info.iter_entry_points('egg_info.writers'):
    # require=False is the change we're making from pip
    writer = ep.load(require=False)

    if writer:
      writer(self, ep.name, egg_info.os.path.join(self.egg_info,ep.name))

  self.find_sources()

egg_info.egg_info.run = replacement_run
execfile(__file__)
