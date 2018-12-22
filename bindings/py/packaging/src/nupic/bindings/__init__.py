# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

"""
## @file
"""
import os,sys

def import_helper(name):
  from os.path import dirname
  import imp
  
      # Fast path: see if the module has already been imported.
  #print("name={}".format(name))
  try:
    return sys.modules[name]
  except KeyError:
    pass
		
  fp = None
  _mod = None
  try:
    basename = name[15:]
    fp, pathname, description = imp.find_module(basename, [dirname(__file__)])
  finally:
    if fp is None:
      print("name={} no import libarary found".format(name))
    else:
      try:
	    #print("name={}, pathname={}".format(name, pathname))
  	    _mod = imp.load_module(name, fp, pathname, description)
      finally:
        # Since we may exit via an exception, close fp explicitly.
        if fp:
            fp.close()
  return _mod;
  
algorithms = import_helper('nupic.bindings.algorithms')
engine_internal = import_helper('nupic.bindings.engine_internal')
math = import_helper('nupic.bindings.math')

