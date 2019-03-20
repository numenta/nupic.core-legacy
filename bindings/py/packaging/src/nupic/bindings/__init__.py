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

def _import_helper(name):
  import sys
  from os.path import dirname, join

  # Fast path: see if the module has already been imported.
  #print("name={}".format(name))
  try:
    return sys.modules[name]
  except KeyError:
    pass

  _mod = None
  basename = name[15:]
  if sys.version_info[0]+sys.version_info[1]/10 <= 3.4 :	
    # For Python 2.7
    import imp
    fp = None
    try:
      fp, pathname, description = imp.find_module(basename, [dirname(__file__)])
    finally:
      if fp is None:
        print("name={} no import library found".format(name))
      else:
        try:
          print("name={}, pathname={}".format(name, pathname))
          _mod = imp.load_module(name, fp, pathname, description)
        finally:
          # Since we may exit via an exception, close fp explicitly.
          if fp:
              fp.close()
  else:
    # For Python 3.5+
    import importlib.util
    import glob
    spec = None
    filename = glob.glob(join(dirname(__file__), basename +"*"))
    print("filename={}".format(filename))
    if filename:
      spec = importlib.util.spec_from_file_location(name, filename[0])	
    if spec is None:
      print("name=nupic.bindings.{} no import library found".format(name))
    else:
      _mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(_mod);
      sys.modules[name] = _mod
  return _mod;

sdr             = _import_helper('nupic.bindings.sdr')
encoders        = _import_helper('nupic.bindings.encoders')
algorithms      = _import_helper('nupic.bindings.algorithms')
engine_internal = _import_helper('nupic.bindings.engine_internal')
math            = _import_helper('nupic.bindings.math')
