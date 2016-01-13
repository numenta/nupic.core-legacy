# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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



def checkImportBindingsInstalled():
  """Check if nupic.bindings can be imported.

  :returns: None if successful, the import error message if not succesful.
  """
  try:
    import nupic.bindings
  except ImportError as e:
    return e.message
  else:
    return None



def checkImportBindingsExtensions():
  """Check if bindings libraries can be imported.

  :returns: None if successful, the import error message if not succesful.
  """
  try:
    import nupic.bindings.math
    import nupic.bindings.algorithms
    import nupic.bindings.engine_internal
  except ImportError as e:
    return e.message
  else:
    return None



def checkMain():
  """
  This script performs two checks. First it tries to import nupic.bindings
  to check that it is correctly installed. Then it tries to import the C
  extensions under nupic.bindings. Appropriate user-friendly status messages 
  are printed depend on the outcome.
  """
  msg = checkImportBindingsInstalled()
  if msg is not None:
    print ("Could not import nupic.bindings. It must be installed before use. "
           "Error message:{}{}".format(os.linesep, msg))
    return

  msg = checkImportBindingsExtensions()
  if msg is not None:
    print ("Could not import C extensions for nupic.bindings. Make sure that "
           "the package was properly installed. Error message:{}{}".format(
               os.linesep, msg))
    return

  print "NuPIC bindings could be successfully imported."
