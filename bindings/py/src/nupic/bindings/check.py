# Copyright 2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.



def checkImportBindingsInstalled():
  """Check that nupic.bindings is installed (can be imported).

  Throws ImportError on failure.
  """
  import nupic.bindings



def checkImportBindingsExtensions():
  """Check that nupic.bindings extension libraries can be imported.

  Throws ImportError on failure.
  """
  import nupic.bindings.math
  import nupic.bindings.algorithms
  import nupic.bindings.engine_internal



def checkMain():
  """
  This script performs two checks. First it tries to import nupic.bindings
  to check that it is correctly installed. Then it tries to import the C
  extensions under nupic.bindings. Appropriate user-friendly status messages
  are printed depend on the outcome.
  """
  try:
    checkImportBindingsInstalled()
  except ImportError as e:
    print ("Could not import nupic.bindings. It must be installed before use. "
           "Error message:")
    print e.message
    return

  try:
    checkImportBindingsExtensions()
  except ImportError as e:
    print ("Could not import C extensions for nupic.bindings. Make sure that "
           "the package was properly installed. Error message:")
    print e.message
    return

  print "Successfully imported nupic.bindings."
