# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2015, Numenta, Inc.
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



def checkImportBindingsInstalled():
  """Check that htm.bindings is installed (can be imported).

  Throws ImportError on failure.
  """
  import htm.bindings



def checkImportBindingsExtensions():
  """Check that htm.bindings extension libraries can be imported.

  Throws ImportError on failure.
  """
  import htm.bindings.math
  import htm.bindings.algorithms
  import htm.bindings.engine_internal
  import htm.bindings.encoders
  import htm.bindings.sdr



def checkMain():
  """
  This script performs two checks. First it tries to import htm.bindings
  to check that it is correctly installed. Then it tries to import the C
  extensions under htm.bindings. Appropriate user-friendly status messages
  are printed depend on the outcome.
  """
  try:
    checkImportBindingsInstalled()
  except ImportError as e:
    print ("Could not import htm.bindings. It must be installed before use. "
           "Error message:")
    print(e.message)
    return

  try:
    checkImportBindingsExtensions()
  except ImportError as e:
    print ("Could not import C extensions for htm.bindings. Make sure that "
           "the package was properly installed. Error message:")
    print(e.message)
    return

  print("Successfully imported htm.bindings.")
