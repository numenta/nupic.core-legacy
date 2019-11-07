# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2014-2015, Numenta, Inc.
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

import unittest
import sys

from htm.bindings.regions.PyRegion import PyRegion



# Classes used for testing

class X(PyRegion):
  def __init__(self):
    self.x = 5



class Y(PyRegion):
  def __init__(self):
    self.zzz = 5
    self._zzz = 3
  def initialize(self): pass
  def compute(self): pass
  def getOutputElementCount(self): pass



class Z(object):
  def __init__(self):
    y = Y()
    y.setParameter('zzz', 0, 4)



class PyRegionTest(unittest.TestCase):




  def testUnimplementedNotImplementedMethods(self):
    """Test unimplemented @not_implemented methods"""
    # Can instantiate because all abstract methods are implemented
    y = Y()

    # Can call the default getParameter() from PyRegion
    self.assertEqual(y.getParameter('zzz', -1), 5)

    # Accessing an attribute whose name starts with '_' via getParameter()
    with self.assertRaises(Exception) as cw:
      _ = y.getParameter('_zzz', -1) == 5

    self.assertEqual(str(cw.exception), "Parameter name must not " +
      "start with an underscore")

    # Calling not implemented method result in NotImplementedError
    with self.assertRaises(NotImplementedError) as cw:
      y.setParameter('zzz', 4, 5)

    self.assertEqual(str(cw.exception),
                     "The method setParameter is not implemented.")

  def testCallUnimplementedMethod(self):
    """Test calling an unimplemented method"""
    with self.assertRaises(NotImplementedError) as cw:
      _z = Z()

    self.assertEqual(str(cw.exception),
                     "The method setParameter is not implemented.")

  def testPickle(self):
    """
    Test region pickling/unpickling.
    """
    y = Y()

    if sys.version_info[0] >= 3:
      import pickle
      proto = 3
    else:
      import cpickle as pickle
      proto = 2

    # Simple test: make sure that dumping / loading works...
    pickledRegion = pickle.dumps(y, proto)
    y2 = pickle.loads(pickledRegion)
    self.assertEqual(y.zzz, y2.zzz,  "Simple Region pickle/unpickle failed.")
    



if __name__ == "__main__":
  unittest.main()
