#!/usr/bin/env python
# Copyright 2013 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Cast mode test."""

import sys

import numpy
import unittest

from nupic.bindings.math import SM32



class TestCastMode(unittest.TestCase):


  @unittest.skipIf(sys.platform == "linux2",
                   "Castmode test disabled on linux -- fails")
  def testCastMode(self):
    """Test for an obscure error that is fixed by the -castmode flag to swig.

    This code will throw an exception if the error exists.
    """
    hist = SM32(5, 10)
    t = numpy.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype='float32')

    hist.setRowFromDense(0, t)
    hist.setRowFromDense(1, t)
    self.assertSequenceEqual(tuple(hist.getRow(1)),
                             (0, 0, 1, 0, 1, 0, 0, 1, 0, 1))



if __name__ == "__main__":
  unittest.main()
