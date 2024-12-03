#!/usr/bin/env python
# Copyright 2013 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit tests for array algorithms."""

import unittest

import numpy

from nupic.bindings.math import nearlyZeroRange



class TestArrayAlgos(unittest.TestCase):


  def setUp(self):
    self.x = numpy.zeros((10))


  def testNearlyZeroRange1(self):
    self.assertTrue(nearlyZeroRange(self.x))


  def testNearlyZeroRange2(self):
    self.assertTrue(nearlyZeroRange(self.x, 1e-8))


  def testNearlyZeroRange3(self):
    self.assertTrue(nearlyZeroRange(self.x, 2))



if __name__ == '__main__':
  unittest.main()
