# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Zbysek Zapadlik
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
Test the anomaly score class.
"""
# python 2to3 compatibility fix
from __future__ import print_function #fixes "print('', file=xxx)" invalid syntax error in py2

import copy
import datetime
import unittest

from htm.algorithms.anomaly import Anomaly as an
from htm.bindings.sdr import SDR
from unittest import TestCase as TestCaseBase


class AnomalyTest(TestCaseBase):

  def testZeroAnomaly(self):
    """
    Anomaly score is equal to 0.0, if all active columns
    were overlapping with the columns containing predictive cells
    """

    activeCols = SDR(100)
    predictiveCols = SDR(100)

    activeCols.sparse = [0, 2, 89, 99]
    predictiveCols.sparse = [0, 2, 89, 99]

    score = an.calculateRawAnomaly(activeCols, predictiveCols)

    self.assertEqual(score, 0.0)

  def testFullAnomaly(self):
    """
    Anomaly score is equal to 1.0, if none of the active columns
    were overlapping with the columns containing predictive cells
    """

    activeCols = SDR(100)
    predictiveCols = SDR(100)

    activeCols.sparse = [60, 69, 80, 91]
    predictiveCols.sparse = [56, 95, 68, 2]

    score = an.calculateRawAnomaly(activeCols, predictiveCols)

    self.assertEqual(score, 1.0)

  def testHalfAnomaly(self):
    """
    Anomaly score is equal to 0.5, if half of the active columns
    were overlapping with the columns containing predictive cells
    """

    activeCols = SDR(100)
    predictiveCols = SDR(100)

    activeCols.sparse = [60, 69, 80, 91]
    predictiveCols.sparse = [60, 80, 55, 1]

    score = an.calculateRawAnomaly(activeCols, predictiveCols)

    self.assertEqual(score, 0.5)

  def testWrongDim(self):
    """
    Tests given wrong dimensions of SDRs
    """

    activeCols = SDR(50)
    predictiveCols = SDR(40)

    self.assertRaises(ValueError, an.calculateRawAnomaly, activeCols, predictiveCols)

if __name__ == "__main__":
  unittest.main()
