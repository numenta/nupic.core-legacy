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
    predictiveCells = SDR([100, 30])

    activeCols.sparse = [0, 2]
    predictiveCells.sparse = [0, 29, 60, 89]  # this means cells in column 0 and 2

    score = an.calculateRawAnomaly(activeCols, predictiveCells)

    self.assertEqual(score, 0.0)

  def testFullAnomaly(self):
    """
    Anomaly score is equal to 1.0, if none of the active columns
    were overlapping with the columns containing predictive cells
    """

    activeCols = SDR(100)
    predictiveCells = SDR([100, 30])

    activeCols.sparse = [0, 2]
    predictiveCells.sparse = [90, 95, 160, 300]  # this means cells in column 0 and 2

    score = an.calculateRawAnomaly(activeCols, predictiveCells)

    self.assertEqual(score, 1.0)

  def testHalfAnomaly(self):
    """
    Anomaly score is equal to 0.5, if half of the active columns
    were overlapping with the columns containing predictive cells
    """

    activeCols = SDR(100)
    predictiveCells = SDR([100, 30])

    activeCols.sparse = [20, 5]
    predictiveCells.sparse = [20*30, 20*30+2, 950, 10, 8, 9, 0] # only cells (20*30) and (20*30+2) are matching with active col 20

    score = an.calculateRawAnomaly(activeCols, predictiveCells)

    self.assertEqual(score, 0.5)



if __name__ == "__main__":
  unittest.main()
