# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""Unit test for segment sparse matrix."""

import numpy as np
import unittest

from nupic.bindings.math import SegmentSparseMatrix


class SegmentSparseMatrixTest(unittest.TestCase):
  def testAddRows(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    self.assertEqual(0, ssm.matrix.nRows())

    ssm.createSegment(42)
    self.assertEqual(1, ssm.matrix.nRows())

    ssm.createSegments([42, 43, 44])
    self.assertEqual(4, ssm.matrix.nRows())


  def testNoRowLeaks(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    created = ssm.createSegments([42, 43, 44, 45, 46])
    ssm.destroySegments([created[1], created[2], created[3]])
    ssm.createSegments([50, 51, 52, 53])

    self.assertEquals(6, ssm.matrix.nRows())


  def testGetSegmentCounts(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    created = ssm.createSegments([42, 42, 43, 44, 45, 43])
    ssm.destroySegments([created[2], created[4]])

    np.testing.assert_equal(ssm.getSegmentCounts([42, 43, 44, 45, 46]),
                            [2, 1, 1, 0, 0])


  def testSortSegmentsByCell(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    segments = ssm.createSegments([42, 41, 49, 45, 0, 2047])
    segmentsSorted = [segments[4],
                      segments[1],
                      segments[0],
                      segments[3],
                      segments[2],
                      segments[5]]

    ssm.sortSegmentsByCell(segments)

    np.testing.assert_equal(segments, segmentsSorted)


  def testFilterSegmentsByCell(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    # Shuffled [42, 42, 42, 43, 46, 47, 48]
    cellsWithSegments = [47, 42, 46, 43, 42, 48, 42]

    segments = ssm.createSegments(cellsWithSegments)
    ssm.sortSegmentsByCell(segments)

    # Include everything
    everything = sorted(cellsWithSegments)
    np.testing.assert_equal(ssm.filterSegmentsByCell(segments, everything),
                            segments)

    # Subset, one cell with multiple segments
    np.testing.assert_equal(ssm.filterSegmentsByCell(segments,
                                                     [42, 43, 48]),
                            [segments[0],
                             segments[1],
                             segments[2],
                             segments[3],
                             segments[6]])

    # Subset, some cells without segments
    np.testing.assert_equal(ssm.filterSegmentsByCell(segments,
                                                     [43, 44, 45, 48]),
                            [segments[3],
                             segments[6]])


  def testMapSegmentsToCells(self):
    ssm = SegmentSparseMatrix(2048, 1000)

    segments = ssm.createSegments([42, 42, 42, 43, 44, 45])

    np.testing.assert_equal(
      ssm.mapSegmentsToCells(segments),
      [42, 42, 42, 43, 44, 45])
    np.testing.assert_equal(ssm.mapSegmentsToCells([segments[3],
                                                    segments[3],
                                                    segments[0]]),
                            [43, 43, 42])
