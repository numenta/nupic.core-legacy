# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
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

"""Topology unit tests"""

import unittest

from htm.bindings.math import (DefaultTopology,
                                 NoTopology,
                                 coordinatesFromIndex,
                                 indexFromCoordinates,
                                 neighborhood,
                                 wrappingNeighborhood)

from htm.bindings.math import Random
from htm.bindings.sdr import SDR

class TestTopology(unittest.TestCase):

  def testDefaultTopology(self):
    rng     = Random( 42 )
    presyn  = SDR([10,10])
    postsyn = SDR([10,10])
    postsyn.dense[0,4] = 1
    postsyn.dense = postsyn.dense
    noWrap = DefaultTopology( 1.0, 1.1, False )
    noWrapPP = noWrap( postsyn, presyn.dimensions, rng )
    assert( noWrapPP.dimensions == presyn.dimensions )
    assert( set(noWrapPP.sparse) == set([ 3, 4, 5, 13, 14, 15 ]))

    wrap   = DefaultTopology( 1.0, 1.1, True )
    wrapPP = wrap( postsyn, presyn.dimensions, rng )
    assert( set(wrapPP.sparse) == set([ 3, 4, 5, 13, 14, 15, 93, 94, 95 ]))

    potentialPct = DefaultTopology( 0.5, 1.1, False )
    assert( potentialPct( postsyn, presyn.dimensions, rng ).getSum() == 3 )

  def testNoTopology(self):
    rng = Random( 42 )
    presyn  = SDR([5,6,7,8])
    postsyn = SDR([6,5,4,3,2,1])
    postsyn.randomize( 1. / postsyn.size )
    for sparsity in (0., 1., 1. / presyn.size ):
      function = NoTopology( sparsity )
      pp = function( postsyn, presyn.dimensions, rng )
      assert( pp.dimensions == presyn.dimensions )
      assert( abs(pp.getSparsity() - sparsity) < (.25 / presyn.size))

  def testIndexFromCoordinates(self):
    self.assertEqual(0, indexFromCoordinates((0,), (100,)))
    self.assertEqual(50, indexFromCoordinates((50,), (100,)))
    self.assertEqual(99, indexFromCoordinates((99,), (100,)))

    self.assertEqual(0, indexFromCoordinates((0, 0), (100, 80)))
    self.assertEqual(10, indexFromCoordinates((0, 10), (100, 80)))
    self.assertEqual(80, indexFromCoordinates((1, 0), (100, 80)))
    self.assertEqual(90, indexFromCoordinates((1, 10), (100, 80)))

    self.assertEqual(0, indexFromCoordinates((0, 0, 0), (100, 10, 8)))
    self.assertEqual(7, indexFromCoordinates((0, 0, 7), (100, 10, 8)))
    self.assertEqual(8, indexFromCoordinates((0, 1, 0), (100, 10, 8)))
    self.assertEqual(80, indexFromCoordinates((1, 0, 0), (100, 10, 8)))
    self.assertEqual(88, indexFromCoordinates((1, 1, 0), (100, 10, 8)))
    self.assertEqual(89, indexFromCoordinates((1, 1, 1), (100, 10, 8)))


  def testCoordinatesFromIndex(self):
    self.assertEqual([0], coordinatesFromIndex(0, [100]));
    self.assertEqual([50], coordinatesFromIndex(50, [100]));
    self.assertEqual([99], coordinatesFromIndex(99, [100]));

    self.assertEqual([0, 0], coordinatesFromIndex(0, [100, 80]));
    self.assertEqual([0, 10], coordinatesFromIndex(10, [100, 80]));
    self.assertEqual([1, 0], coordinatesFromIndex(80, [100, 80]));
    self.assertEqual([1, 10], coordinatesFromIndex(90, [100, 80]));

    self.assertEqual([0, 0, 0], coordinatesFromIndex(0, [100, 10, 8]));
    self.assertEqual([0, 0, 7], coordinatesFromIndex(7, [100, 10, 8]));
    self.assertEqual([0, 1, 0], coordinatesFromIndex(8, [100, 10, 8]));
    self.assertEqual([1, 0, 0], coordinatesFromIndex(80, [100, 10, 8]));
    self.assertEqual([1, 1, 0], coordinatesFromIndex(88, [100, 10, 8]));
    self.assertEqual([1, 1, 1], coordinatesFromIndex(89, [100, 10, 8]));


  # ===========================================================================
  # NEIGHBORHOOD
  # ===========================================================================


  def expectNeighborhoodIndices(self, centerCoords, radius, dimensions, expected):
    centerIndex = indexFromCoordinates(centerCoords, dimensions)

    numIndices = 0

    for index, expectedIndex in zip(neighborhood(centerIndex, radius,
                                                 dimensions),
                                    expected):
      numIndices += 1
      self.assertEqual(index, expectedIndex)

    self.assertEqual(numIndices, len(expected))


  def expectNeighborhoodCoords(self, centerCoords, radius, dimensions, expected):
    centerIndex = indexFromCoordinates(centerCoords, dimensions)

    numIndices = 0

    for index, expectedIndex in zip(neighborhood(centerIndex, radius,
                                                 dimensions),
                                    expected):
      numIndices += 1
      self.assertEqual(index, indexFromCoordinates(expectedIndex, dimensions))

    self.assertEqual(numIndices, len(expected))


  def testNeighborhoodOfOrigin1D(self):
    self.expectNeighborhoodIndices(
      centerCoords = (0,),
      dimensions = (100,),
      radius = 2,
      expected = (0, 1, 2))


  def testNeighborhoodOfOrigin2D(self):
    self.expectNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (100, 80),
      radius = 2,
      expected = ((0, 0), (0, 1), (0, 2),
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1), (2, 2)))


  def testNeighborhoodOfOrigin3D(self):
    self.expectNeighborhoodCoords(
      centerCoords = (0, 0, 0),
      dimensions = (100, 80, 60),
      radius = 1,
      expected = ((0, 0, 0), (0, 0, 1),
                  (0, 1, 0), (0, 1, 1),
                  (1, 0, 0), (1, 0, 1),
                  (1, 1, 0), (1, 1, 1)))


  def testNeighborhoodInMiddle1D(self):
    self.expectNeighborhoodIndices(
      centerCoords = (50,),
      dimensions = (100,),
      radius = 1,
      expected = (49, 50, 51))


  def testNeighborhoodOfMiddle2D(self):
    self.expectNeighborhoodCoords(
      centerCoords = (50, 50),
      dimensions = (100, 80),
      radius = 1,
      expected = ((49, 49), (49, 50), (49, 51),
                  (50, 49), (50, 50), (50, 51),
                  (51, 49), (51, 50), (51, 51)))


  def testNeighborhoodOfEnd2D(self):
    self.expectNeighborhoodCoords(
      centerCoords = (99, 79),
      dimensions = (100, 80),
      radius = 2,
      expected = ((97, 77), (97, 78), (97, 79),
                  (98, 77), (98, 78), (98, 79),
                  (99, 77), (99, 78), (99, 79)))


  def testNeighborhoodWiderThanWorld(self):
    self.expectNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (3, 2),
      radius = 3,
      expected = ((0, 0), (0, 1),
                  (1, 0), (1, 1),
                  (2, 0), (2, 1)))


  def testNeighborhoodRadiusZero(self):
    self.expectNeighborhoodIndices(
      centerCoords = (0,),
      dimensions = (100,),
      radius = 0,
      expected = (0,))

    self.expectNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (100, 80),
      radius = 0,
      expected = ((0, 0),))

    self.expectNeighborhoodCoords(
      centerCoords = (0, 0, 0),
      dimensions = (100, 80, 60),
      radius = 0,
      expected = ((0, 0, 0),))


  def testNeighborhoodDimensionOne(self):
    self.expectNeighborhoodCoords(
      centerCoords = (5, 0),
      dimensions = (10, 1),
      radius = 1,
      expected = ((4, 0), (5, 0), (6, 0)))

    self.expectNeighborhoodCoords(
      centerCoords = (5, 0, 0),
      dimensions = (10, 1, 1),
      radius = 1,
      expected = ((4, 0, 0), (5, 0, 0), (6, 0, 0)))


  # ===========================================================================
  # WRAPPING NEIGHBORHOOD
  # ===========================================================================

  def expectWrappingNeighborhoodIndices(self, centerCoords, radius, dimensions,
                                        expected):
    centerIndex = indexFromCoordinates(centerCoords, dimensions)

    numIndices = 0

    for index, expectedIndex in zip(wrappingNeighborhood(centerIndex, radius,
                                                         dimensions),
                                    expected):
      numIndices += 1
      self.assertEqual(index, expectedIndex)

    self.assertEqual(numIndices, len(expected))


  def expectWrappingNeighborhoodCoords(self, centerCoords, radius, dimensions,
                                       expected):
    centerIndex = indexFromCoordinates(centerCoords, dimensions)

    numIndices = 0

    for index, expectedIndex in zip(wrappingNeighborhood(centerIndex, radius,
                                                         dimensions),
                                    expected):
      numIndices += 1
      self.assertEqual(index, indexFromCoordinates(expectedIndex, dimensions))

    self.assertEqual(numIndices, len(expected))


  def testWrappingNeighborhoodOfOrigin1D(self):
    self.expectWrappingNeighborhoodIndices(
      centerCoords = (0,),
      dimensions = (100,),
      radius = 1,
      expected = (99, 0, 1))


  def testWrappingNeighborhoodOfOrigin2D(self):
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (100, 80),
      radius = 1,
      expected = ((99, 79), (99, 0), (99, 1),
                  (0, 79), (0, 0), (0, 1),
                  (1, 79), (1, 0), (1, 1)))


  def testWrappingNeighborhoodOfOrigin3D(self):
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (0, 0, 0),
      dimensions = (100, 80, 60),
      radius = 1,
      expected = ((99, 79, 59), (99, 79, 0), (99, 79, 1),
                  (99, 0, 59), (99, 0, 0), (99, 0, 1),
                  (99, 1, 59), (99, 1, 0), (99, 1, 1),
                  (0, 79, 59), (0, 79, 0), (0, 79, 1),
                  (0, 0, 59), (0, 0, 0), (0, 0, 1),
                  (0, 1, 59), (0, 1, 0), (0, 1, 1),
                  (1, 79, 59), (1, 79, 0), (1, 79, 1),
                  (1, 0, 59), (1, 0, 0), (1, 0, 1),
                  (1, 1, 59), (1, 1, 0), (1, 1, 1),))


  def testWrappingNeighborhoodInMiddle1D(self):
    self.expectWrappingNeighborhoodIndices(
      centerCoords = (50,),
      dimensions = (100,),
      radius = 1,
      expected = (49, 50, 51))


  def testWrappingNeighborhoodOfMiddle2D(self):
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (50, 50),
      dimensions = (100, 80),
      radius = 1,
      expected = ((49, 49), (49, 50), (49, 51),
                  (50, 49), (50, 50), (50, 51),
                  (51, 49), (51, 50), (51, 51)))


  def testWrappingNeighborhoodOfEnd2D(self):
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (99, 79),
      dimensions = (100, 80),
      radius = 1,
      expected = ((98, 78), (98, 79), (98, 0),
                  (99, 78), (99, 79), (99, 0),
                  (0, 78), (0, 79), (0, 0)))


  def testWrappingNeighborhoodWiderThanWorld(self):
    # The order is weird because it starts walking from {-3, -3} and avoids
    # walking the same point twice.
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (3, 2),
      radius = 3,
      expected = ((0, 1), (0, 0),
                  (1, 1), (1, 0),
                  (2, 1), (2, 0)))


  def testWrappingNeighborhoodRadiusZero(self):
    self.expectWrappingNeighborhoodIndices(
      centerCoords = (0,),
      dimensions = (100,),
      radius = 0,
      expected = (0,))

    self.expectWrappingNeighborhoodCoords(
      centerCoords = (0, 0),
      dimensions = (100, 80),
      radius = 0,
      expected = ((0, 0),))

    self.expectWrappingNeighborhoodCoords(
      centerCoords = (0, 0, 0),
      dimensions = (100, 80, 60),
      radius = 0,
      expected = ((0, 0, 0),))


  def testWrappingNeighborhoodDimensionOne(self):
    self.expectWrappingNeighborhoodCoords(
      centerCoords = (5, 0),
      dimensions = (10, 1),
      radius = 1,
      expected = ((4, 0), (5, 0), (6, 0)))

    self.expectWrappingNeighborhoodCoords(
      centerCoords = (5, 0, 0),
      dimensions = (10, 1, 1),
      radius = 1,
      expected = ((4, 0, 0), (5, 0, 0), (6, 0, 0)))
