#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""Unit tests for connections classes"""

import numpy as np
import unittest

from nupic.bindings.math import Random, SparseMatrixConnections

class ConnectionsTest(unittest.TestCase):

  def test_computeActivity(self):
    for (name, cells, inputs, activeInputs,
         initialPermanence,
         expected) in (("Basic test",
                        [1, 2, 3], [42, 43, 44],
                        [42, 44], 0.45,
                        [2, 2, 2]),
                       ("Small permanence",
                        [1, 2, 3], [42, 43, 44],
                        [42, 44], 0.01,
                        [2, 2, 2]),
                       ("No segments",
                        [], [42, 43, 44],
                        [42, 44], 0.45,
                        []),
                       ("No active inputs",
                        [1, 2, 3], [42, 43, 44],
                        [], 0.45,
                        [0, 0, 0])
         ):

      connections = SparseMatrixConnections(2048, 2048)
      segments = connections.createSegments(cells)
      connections.growSynapses(segments, inputs, initialPermanence)

      overlaps = connections.computeActivity(activeInputs)
      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_computeActivity_thresholded(self):
    for (name, cells, inputs, activeInputs,
         initialPermanence, connectedPermanence,
         expected) in (("Accepted",
                        [1, 2, 3], [42, 43, 44],
                        [42, 44], 0.55, 0.5,
                        [2, 2, 2]),
                       ("Rejected",
                        [1, 2, 3], [42, 43, 44],
                        [42, 44], 0.55, 0.6,
                        [0, 0, 0]),
                       ("No segments",
                        [], [42, 43, 44],
                        [42, 44], 0.55, 0.5,
                        []),
                       ("No active inputs",
                        [1, 2, 3], [42, 43, 44],
                        [], 0.55, 0.5,
                        [0, 0, 0])
         ):

      connections = SparseMatrixConnections(2048, 2048)
      segments = connections.createSegments(cells)
      connections.growSynapses(segments, inputs, initialPermanence)

      overlaps = connections.computeActivity(activeInputs, connectedPermanence)
      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_adjustSynapses(self):
    for (name, cells, inputs,
         adjustedSegments, activeInputs,
         initialPermanence, activeDelta, inactiveDelta, connectedPermanence,
         expected) in (("Basic test",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.45, 0.1, -0.1, 0.5,
                        [2, 0, 2]),
                       ("Reward inactive",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.45, -0.1, 0.1, 0.5,
                        [1, 0, 1]),
                       ("No segments",
                        [1, 2, 3], [42, 43, 44],
                        [], [42, 44],
                        0.45, 0.1, -0.1, 0.5,
                        [0, 0, 0]),
                       ("No active synapses",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [],
                        0.45, 0.1, -0.1, 0.5,
                        [0, 0, 0]),
                       ("Delta of zero",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.55, 0.0, 0.0, 0.5,
                        [3, 3, 3])
         ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(segments, inputs, initialPermanence)
      connections.adjustSynapses(segments[adjustedSegments],
                                 activeInputs, activeDelta, inactiveDelta)

      overlaps = connections.computeActivity(inputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_adjustActiveSynapses(self):
    for (name, cells, inputs,
         adjustedSegments, activeInputs,
         initialPermanence, delta, connectedPermanence,
         expected) in (("Basic test",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.45, 0.1, 0.5,
                        [2, 0, 2]),
                       ("Negative increment",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.55, -0.1, 0.5,
                        [1, 3, 1]),
                       ("No segments",
                        [1, 2, 3], [42, 43, 44],
                        [], [42, 44],
                        0.45, 0.1, 0.5,
                        [0, 0, 0]),
                       ("No active synapses",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [],
                        0.45, 0.1, 0.5,
                        [0, 0, 0]),
                       ("Delta of zero",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.55, 0.0, 0.5,
                        [3, 3, 3])
         ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(segments, inputs, initialPermanence)
      connections.adjustActiveSynapses(segments[adjustedSegments],
                                       activeInputs, delta)

      overlaps = connections.computeActivity(inputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_adjustInactiveSynapses(self):
    for (name, cells, inputs,
         adjustedSegments, activeInputs,
         initialPermanence, delta, connectedPermanence,
         expected) in (("Basic test",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.45, 0.1, 0.5,
                        [1, 0, 1]),
                       ("Negative increment",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.55, -0.1, 0.5,
                        [2, 3, 2]),
                       ("No segments",
                        [1, 2, 3], [42, 43, 44],
                        [], [42, 44],
                        0.45, 0.1, 0.5,
                        [0, 0, 0]),
                       ("No active synapses",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [],
                        0.45, 0.1, 0.5,
                        [3, 0, 3]),
                       ("Delta of zero",
                        [1, 2, 3], [42, 43, 44],
                        [0, 2], [42, 44],
                        0.55, 0.0, 0.5,
                        [3, 3, 3])
         ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(segments, inputs, initialPermanence)
      connections.adjustInactiveSynapses(segments[adjustedSegments],
                                         activeInputs, delta)

      overlaps = connections.computeActivity(inputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_whenPermanenceFallsBelowZero(self):
    connections = SparseMatrixConnections(2048, 2048)

    segments = connections.createSegments([1, 2, 3])

    connections.growSynapses(segments, [42, 43], 0.05)
    connections.adjustSynapses(segments, [42, 43], -0.06, 0.0)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [0, 0, 0])

    connections.growSynapses(segments, [42, 43], 0.05)
    connections.adjustSynapses(segments, [], 0.0, -0.06)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [0, 0, 0])

    connections.growSynapses(segments, [42, 43], 0.05)
    connections.adjustActiveSynapses(segments, [42, 43], -0.06)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [0, 0, 0])

    connections.growSynapses(segments, [42, 43], 0.05)
    connections.adjustInactiveSynapses(segments, [], -0.06)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [0, 0, 0])


  def test_growSynapses(self):
    for (name,
         cells, growingSegments,
         presynapticInputs, activeInputs,
         initialPermanence, connectedPermanence,
         expected) in (("Basic test",
                        [1, 2, 3], [0, 2],
                        [42, 43, 44], [42, 43],
                        0.55, 0.5,
                        [2, 0, 2]),
                       ("No segments selected",
                        [1, 2, 3], [],
                        [42, 43, 44], [42, 43],
                        0.55, 0.5,
                        [0, 0, 0]),
                       ("No inputs selected",
                        [1, 2, 3], [0, 2],
                        [], [42, 43],
                        0.55, 0.5,
                        [0, 0, 0])
         ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(segments[growingSegments], presynapticInputs,
                               initialPermanence)

      overlaps = connections.computeActivity(activeInputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


  def test_growSynapsesToSample_single(self):

    rng = Random()

    for (name,
         cells, growingSegments,
         initialConnectedInputs, presynapticInputs, activeInputs,
         initialPermanence, connectedPermanence, sampleSize,
         expected) in (("Basic test",
                        [1, 2, 3], [0, 2],
                        [], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5, 2,
                        [2, 0, 2]),
                       ("One already connected",
                        [1, 2, 3], [0, 2],
                        [42], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5, 2,
                        [3, 0, 3]),
                       ("Higher sample size than axon count",
                        [1, 2, 3], [0, 2],
                        [], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5, 10,
                        [4, 0, 4]),
                       ("Higher sample size than available axon count",
                        [1, 2, 3], [0, 2],
                        [42, 43], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5, 3,
                        [4, 0, 4])
                       ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(
        segments[growingSegments], initialConnectedInputs, initialPermanence)

      connections.growSynapsesToSample(
        segments[growingSegments], presynapticInputs,
        sampleSize, initialPermanence, rng)

      overlaps = connections.computeActivity(activeInputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


    for (name,
         cells, growingSegments,
         initialConnectedInputs, presynapticInputs, activeInputs,
         initialPermanence, connectedPermanence,
         sampleSize) in (("Basic randomness test",
                          [1, 2, 3], [0, 2],
                          [], [42, 43, 44, 45],
                          [42, 43], 0.55, 0.5, 2),
         ):

      # Activate a subset of the inputs. The resulting overlaps should
      # differ on various trials.

      firstResult = None
      differingResults = False

      for _ in xrange(20):
        connections = SparseMatrixConnections(2048, 2048)

        segments = connections.createSegments(cells)

        connections.growSynapses(
          segments[growingSegments], initialConnectedInputs, initialPermanence)

        connections.growSynapsesToSample(
          segments[growingSegments], presynapticInputs,
          sampleSize, initialPermanence, rng)

        overlaps = connections.computeActivity(activeInputs,
                                               connectedPermanence)

        if firstResult is None:
          firstResult = overlaps[segments]
        else:
          differingResults = not np.array_equal(overlaps[segments], firstResult)
          if differingResults:
            break

      self.assertTrue(differingResults, name)


  def test_growSynapsesToSample_multi(self):

    rng = Random()

    for (name,
         cells, growingSegments,
         initialConnectedInputs, presynapticInputs, activeInputs,
         initialPermanence, connectedPermanence, sampleSizes,
         expected) in (("Basic test",
                        [1, 2, 3], [0, 2],
                        [], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5,
                        [2, 3],
                        [2, 0, 3]),
                       ("One already connected",
                        [1, 2, 3], [0, 2],
                        [42], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5,
                        [1, 2],
                        [2, 0, 3]),
                       ("Higher sample size than axon count",
                        [1, 2, 3], [0, 2],
                        [], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5,
                        [5, 10],
                        [4, 0, 4]),
                       ("Higher sample size than available axon count",
                        [1, 2, 3], [0, 2],
                        [42, 43], [42, 43, 44, 45],
                        [42, 43, 44, 45], 0.55, 0.5,
                        [3, 3],
                        [4, 0, 4])
                       ):

      connections = SparseMatrixConnections(2048, 2048)

      segments = connections.createSegments(cells)

      connections.growSynapses(
        segments[growingSegments], initialConnectedInputs, initialPermanence)

      connections.growSynapsesToSample(
        segments[growingSegments], presynapticInputs,
        sampleSizes, initialPermanence, rng)

      overlaps = connections.computeActivity(activeInputs, connectedPermanence)

      np.testing.assert_equal(overlaps[segments], expected, name)


    for (name,
         cells, growingSegments,
         initialConnectedInputs, presynapticInputs, activeInputs,
         initialPermanence, connectedPermanence,
         sampleSizes) in (("Basic randomness test",
                           [1, 2, 3], [0, 2],
                           [], [42, 43, 44, 45],
                           [42, 43], 0.55, 0.5, [2, 3]),
         ):

      # Activate a subset of the inputs. The resulting overlaps should
      # differ on various trials.

      firstResult = None
      differingResults = False

      for _ in xrange(20):
        connections = SparseMatrixConnections(2048, 2048)

        segments = connections.createSegments(cells)

        connections.growSynapses(
          segments[growingSegments], initialConnectedInputs, initialPermanence)

        connections.growSynapsesToSample(
          segments[growingSegments], presynapticInputs,
          sampleSizes, initialPermanence, rng)

        overlaps = connections.computeActivity(activeInputs,
                                               connectedPermanence)

        if firstResult is None:
          firstResult = overlaps[segments]
        else:
          differingResults = not np.array_equal(overlaps[segments], firstResult)
          if differingResults:
            break

      self.assertTrue(differingResults, name)


  def test_clipPermanences(self):
    connections = SparseMatrixConnections(2048, 2048)

    # Destroy synapses with permanences <= 0.0
    segments = connections.createSegments([1, 2, 3])
    connections.growSynapses(segments, [42, 43, 44], 0.05)
    connections.growSynapses(segments, [45, 46], 0.1)
    connections.adjustInactiveSynapses(segments, [], -0.1)
    connections.clipPermanences(segments)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [0, 0, 0])

    # Clip permanences to 1.0
    connections.growSynapses(segments, [42, 43, 44], 0.95)
    connections.adjustInactiveSynapses(segments, [], 0.50)
    connections.clipPermanences(segments)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [3, 3, 3])
    connections.adjustInactiveSynapses(segments, [], -0.5)
    overlaps1 = connections.computeActivity([42, 43, 44], 0.49)
    overlaps2 = connections.computeActivity([42, 43, 44], 0.51)
    np.testing.assert_equal(overlaps1, [3, 3, 3])
    np.testing.assert_equal(overlaps2, [0, 0, 0])


  def test_mapSegmentsToSynapseCounts(self):
    connections = SparseMatrixConnections(2048, 2048)

    segments = connections.createSegments([1, 2, 3])
    connections.growSynapses(segments, [42, 43, 44], 0.5)
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments),
                            [3, 3, 3])

    segments2 = connections.createSegments([4, 5])
    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts(segments2),
                            [0, 0])

    np.testing.assert_equal(connections.mapSegmentsToSynapseCounts([]),
                            [])
