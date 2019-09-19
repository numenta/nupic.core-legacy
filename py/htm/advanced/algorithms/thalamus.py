# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Numenta, Inc. 
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
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""

An implementation of thalamic control and routing as proposed in the Cosyne
submission:

    A dendritic mechanism for dynamic routing and control in the thalamus Carmen Varela & Subutai Ahmad

"""



import numpy as np

from htm.bindings.sdr import SDR
from htm.bindings.math import Random
from .connections import Connections


class Thalamus(object):
    """

    A simple discrete time thalamus.    This thalamus has a 2D TRN layer and a 2D
    relay cell layer. L6 cells project to the dendrites of TRN cells - these
    connections are learned. TRN cells project to the dendrites of relay cells in
    a fixed fan-out pattern. A 2D feed forward input source projects to the relay
    cells in a fixed fan-out pattern.

    The output of the thalamus is the activity of each relay cell. This activity
    can be in one of three states: inactive, active (tonic), and active (burst).

    TRN cells control whether the relay cells will burst. If any dendrite on a TRN
    cell recognizes the current L6 pattern, it de-inactivates the T-type CA2+
    channels on the dendrites of any relay cell it projects to. These relay cells
    are then in "burst-ready mode".

    Feed forward activity is in the form of a binary vector corresponding to
    active/spiking axons (e.g. from ganglion cells). Any relay cells that receive
    input from an axon will either output tonic or burst activity depending on the
    state of the T-type CA2+ channels on their dendrites. Relay cells that don't
    receive input will remain inactive, regardless of their dendritic state.

    Usage:

        1. Train the TRN cells on a bunch of L6 patterns: learnL6Pattern()

        2. De-inactivate relay cells by sending in an L6 pattern: deInactivateCells()

        3. Compute feed forward activity for an input: computeFeedForwardActivity()

        4. reset()

        5. Goto 2

    """

    def __init__(self,
                 trnCellShape=(32, 32),
                 relayCellShape=(32, 32),
                 inputShape=(32, 32),
                 l6CellCount=1024,
                 trnThreshold=10,
                 relayThreshold=1,
                 connectedPermanence=0.5,
                 seed=42):
        """

        :param trnCellShape:
            a 2D shape for the TRN

        :param relayCellShape:
            a 2D shape for the relay cells

        :param l6CellCount:
            number of L6 cells

        :param trnThreshold:
            dendritic threshold for TRN cells. This is the min number of active L6
            cells on a dendrite for the TRN cell to recognize a pattern on that
            dendrite.

        :param relayThreshold:
            dendritic threshold for relay cells. This is the min number of active TRN
            cells on a dendrite for the relay cell to recognize a pattern on that
            dendrite.

        :param connectedPermanence:
            Permanence required for a distal synapse to be connected.

        :param seed:
                Seed for the random number generator.
        """

        # Shapes of TRN cell layer, relay cell layer, and feed forward input layer
        self.trnCellShape = trnCellShape
        self.trnWidth = trnCellShape[0]
        self.trnHeight = trnCellShape[1]

        self.relayCellShape = relayCellShape
        self.relayWidth = relayCellShape[0]
        self.relayHeight = relayCellShape[1]

        self.inputShape = inputShape
        self.inputWidth = inputShape[0]
        self.inputHeight = inputShape[1]


        self.l6CellCount = l6CellCount
        self.relayThreshold = relayThreshold
        self.seed = seed
        self.rng = Random(seed)
        self.trnActivationThreshold = trnThreshold
        self.connectedPermanence = connectedPermanence
        
        self.trnCellCount = trnCellShape[0]*trnCellShape[1]
        self.relayCellCount = relayCellShape[0]*relayCellShape[1]

        self.trnConnections = Connections(self.trnCellCount, connectedPermanence)

        self.relayConnections = Connections(self.relayCellCount, connectedPermanence)
        
        self.maxSegmentsPerTrnCell = l6CellCount
        self.maxSegmentsPerRelayCell = trnCellShape[0]*trnCellShape[1]

        # Initialize/reset variables that are updated with calls to compute
        self.reset()

        self._initializeTRNToRelayCellConnections()


    def learnL6Pattern(self, l6Pattern, cellsToLearnOn):
        """
        Learn the given l6Pattern on TRN cell dendrites. The TRN cells to learn
        are given in cellsTeLearnOn. Each of these cells will learn this pattern on
        a single dendritic segment.

        :param l6Pattern:
            An SDR from L6. List of indices corresponding to L6 cells.

        :param cellsToLearnOn:
            Each cell index is (x,y) corresponding to the TRN cells that should learn
            this pattern. For each cell, create a new dendrite that stores this
            pattern. The SDR is stored on this dendrite
        """
        cellIndices = [self.trnCellIndex(x) for x in cellsToLearnOn]
        for cell in cellIndices:
            newSegment = self.trnConnections.createSegment(cell, self.maxSegmentsPerTrnCell)
            self.trnConnections.growSynapses(newSegment, l6Pattern.sparse, self.connectedPermanence)

        # print("Learning L6 SDR:", l6Pattern,
        #             "new segments: ", newSegments,
        #             "cells:", self.trnConnections.mapSegmentsToCells(newSegments))


    def deInactivateCells(self, l6Input):
        """
        Activate trnCells according to the l6Input. These in turn will impact 
        bursting mode in relay cells that are connected to these trnCells.
        Given the feedForwardInput, compute which cells will be silent, tonic,
        or bursting.
        
        :param l6Input:
            An SDR from L6. List of indices corresponding to L6 cells.

        :return: nothing
        """
        # Figure out which TRN cells recognize the L6 pattern.
        self.trnOverlaps = self.trnConnections.computeActivity(l6Input, False)
        self.activeTRNSegments = np.flatnonzero(self.trnOverlaps >= self.trnActivationThreshold)
        self.activeTRNCellIndices = self.trnConnections.mapSegmentsToCells(self.activeTRNSegments)

        # for s, idx in zip(self.activeTRNSegments, self.activeTRNCellIndices):
        #     print(self.trnOverlaps[s], idx, self.trnIndextoCoord(idx))

        # Figure out which relay cells have dendrites in de-inactivated state
        activeTRNCells = SDR(self.trnCellCount)
        activeTRNCells.sparse = self.activeTRNCellIndices
        self.relayOverlaps = self.relayConnections.computeActivity(activeTRNCells, False)
        self.activeRelaySegments = np.flatnonzero(self.relayOverlaps >= self.relayThreshold)
        self.burstReadyCellIndices = self.relayConnections.mapSegmentsToCells(self.activeRelaySegments)

        self.burstReadyCells.reshape(-1)[self.burstReadyCellIndices] = 1


    def computeFeedForwardActivity(self, feedForwardInput):
        """
        Activate trnCells according to the l6Input. These in turn will impact
        bursting mode in relay cells that are connected to these trnCells.
        Given the feedForwardInput, compute which cells will be silent, tonic,
        or bursting.

        :param feedForwardInput:
            a numpy matrix of shape relayCellShape containing 0's and 1's

        :return:
            feedForwardInput is modified to contain 0, 1, or 2. A "2" indicates
            bursting cells.
        """
        ff = feedForwardInput.copy()
        # For each relay cell, see if any of its FF inputs are active.
        for x in range(self.relayWidth):
            for y in range(self.relayHeight):
                inputCells = self._preSynapticFFCells(x, y)
                for idx in inputCells:
                    if feedForwardInput[idx] != 0:
                        ff[x, y] = 1.0
                        continue

        # If yes, and it is in burst mode, this cell bursts
        # If yes, and it is not in burst mode, then we just get tonic input.

        # ff += self.burstReadyCells * ff
        ff2 = ff * 0.4 + self.burstReadyCells * ff
        return ff2


    def reset(self):
        """
        Set everything back to zero
        """
        self.trnOverlaps = []
        self.activeTRNSegments = []
        self.activeTRNCellIndices = []
        self.relayOverlaps = []
        self.activeRelaySegments = []
        self.burstReadyCellIndices = []
        self.burstReadyCells = np.zeros((self.relayWidth, self.relayHeight))


    def trnCellIndex(self, coord):
        """
        Map a 2D coordinate to 1D cell index.

        :param coord: a 2D coordinate

        :return: integer index
        """
        return coord[1] * self.trnWidth + coord[0]


    def trnIndextoCoord(self, i):
        """
        Map 1D cell index to a 2D coordinate

        :param i: integer 1D cell index

        :return: (x, y), a 2D coordinate
        """
        x = i % self.trnWidth
        y = i // self.trnWidth
        return x, y


    def relayCellIndex(self, coord):
        """
        Map a 2D coordinate to 1D cell index.

        :param coord: a 2D coordinate

        :return: integer index
        """
        return coord[1] * self.relayWidth + coord[0]


    def relayIndextoCoord(self, i):
        """
        Map 1D cell index to a 2D coordinate

        :param i: integer 1D cell index

        :return: (x, y), a 2D coordinate
        """
        x = i % self.relayWidth
        y = i // self.relayWidth
        return x, y


    def _initializeTRNToRelayCellConnections(self):
        """
        Initialize TRN to relay cell connectivity. For each relay cell, create a
        dendritic segment for each TRN cell it connects to.
        """
        for x in range(self.relayWidth):
            for y in range(self.relayHeight):

                # Create one dendrite for each trn cell that projects to this relay cell
                # This dendrite contains one synapse corresponding to this TRN->relay
                # connection.
                relayCellIndex = self.relayCellIndex((x,y))
                trnCells = self._preSynapticTRNCells(x, y)
                for trnCell in trnCells:
                    newSegment = self.relayConnections.createSegment(relayCellIndex, self.maxSegmentsPerRelayCell)
                    self.relayConnections.growSynapses(newSegment, np.array([self.trnCellIndex(trnCell)]), self.connectedPermanence)


    def _initializeRelayCellDendrites(self):
        """
        Initialize relay cell dendrites. If we assume that tau TRN cells connect to
        a given relay cell, and gamma feed-forward (FF) axons (e.g. ganglion cell
        axons) connect to each relay cell, we create tau * gamma dendritic segments
        on the relay cell. Each dendrite will have one of the TRN connections, and
        one of the FF connections.

        """
        pass


    def _preSynapticTRNCells(self, i, j):
        """
        Given a relay cell at the given coordinate, return a list of the (x,y)
        coordinates of all TRN cells that project to it. This assumes a 3X3 fan-in.

        :param i, j: relay cell Coordinates

        :return:
        """
        xmin = max(i - 1, 0)
        xmax = min(i + 2, self.trnWidth)
        ymin = max(j - 1, 0)
        ymax = min(j + 2, self.trnHeight)
        trnCells = [(x, y) for x in range(xmin, xmax) for y in range(ymin, ymax)]

        return trnCells


    def _preSynapticFFCells(self, i, j):
        """
        Given a relay cell at the given coordinate, return a list of the (x,y)
        coordinates of all feed forward cells that project to it. This assumes a 3X3
        fan-in.

        :param i, j: relay cell Coordinates

        :return:
        """
        xmin = max(i - 1, 0)
        xmax = min(i + 2, self.inputWidth)
        ymin = max(j - 1, 0)
        ymax = min(j + 2, self.inputHeight)
        inputCells = [(x, y) for x in range(xmin, xmax) for y in range(ymin, ymax)]

        return inputCells

