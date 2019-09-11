# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc. 
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
Test the Temporal Memory with explicit basal and apical input. Test that it
correctly uses the "apical tiebreak" approach to basal/apical input.
"""

from abc import ABCMeta, abstractmethod
import random



class ApicalTiebreakTestBase(object, metaclass=ABCMeta):
    """
    Test that a Temporal Memory successfully uses the following approach to basal
    and apical connections:

    - Basal input provides the "context". For a cell to be predicted, it must have
        an active basal segment.
    - When multiple cells in a single minicolumn have basal support, they are all
        predicted *unless* one of them also has an active apical segment. In that
        case, only the cells with basal and apical support are predicted.

    The apical dendrites resolve ambiguity when there are multiple cells in a
    minicolumn with basal support. In other words, they handle the situation where
    the basal input is a union.
    """
    apicalInputSize = 1000
    basalInputSize = 1000
    columnCount = 2048
    w = 40


    def setUp(self):

        self.cellsPerColumn = None

        print("\n"
             "======================================================\n"
             "Test: {0} \n"
             "{1}\n"
             "======================================================\n"
        .format(self.id(), self.shortDescription()))


    def testBasalInputRequiredForPredictions(self):
        """
        Learn A for basalInput1, apicalInput1.

        Now observe A with apicalInput1 but no basal input. It should burst.
        """

        self.init()

        activeColumns = self.randomColumnPattern()
        basalInput = self.randomBasalPattern()
        apicalInput = self.randomApicalPattern()

        for _ in range(3):
            self.compute(activeColumns, basalInput, apicalInput, learn=True)

        self.compute(activeColumns, basalInput=(), apicalInput=apicalInput, learn=False)

        self.assertEqual(set(activeColumns), self.getBurstingColumns())


    def testBasalPredictionsWithoutApical(self):
        """
        Learn A for two contexts:
        - basalInput1, apicalInput1
        - basalInput2, apicalInput2

        Now observe A with a union of basalInput1 and basalInput2, and no apical
        input. It should predict both contexts.
        """

        self.init()

        activeColumns = self.randomColumnPattern()
        basalInput1 = self.randomBasalPattern()
        basalInput2 = self.randomBasalPattern()
        apicalInput1 = self.randomApicalPattern()
        apicalInput2 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(activeColumns, basalInput1, apicalInput1, learn=True)
            activeCells1 = set(self.getActiveCells())
            self.compute(activeColumns, basalInput2, apicalInput2, learn=True)
            activeCells2 = set(self.getActiveCells())

        self.compute(activeColumns, basalInput1 | basalInput2, apicalInput=(), learn=False)

        self.assertEqual(activeCells1 | activeCells2, set(self.getActiveCells()))


    def testApicalNarrowsThePredictions(self):
        """
        Learn A for two contexts:
        - basalInput1, apicalInput1
        - basalInput2, apicalInput2

        Now observe A with a union of basalInput1 and basalInput2, and apicalInput1.
        It should only predict one context.
        """

        self.init()

        activeColumns = self.randomColumnPattern()
        basalInput1 = self.randomBasalPattern()
        basalInput2 = self.randomBasalPattern()
        apicalInput1 = self.randomApicalPattern()
        apicalInput2 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(activeColumns, basalInput1, apicalInput1, learn=True)
            activeCells1 = set(self.getActiveCells())
            self.compute(activeColumns, basalInput2, apicalInput2, learn=True)
            _ = set(self.getActiveCells())

        self.compute(activeColumns, basalInput1 | basalInput2, apicalInput1, learn=False)

        self.assertEqual(activeCells1, set(self.getActiveCells()))


    def testUnionOfFeedback(self):
        """
        Learn A for three contexts:
        - basalInput1, apicalInput1
        - basalInput2, apicalInput2
        - basalInput3, apicalInput3

        Now observe A with a union of all 3 basal inputs, and a union of
        apicalInput1 and apicalInput2. It should predict 2 of the 3 contexts.
        """

        self.init()

        activeColumns = self.randomColumnPattern()
        basalInput1 = self.randomBasalPattern()
        basalInput2 = self.randomBasalPattern()
        basalInput3 = self.randomBasalPattern()
        apicalInput1 = self.randomApicalPattern()
        apicalInput2 = self.randomApicalPattern()
        apicalInput3 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(activeColumns, basalInput1, apicalInput1, learn=True)
            activeCells1 = set(self.getActiveCells())
            self.compute(activeColumns, basalInput2, apicalInput2, learn=True)
            activeCells2 = set(self.getActiveCells())
            self.compute(activeColumns, basalInput3, apicalInput3, learn=True)
            _ = set(self.getActiveCells())

        self.compute(activeColumns, basalInput1 | basalInput2 | basalInput3,
                                 apicalInput1 | apicalInput2, learn=False)

        self.assertEqual(activeCells1 | activeCells2, set(self.getActiveCells()))



    # ==============================
    # Helper functions
    # ==============================


    def init(self, overrides=None):
        """
        Initialize Temporal Memory, and other member variables.

        @param overrides (dict)
        Overrides for default Temporal Memory parameters
        """

        params = {
            "columnCount": self.columnCount,
            "basalInputSize": self.basalInputSize,
            "apicalInputSize": self.apicalInputSize,
            "cellsPerColumn": 32,
            "initialPermanence": 0.5,
            "connectedPermanence": 0.6,
            "minThreshold": 25,
            "sampleSize": 30,
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.02,
            "predictedSegmentDecrement": 0.0,
            "activationThreshold": 25,
            "seed": 42,
        }

        params.update(overrides or {})

        self.cellsPerColumn = params["cellsPerColumn"]

        self.constructTM(**params)


    def getBurstingColumns(self):
        predicted = set(cell // self.cellsPerColumn for cell in    self.getPredictedCells())
        active = set(cell // self.cellsPerColumn for cell in    self.getActiveCells())

        return active - predicted


    def randomColumnPattern(self):
        return set(random.sample(range(self.columnCount), self.w))


    def randomApicalPattern(self):
        return set(random.sample(range(self.apicalInputSize), self.w))


    def randomBasalPattern(self):
        return set(random.sample(range(self.basalInputSize), self.w))

    # ==============================
    # Extension points
    # ==============================

    @abstractmethod
    def constructTM(self, columnCount, basalInputSize, apicalInputSize,
                    cellsPerColumn, initialPermanence, connectedPermanence,
                    minThreshold, sampleSize, permanenceIncrement,
                    permanenceDecrement, predictedSegmentDecrement,
                    activationThreshold, seed):
        """
        Construct a new TemporalMemory from these parameters.
        """
        pass


    @abstractmethod
    def compute(self, activeColumns, basalInput, apicalInput, learn):
        """
        Run one timestep of the TemporalMemory.
        """
        pass


    @abstractmethod
    def getActiveCells(self):
        """
        Get the currently active cells.
        """
        pass


    @abstractmethod
    def getPredictedCells(self):
        """
        Get the cells that were predicted for the current timestep.

        In other words, the set of "correctly predicted cells" is the intersection
        of these cells and the active cells.
        """
        pass
