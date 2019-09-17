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
Sequence memory tests that focus on the effects of feedback.
"""

from abc import ABCMeta, abstractmethod
import random

class ApicalTiebreakSequencesTestBase(object, metaclass=ABCMeta):
    """
    Test that a Temporal Memory uses apical dendrites as part of sequence
    inference.

    The expected basal / apical algorithm is:

    - Basal input provides the "context". For a cell to be predicted, it must have
        an active basal segment.
    - When multiple cells in a single minicolumn have basal support, they are all
        predicted *unless* one of them also has an active apical segment. In that
        case, only the cells with basal and apical support are predicted.

    The apical dendrites resolve ambiguity when there are multiple cells in a
    minicolumn with basal support. In other words, they handle the situation where
    the previous input is bursting.
    """
    columnCount = 2048
    w = 40
    apicalInputSize = 1000


    def setUp(self):

        self.cellsPerColumn = None

        print("\n"
             "======================================================\n"
             "Test: {0} \n"
             "{1}\n"
             "======================================================\n"
        .format(self.id(), self.shortDescription()))


    def testSequenceMemory_BasalInputRequiredForPredictions(self):
        """
        Learn ABCDE with F1.
        Reset, then observe B with F1.

        It should burst, despite the fact that the B cells have apical support.
        """

        self.init()

        abcde = [self.randomColumnPattern() for _ in range(5)]
        feedback = self.randomApicalPattern()

        for _ in range(4):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=feedback, learn=True)

        self.reset()
        self.compute(abcde[1], apicalInput=feedback, learn=False)

        self.assertEqual([], list(self.getPredictedCells()))
        self.assertEqual(set(abcde[1]), self.getBurstingColumns())


    def testSequenceMemory_BasalPredictionsWithoutFeedback(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE. Without feedback, two patterns are predicted.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=feedback1, learn=True)

            eCells = set(self.getActiveCells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=feedback2, learn=True)

            yCells = set(self.getActiveCells())

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apicalInput=(), learn=False)

        # The E cells should be active, and so should any Y cells that happen to be
        # in a minicolumn shared between E and Y.
        expectedActive = eCells | set(self.filterCellsByColumn(yCells, abcde[4]))

        self.assertEqual(expectedActive, set(self.getActiveCells()))
        self.assertEqual(eCells | yCells, set(self.getPredictedCells()))


    def testSequenceMemory_FeedbackNarrowsThePredictions(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE with F1. One pattern is predicted.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=feedback1, learn=True)

            eCells = set(self.getActiveCells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=feedback2, learn=True)

            yCells = set(self.getActiveCells())

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apicalInput=feedback1, learn=False)

        self.assertEqual(eCells, set(self.getActiveCells()))
        self.assertEqual(eCells, set(self.getPredictedCells()))


    def testSequenceMemory_IncorrectFeedbackLeadsToBursting(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE with F2. E should burst.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=feedback1, learn=True)

            eCells = set(self.getActiveCells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=feedback2, learn=True)

            yCells = set(self.getActiveCells())

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apicalInput=feedback2, learn=False)

        self.assertEqual(yCells, set(self.getPredictedCells()))

        # E should burst, except for columns that happen to be shared with Y.
        self.assertEqual(set(abcde[4]) - set(xbcdy[4]),
                                         set(self.getBurstingColumns()))


    def testSequenceMemory_UnionOfFeedback(self):
        """
        Train on ABCDE with F1, XBCDY with F2, MBCDN with F3.
        Test with BCDE with F1 | F2. The last step should predict E and Y.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        mbcdn = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()
        feedback3 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(20):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=(), learn=True)

            self.reset()
            for pattern in mbcdn:
                self.compute(pattern, apicalInput=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apicalInput=feedback1, learn=True)

            eCells = set(self.getActiveCells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apicalInput=feedback2, learn=True)

            yCells = set(self.getActiveCells())

            self.reset()
            for pattern in mbcdn:
                self.compute(pattern, apicalInput=feedback3, learn=True)

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apicalInput=feedback1 | feedback2, learn=False)

        # The E cells should be active, and so should any Y cells that happen to be
        # in a minicolumn shared between E and Y.
        expectedActive = eCells | set(self.filterCellsByColumn(yCells, abcde[4]))

        self.assertEqual(expectedActive, set(self.getActiveCells()))
        self.assertEqual(eCells | yCells, set(self.getPredictedCells()))



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
            "apicalInputSize": self.apicalInputSize,
            "cellsPerColumn": 32,
            "initialPermanence": 0.5,
            "connectedPermanence": 0.6,
            "minThreshold": 25,
            "sampleSize": 30,
            "permanenceIncrement": 0.1,
            "permanenceDecrement": 0.02,
            "predictedSegmentDecrement": 0.08,
            "activationThreshold": 25,
            "seed": 42,
        }

        params.update(overrides or {})

        self.cellsPerColumn = params["cellsPerColumn"]

        self.constructTM(**params)


    def getBurstingColumns(self):
        predicted = set(cell // self.cellsPerColumn for cell in self.getPredictedCells())
        active = set(cell // self.cellsPerColumn for cell in self.getActiveCells())

        return active - predicted


    def randomColumnPattern(self):
        return set(random.sample(list(range(self.columnCount)), self.w))


    def randomApicalPattern(self):
        return set(random.sample(list(range(self.apicalInputSize)), self.w))


    def filterCellsByColumn(self, cells, columns):
        return [cell for cell in cells if (cell // self.cellsPerColumn) in columns]


    # ==============================
    # Extension points
    # ==============================

    @abstractmethod
    def constructTM(self, columnCount, apicalInputSize, cellsPerColumn,
                    initialPermanence, connectedPermanence, minThreshold,
                    sampleSize, permanenceIncrement, permanenceDecrement,
                    predictedSegmentDecrement, activationThreshold, seed):
        """
        Construct a new TemporalMemory from these parameters.
        """
        pass


    @abstractmethod
    def compute(self, activeColumns, apicalInput, learn):
        """
        Run one timestep of the TemporalMemory.
        """
        pass


    @abstractmethod
    def reset(self):
        """
        Reset the TemporalMemory.
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
