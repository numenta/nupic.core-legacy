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
Sequence memory tests
"""

from abc import ABCMeta, abstractmethod
import random


class SequenceMemoryTestBase(object, metaclass=ABCMeta):
    """
    ============================================================================
                                    Basic First Order Sequences
    ============================================================================

    These tests ensure the most basic (first order) sequence learning mechanism
    is working.

    The parameters used in those tests are the typical default parameters for
    temporal memory, unless stated otherwise in the experiment (when self.init()
    is called).

            columnDimensions: (2048,)
            cellsPerColumn: 32
            initialPermanence: 0.5
            connectedPermanence: 0.6
            minThreshold: 25
            maxNewSynapseCount: 30
            permanenceIncrement: 0.1
            permanenceDecrement: 0.02
            predictedSegmentDecrement: 0.01
            activationThreshold: 25
            seed: 42

    Note: this is not a high order sequence, so one cell per column is fine.

    Input Sequence: We train with M input sequences, each consisting of N random
    patterns. Each pattern consists of a random number of bits on. The number of
    1's in each pattern should be between 38 and 40 columns.

    Each input pattern can optionally have an amount of spatial noise represented
    by X, where X is the probability of switching an on bit with a random bit.

    Training: The ETM is trained with P passes of the M sequences. There
    should be a reset between sequences. The total number of iterations during
    training is P*N*M.

    Testing: Run inference through the same set of sequences, with a reset before
    each sequence. For each sequence the system should accurately predict the
    pattern at the next time step up to and including the N-1'st pattern. The
    number of predicted inactive cells at each time step should be reasonably low.

    We can also calculate the number of synapses that should be
    learned. We raise an error if too many or too few were learned.

    B1) Basic sequence learner.    M=1, N=100, P=1.

    B2) Same as above, except P=2. Test that permanences go up and that no
    additional synapses are learned. [TODO]

    B3) N=300, M=1, P=1. (See how high we can go with N)

    B4) N=100, M=3, P=1. (See how high we can go with N*M)

    B5) Like B1 but with cellsPerColumn = 32. First order sequences should still
    work just fine.

    B6) Like B4 but with cellsPerColumn = 32. First order sequences should still
    work just fine.

    B7) Like B1 but with slower learning. Set the following parameters
    differently:

            initialPermanence = 0.2
            connectedPermanence = 0.7
            permanenceIncrement = 0.2

    Now we train the TP with the B1 sequence 4 times (P=4). This will increment
    the permanences to be above 0.8 and at that point the inference will be
    correct. This test will ensure the basic match function and segment
    activation rules are working correctly.

    B8) Like B7 but with 32 cells per column. Should still work.

    B9) Like B7 but present the sequence less than 4 times: the inference should
    be incorrect.

    B10) Like B2, except that cells per column = 32. Should still add zero
    additional synapses. [TODO]

    B11) Like B5, but with each pattern corrupted by a small amount of spatial
    noise (X = 0.05).

    B12) Test accessors.

    ============================================================================
                                    High Order Sequences
    ============================================================================

    These tests ensure that high order sequences can be learned in a multiple
    cells per column instantiation.

    Parameters: Same as Basic First Order Tests above, but with varying cells per
    column.

    Input Sequence: We train with M input sequences, each consisting of N random
    patterns. Each pattern consists of a random number of bits on. The number of
    1's in each pattern should be between 21 and 25 columns. The sequences are
    constructed to contain shared subsequences, such as:

    A B C D E F G H I J
    K L M D E F N O P Q

    The position and length of shared subsequences are parameters in the tests.

    Each input pattern can optionally have an amount of spatial noise represented
    by X, where X is the probability of switching an on bit with a random bit.

    Training: Identical to basic first order tests above.

    Testing: Identical to basic first order tests above unless noted.

    We can also calculate the number of segments and synapses that should be
    learned. We raise an error if too many or too few were learned.

    H1) Learn two sequences with a shared subsequence in the middle. Parameters
    should be the same as B1. Since cellsPerColumn == 1, it should make more
    predictions than necessary.

    H2) Same as H1, but with cellsPerColumn == 32, and train multiple times.
    It should make just the right number of predictions.

    H3) Like H2, except the shared subsequence is in the beginning (e.g.
    "ABCDEF" and "ABCGHIJ"). At the point where the shared subsequence ends, all
    possible next patterns should be predicted. As soon as you see the first
    unique pattern, the predictions should collapse to be a perfect prediction.

    H4) Shared patterns. Similar to H2 except that patterns are shared between
    sequences.    All sequences are different shufflings of the same set of N
    patterns (there is no shared subsequence).

    H5) Combination of H4) and H2). Shared patterns in different sequences, with
    a shared subsequence.

    H6) Stress test: every other pattern is shared. [TODO]

    H7) Start predicting in the middle of a sequence. [TODO]

    H8) Hub capacity. How many patterns can use that hub? [TODO]

    H9) Sensitivity to small amounts of spatial noise during inference
    (X = 0.05).
    Parameters the same as B11, and sequences like H2.

    H10) Higher order patterns with alternating elements.

    Create the following 4 sequences:

             A B A B A C
             A B A B D E
             A B F G H I
             A J K L M N

    After training we should verify that the expected transitions are in the
    model. Prediction accuracy should be perfect. In addition, during inference,
    after the first element is presented, the columns should not burst any more.
    Need to verify, for the first sequence, that the high order representation
    when presented with the second A and B is different from the representation
    in the first presentation. [TODO]
    """
    VERBOSITY = 1
    n = 2048
    w = 40
    feedback_size = 400


    def testB1(self):
        """Basic sequence learner.    M=1, N=100, P=1."""
        self.init({"cellsPerColumn": 1})

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(2):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))



    def testB3(self):
        """N=300, M=1, P=1. (See how high we can go with N)"""
        self.init({"cellsPerColumn": 1})

        sequence = [self.randomPattern() for _ in range(300)]

        # Learn
        for _ in range(2):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))


    def testB4(self):
        """N=100, M=3, P=1. (See how high we can go with N*M)"""
        self.init({"cellsPerColumn": 1})

        sequences = [[self.randomPattern() for _ in range(300)]
                                 for _ in range(3)]

        # Learn
        for _ in range(2):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))

            self.reset()


    def testB5(self):
        """Like B1 but with cellsPerColumn = 32.
        First order sequences should still work just fine."""

        self.init()

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(2):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))


    def testB6(self):
        """Like B4 but with cellsPerColumn = 32.
        First order sequences should still work just fine."""

        self.init()

        sequences = [[self.randomPattern() for _ in range(300)] for _ in range(3)]

        # Learn
        for _ in range(2):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))

            self.reset()


    def testB7(self):
        """Like B1 but with slower learning.

        Set the following parameters differently:

            initialPermanence = 0.2
            connectedPermanence = 0.7
            permanenceIncrement = 0.2

        Now we train the TM with the B1 sequence 4 times (P=4). This will
        increment the permanences to be above 0.8 and at that point the inference
        will be correct. This test will ensure the basic match function and
        segment activation rules are working correctly.
        """

        self.init({"initialPermanence": 0.2,
                             "connectedPermanence": 0.7,
                             "permanenceIncrement": 0.2,
                             "cellsPerColumn": 1})

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(4):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))


    def testB8(self):
        """Like B7 but with 32 cells per column.
        Should still work."""

        self.init({"initialPermanence": 0.2,
                             "connectedPermanence": 0.7,
                             "permanenceIncrement": 0.2})

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(4):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))


    def testB9(self):
        """Like B7 but present the sequence less than 4 times.
        The inference should be incorrect."""

        self.init({"initialPermanence": 0.2,
                             "connectedPermanence": 0.7,
                             "permanenceIncrement": 0.2})

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(3):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):
            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(set(), set(self.getPredictedCells()))
                self.assertEqual(self.w * 32, len(self.getActiveCells()))


    def testB11(self):
        """Like B5, but with each pattern corrupted by a small amount of spatial
        noise (X = 0.02)."""

        self.init()

        sequence = [self.randomPattern() for _ in range(100)]

        # Learn
        for _ in range(2):
            for pattern in sequence:
                self.compute(pattern, learn=True)

            self.reset()

        # Predict
        for i, pattern in enumerate(sequence):

            pattern = noisy(pattern, 3, self.n)

            self.compute(pattern, learn=False)

            if i > 0:
                self.assertEqual(self.w - 3, len(self.getPredictedActiveCells()))
                self.assertEqual(3, len(self.getPredictedInactiveCells()))
                self.assertEqual(3, len(self.getBurstingColumns()))


    def testH1(self):
        """Learn two sequences with a short shared pattern.

        Parameters should be the same as B1.
        Since cellsPerColumn == 1, it should make more predictions than necessary.
        """
        self.init({"cellsPerColumn": 1})

        random.seed(37)
        sharedSubsequence = [self.randomPattern() for _ in range(5)]

        sequences = [[self.randomPattern() for _ in range(10)] +
                     sharedSubsequence +
                     [self.randomPattern() for _ in range(5)]
                     for _ in range(2)]

        # Learn
        for _ in range(20):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    if i == 15:
                        # At the end of both shared sequences, there should be
                        # predicted but inactive columns
                        self.assertTrue(set(self.getActiveCells()).issubset(self.getPredictedCells()))
                        self.assertGreater(len(self.getPredictedCells()), len(self.getActiveCells()))
                    else:
                        self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))

            self.reset()


    def testH2(self):
        """Same as H1, but with cellsPerColumn == 32, and train more times.

        It should make just the right number of predictions.
        """

        self.init()

        random.seed(38)
        sharedSubsequence = [self.randomPattern() for _ in range(5)]

        sequences = [[self.randomPattern() for _ in range(10)] +
                     sharedSubsequence +
                     [self.randomPattern() for _ in range(5)]
                     for _ in range(2)]

        # Learn
        for _ in range(20):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))

            self.reset()


    def testH3(self):
        """Like H2, except the shared subsequence is in the beginning.
        (e.g. "ABCDEF" and "ABCGHIJ") At the point where the shared subsequence
        ends, all possible next patterns should be predicted. As soon as you see
        the first unique pattern, the predictions should collapse to be a perfect
        prediction."""

        self.init()

        random.seed(39)
        sharedSubsequence = [self.randomPattern() for _ in range(5)]

        sequences = [sharedSubsequence +
                     [self.randomPattern() for _ in range(15)]
                     for _ in range(2)]

        # Learn
        for _ in range(20):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    if i == 5:
                        # At the end of each shared sequence, there should be
                        # predicted but inactive columns
                        self.assertTrue(set(self.getActiveCells()).issubset(self.getPredictedCells()))
                        self.assertGreater(len(self.getPredictedCells()), len(self.getActiveCells()))
                    else:
                        self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))

            self.reset()


    def testH4(self):
        """Shared patterns.

        Similar to H2 except that patterns are shared between
        sequences. All sequences are different shufflings of the same set of N
        patterns (there is no intentional shared subsequence).
        """
        self.init()

        random.seed(40)
        elements = [self.randomPattern() for _ in range(10)]

        while True:
            sequences = []
            for _ in range(2):
                sequence = list(elements)
                random.shuffle(sequence)
                sequences.append(sequence)

            # If we're not careful, randomness will cause this test to fail.
            if len(getLongestSharedSubsequence(sequences)) < 4:
                break

        # Learn
        for _ in range(40):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    if i > 5:
                        self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))
                    else:
                        # Allow it a few timesteps to disambiguate, in case there are random
                        # shared subsequences.
                        self.assertTrue(set(self.getActiveCells()).issubset(self.getPredictedCells()))

            self.reset()


    def testH5(self):
        """Combination of H4) and H2).

        Shared patterns in different sequences, with a shared subsequence.
        """

        self.init()

        random.seed(41)
        elements = [self.randomPattern() for _ in range(20)]
        sharedSubsequence = [self.randomPattern() for _ in range(5)]

        sequences = []
        for _ in range(2):
            sublist = list(elements)
            random.shuffle(sublist)
            sequences.append(sublist[0:10] +
                                             sharedSubsequence +
                                             sublist[10:])

        # Learn
        for _ in range(80):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                if i > 0:
                    if i > 5:
                        self.assertEqual(set(self.getPredictedCells()), set(self.getActiveCells()))
                    else:
                        # Allow it a few timesteps to disambiguate, in case there are random
                        # shared subsequences.
                        self.assertTrue(set(self.getActiveCells()).issubset(self.getPredictedCells()))

            self.reset()


    def testH9(self):
        """Sensitivity to small amounts of spatial noise during inference
        (X = 0.05).

        Parameters are the same as in B11, and sequences are like in H2.
        """

        self.init()

        random.seed(42)
        sharedSubsequence = [self.randomPattern() for _ in range(5)]

        sequences = [[self.randomPattern() for _ in range(10)] +
                     sharedSubsequence +
                     [self.randomPattern() for _ in range(5)]
                     for _ in range(2)]

        # Learn
        for _ in range(20):
            for sequence in sequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        # Predict
        for sequence in sequences:
            for i, pattern in enumerate(sequence):

                pattern = noisy(pattern, 3, self.n)

                self.compute(pattern, learn=False)

                if i > 0:
                    self.assertEqual(self.w - 3, len(self.getPredictedActiveCells()))
                    self.assertEqual(3, len(self.getPredictedInactiveCells()))
                    self.assertEqual(3, len(self.getBurstingColumns()))

            self.reset()


    def testH10(self):
        """Orphan Decay mechanism reduce predicted inactive cells (extra
        predictions).

        Test feeds in noisy sequences (X = 0.05) to TM with and without orphan
        decay. TM with orphan decay should has many fewer predicted inactive
        columns.
        Parameters are the same as in B11, and sequences like in H9.
        """

        random.seed(43)
        sharedSubsequence = [self.randomPattern() for _ in range(3)]

        sequences = [[self.randomPattern() for _ in range(5)] +
                     sharedSubsequence +
                     [self.randomPattern() for _ in range(2)]
                     for _ in range(2)]

        # Add the same noise for both tests.
        allNoisySequences = [[[noisy(pattern, 2, self.n) for pattern in sequence] for sequence in sequences] for _ in range(10)]

        # Learn immediately so that we're sure there will be incorrect predictions.

        # train TM on noisy sequences with orphan decay turned off
        self.init({"initialPermanence": 0.70, "predictedSegmentDecrement": 0.0})

        for noisySequences in allNoisySequences:
            for sequence in noisySequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        numPredictedInactiveColumnsBefore = 0

        for sequence in sequences:
            for pattern in sequence:
                self.compute(pattern, learn=False)

                numPredictedInactiveColumnsBefore += len(
                    self.getPredictedInactiveColumns())

            self.reset()


        # train TM on the same set of noisy sequences with orphan decay turned on
        self.init({"initialPermanence": 0.70, "predictedSegmentDecrement": 0.08})

        for noisySequences in allNoisySequences:
            for sequence in noisySequences:
                for pattern in sequence:
                    self.compute(pattern, learn=True)

                self.reset()

        numPredictedInactiveColumnsAfter = 0

        for sequence in sequences:
            for i, pattern in enumerate(sequence):
                self.compute(pattern, learn=False)

                numPredictedInactiveColumnsAfter += len(self.getPredictedInactiveColumns())

                if i > 0:
                    self.assertEqual(len(self.getPredictedActiveColumns()), self.w)

            self.reset()

        self.assertGreater(numPredictedInactiveColumnsBefore, 0)
        self.assertGreater(numPredictedInactiveColumnsBefore, numPredictedInactiveColumnsAfter)


    def init(self, overrides=None):
        """
        Initialize Temporal Memory, and other member variables.

        :param overrides: overrides for default Temporal Memory parameters
        """

        params = {
            "columnCount": self.n,
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


        print("\n"
             "======================================================\n"
             "Test: {0} \n"
             "{1}\n"
             "======================================================\n"
        .format(self.id(), self.shortDescription()))


    # ==============================
    # Helper functions
    # ==============================


    def getPredictedActiveCells(self):
        return set(self.getPredictedCells()) & set(self.getActiveCells())


    def getPredictedInactiveCells(self):
        return set(self.getPredictedCells()) - set(self.getActiveCells())


    def getPredictedActiveColumns(self):
        predicted = set(cell // self.cellsPerColumn for cell in    self.getPredictedCells())
        active = set(cell // self.cellsPerColumn for cell in    self.getActiveCells())

        return active & predicted


    def getBurstingColumns(self):
        predicted = set(cell // self.cellsPerColumn for cell in    self.getPredictedCells())
        active = set(cell // self.cellsPerColumn for cell in    self.getActiveCells())

        return active - predicted


    def getPredictedInactiveColumns(self):
        predicted = set(cell // self.cellsPerColumn for cell in    self.getPredictedCells())
        active = set(cell // self.cellsPerColumn for cell in    self.getActiveCells())

        return predicted - active


    def randomPattern(self):
        return random.sample(range(self.n), self.w)


    # ==============================
    # Extension points
    # ==============================

    @abstractmethod
    def constructTM(self, columnCount, cellsPerColumn, initialPermanence,
                    connectedPermanence, minThreshold, sampleSize,
                    permanenceIncrement, permanenceDecrement,
                    predictedSegmentDecrement, activationThreshold, seed):
        """
        Construct a new TemporalMemory from these parameters.
        """
        pass


    @abstractmethod
    def compute(self, activeColumns, learn):
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



def noisy(pattern, wFlip, n):
    """
    Generate a noisy copy of a pattern.

    Deactivate wFlip cells, and activate wFlip other cells.

    @param pattern (iterable)
    A set of active indices

    @param wFlip (int)
    The number of bits to shuffle

    @param n (int)
    The number of bits in the SDR, active and inactive

    @return (set)
    A noisy set of active indices
    """

    noised = set(pattern)

    noised.difference_update(random.sample(noised, wFlip))

    for _ in range(wFlip):
        while True:
            v = random.randint(0, n - 1)
            if v not in pattern and v not in noised:
                noised.add(v)
                break

    return noised


def containsSublist(list1, sublist):
    for i in range(len(list1)):
        if list1[i:i+len(sublist)] == sublist:
            return True

    return False


def getLongestSharedSubsequence(sequences):
    """
    Find the longest subsequence that occurs more than once in the provided
    sequences.

    @param sequences (list of list of patterns)
    """

    best = []

    for sequenceNumber, currentSequence in enumerate(sequences):
        for i, startPoint in enumerate(currentSequence):
            currentSubsequence = [startPoint]

            while True:

                foundIt = False

                otherSequences = [currentSequence[i+1:]] + sequences[sequenceNumber+1:]
                for otherSequence in otherSequences:
                    foundIt = containsSublist(otherSequence, currentSubsequence)
                    if foundIt:
                        break

                if foundIt:
                    if len(currentSubsequence) > len(best):
                        best = currentSubsequence

                    if i+1 == len(currentSequence):
                        break
                    else:
                        i += 1
                        currentSubsequence = currentSubsequence + [currentSequence[i]]
                else:
                    break

    return best
