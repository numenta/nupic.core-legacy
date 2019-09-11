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

import unittest
import numpy as np

from htm.advanced.data.generators.pattern_machine import PatternMachine

from htm.advanced.algorithms.column_pooler import ColumnPooler
from htm.advanced.algorithms.monitor_mixin.column_pooler_mixin import ColumnPoolerMonitorMixin


class MonitoredColumnPooler(ColumnPoolerMonitorMixin, ColumnPooler):
    pass



class ExtensiveColumnPoolerTest(unittest.TestCase):
    """
    Algorithmic tests for the ColumnPooler region.

    Each test actually tests multiple aspects of the algorithm. For more
    atomic tests refer to column_pooler_unit_test.

    The notation for objects is the following:
        object{patternA, patternB, ...}

    In these tests, the proximally-fed SDR's are simulated as unique (location,
    feature) pairs regardless of actual locations and features, unless stated
    otherwise.
    """

    inputWidth = 2048 * 8
    numInputActiveBits = int(0.02 * inputWidth)
    outputWidth = 4096
    numOutputActiveBits = 40
    seed = 42


    def testNewInputs(self):
        """
        Checks that the behavior is correct when facing unseed inputs.
        """
        self.init()

        # feed the first input, a random SDR should be generated
        initialPattern = self.generateObject(1)
        self.learn(initialPattern, numRepetitions=1, newObject=True)
        representation = self._getActiveRepresentation()
        self.assertEqual(
            len(representation),
            self.numOutputActiveBits,
            "The generated representation is incorrect"
        )

        # feed a new input for the same object, the previous SDR should persist
        newPattern = self.generateObject(1)
        self.learn(newPattern, numRepetitions=1, newObject=False)
        newRepresentation = self._getActiveRepresentation()
        self.assertNotEqual(initialPattern, newPattern)
        self.assertEqual(
            newRepresentation,
            representation,
            "The SDR did not persist when learning the same object"
        )

        # without sensory input, the SDR should persist as well
        emptyPattern = [set()]
        self.learn(emptyPattern, numRepetitions=1, newObject=False)
        newRepresentation = self._getActiveRepresentation()
        self.assertEqual(
            newRepresentation,
            representation,
            "The SDR did not persist after an empty input."
        )


    def testLearnSinglePattern(self):
        """
        A single pattern is learnt for a single object.
        Objects: A{X, Y}
        """
        self.init()

        test_object = self.generateObject(1)
        self.learn(test_object, numRepetitions=2, newObject=True)
        # check that the active representation is sparse
        representation = self._getActiveRepresentation()
        self.assertEqual(
            len(representation),
            self.numOutputActiveBits,
            "The generated representation is incorrect"
        )

        # check that the pattern was correctly learnt
        self.pooler.reset()
        self.infer(feedforwardPattern=test_object[0])
        self.assertEqual(
            self._getActiveRepresentation(),
            representation,
            "The pooled representation is not stable"
        )

        # present new pattern for same test_object
        # it should be mapped to the same representation
        newPattern = [self.generatePattern()]
        self.learn(newPattern, numRepetitions=2, newObject=False)
        # check that the active representation is sparse
        newRepresentation = self._getActiveRepresentation()
        self.assertEqual(
            newRepresentation,
            representation,
            "The new pattern did not map to the same test_object representation"
        )

        # check that the pattern was correctly learnt and is stable
        self.pooler.reset()
        self.infer(feedforwardPattern=test_object[0])
        self.assertEqual(
            self._getActiveRepresentation(),
            representation,
            "The pooled representation is not stable"
        )


    def testLearnSingleObject(self):
        """
        Many patterns are learnt for a single test_object.
        Objects: A{P, Q, R, S, T}
        """
        self.init()

        test_object = self.generateObject(numPatterns=5)
        self.learn(test_object, numRepetitions=2, randomOrder=True, newObject=True)
        representation = self._getActiveRepresentation()

        # check that all patterns map to the same test_object
        for pattern in test_object:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representation,
                "The pooled representation is not stable"
            )

        # if activity stops, check that the representation persists
        self.infer(feedforwardPattern=set())
        self.assertEqual(
            self._getActiveRepresentation(),
            representation,
            "The pooled representation did not persist"
        )


    def testLearnTwoObjectNoCommonPattern(self):
        """
        Same test as before, using two objects, without common pattern.
        Objects: A{P, Q, R, S,T}     B{V, W, X, Y, Z}
        """
        self.init()

        objectA = self.generateObject(numPatterns=5)
        self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
        representationA = self._getActiveRepresentation()

        objectB = self.generateObject(numPatterns=5)
        self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
        representationB = self._getActiveRepresentation()

        self.assertNotEqual(representationA, representationB)

        # check that all patterns map to the same object
        for pattern in objectA:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationA,
                "The pooled representation for the first object is not stable"
            )

        # check that all patterns map to the same object
        for pattern in objectB:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationB,
                "The pooled representation for the second object is not stable"
            )

        # feed union of patterns in object A
        pattern = objectA[0] | objectA[1]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA,
            "The active representation is incorrect"
        )

        # feed unions of patterns in objects A and B
        pattern = objectA[0] | objectB[0]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB,
            "The active representation is incorrect"
        )


    def testLearnTwoObjectsOneCommonPattern(self):
        """
        Same test as before, except the two objects share a pattern
        Objects: A{P, Q, R, S,T}     B{P, W, X, Y, Z}
        """
        self.init()

        objectA = self.generateObject(numPatterns=5)
        self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
        representationA = self._getActiveRepresentation()

        objectB = self.generateObject(numPatterns=5)
        objectB[0] = objectA[0]
        self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
        representationB = self._getActiveRepresentation()

        self.assertNotEqual(representationA, representationB)
        # very small overlap
        self.assertLessEqual(len(representationA & representationB), 3)

        # check that all patterns except the common one map to the same object
        for pattern in objectA[1:]:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationA,
                "The pooled representation for the first object is not stable"
            )

        # check that all patterns except the common one map to the same object
        for pattern in objectB[1:]:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationB,
                "The pooled representation for the second object is not stable"
            )

        # feed shared pattern
        pattern = objectA[0]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB,
            "The active representation is incorrect"
        )

        # feed union of patterns in object A
        pattern = objectA[1] | objectA[2]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA,
            "The active representation is incorrect"
        )

        # feed unions of patterns in objects A and B
        pattern = objectA[1] | objectB[1]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB,
            "The active representation is incorrect"
        )


    def testLearnThreeObjectsOneCommonPattern(self):
        """
        Same test as before, with three objects
        Objects: A{P, Q, R, S,T}     B{P, W, X, Y, Z}     C{W, H, I, K, L}
        """
        self.init()

        objectA = self.generateObject(numPatterns=5)
        self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
        representationA = self._getActiveRepresentation()

        objectB = self.generateObject(numPatterns=5)
        objectB[0] = objectA[0]
        self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
        representationB = self._getActiveRepresentation()

        objectC = self.generateObject(numPatterns=5)
        objectC[0] = objectB[1]
        self.learn(objectC, numRepetitions=3, randomOrder=True, newObject=True)
        representationC = self._getActiveRepresentation()

        self.assertNotEqual(representationA, representationB, representationC)
        # very small overlap
        self.assertLessEqual(len(representationA & representationB), 3)
        self.assertLessEqual(len(representationB & representationC), 3)
        self.assertLessEqual(len(representationA & representationC), 3)


        # check that all patterns except the common one map to the same object
        for pattern in objectA[1:]:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationA,
                "The pooled representation for the first object is not stable"
            )

        # check that all patterns except the common one map to the same object
        for pattern in objectB[2:]:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationB,
                "The pooled representation for the second object is not stable"
            )

        # check that all patterns except the common one map to the same object
        for pattern in objectC[1:]:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationC,
                "The pooled representation for the third object is not stable"
            )

        # feed shared pattern between A and B
        pattern = objectA[0]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB,
            "The active representation is incorrect"
        )

        # feed shared pattern between B and C
        pattern = objectB[1]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationB | representationC,
            "The active representation is incorrect"
        )

        # feed union of patterns in object A
        pattern = objectA[1] | objectA[2]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA,
            "The active representation is incorrect"
        )

        # feed unions of patterns to activate all objects
        pattern = objectA[1] | objectB[1]
        self.pooler.reset()
        self.infer(feedforwardPattern=pattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB | representationC,
            "The active representation is incorrect"
        )


    def testLearnThreeObjectsOneCommonPatternSpatialNoise(self):
        """
        Same test as before, with three objects
        Objects: A{P, Q, R, S,T}     B{P, W, X, Y, Z}     C{W, H, I, K, L}
        """
        self.init()

        objectA = self.generateObject(numPatterns=5)
        self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
        representationA = self._getActiveRepresentation()

        objectB = self.generateObject(numPatterns=5)
        objectB[0] = objectA[0]
        self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
        representationB = self._getActiveRepresentation()

        objectC = self.generateObject(numPatterns=5)
        objectC[0] = objectB[1]
        self.learn(objectC, numRepetitions=3, randomOrder=True, newObject=True)
        representationC = self._getActiveRepresentation()

        self.assertNotEqual(representationA, representationB, representationC)
        # very small overlap
        self.assertLessEqual(len(representationA & representationB), 3)
        self.assertLessEqual(len(representationB & representationC), 3)
        self.assertLessEqual(len(representationA & representationC), 3)


        # check that all patterns except the common one map to the same object
        for pattern in objectA[1:]:
            noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
            self.pooler.reset()
            self.infer(feedforwardPattern=noisyPattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationA,
                "The pooled representation for the first object is not stable"
            )

        # check that all patterns except the common one map to the same object
        for pattern in objectB[2:]:
            noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
            self.pooler.reset()
            self.infer(feedforwardPattern=noisyPattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationB,
                "The pooled representation for the second object is not stable"
            )

        # check that all patterns except the common one map to the same object
        for pattern in objectC[1:]:
            noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
            self.pooler.reset()
            self.infer(feedforwardPattern=noisyPattern)
            self.assertEqual(
                self._getActiveRepresentation(),
                representationC,
                "The pooled representation for the third object is not stable"
            )

        # feed shared pattern between A and B
        pattern = objectA[0]
        noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
        self.pooler.reset()
        self.infer(feedforwardPattern=noisyPattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB,
            "The active representation is incorrect"
        )

        # feed shared pattern between B and C
        pattern = objectB[1]
        noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
        self.pooler.reset()
        self.infer(feedforwardPattern=noisyPattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationB | representationC,
            "The active representation is incorrect"
        )

        # feed union of patterns in object A
        pattern = objectA[1] | objectA[2]
        noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
        self.pooler.reset()
        self.infer(feedforwardPattern=noisyPattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA,
            "The active representation is incorrect"
        )

        # feed unions of patterns to activate all objects
        pattern = objectA[1] | objectB[1]
        noisyPattern = self.proximalPatternMachine.addNoise(pattern, 0.05)
        self.pooler.reset()
        self.infer(feedforwardPattern=noisyPattern)
        self.assertEqual(
            self._getActiveRepresentation(),
            representationA | representationB | representationC,
            "The active representation is incorrect"
        )


    def testInferObjectOverTime(self):
        """Infer an object after touching only ambiguous points."""
        self.init()

        patterns = [self.generatePattern() for _ in range(3)]

        objectA = [patterns[0], patterns[1]]
        objectB = [patterns[1], patterns[2]]
        objectC = [patterns[2], patterns[0]]

        self.learn(objectA, numRepetitions=3, newObject=True)
        representationA = set(self.pooler.getActiveCells())
        self.learn(objectB, numRepetitions=3, newObject=True)
        _ = set(self.pooler.getActiveCells())
        self.learn(objectC, numRepetitions=3, newObject=True)
        representationC = set(self.pooler.getActiveCells())

        self.pooler.reset()
        self.infer(patterns[0])
        self.assertEqual(set(self.pooler.getActiveCells()), representationA | representationC)
        self.infer(patterns[1])
        self.assertEqual(set(self.pooler.getActiveCells()), representationA)


    def testLearnOneObjectInTwoColumns(self):
        """Learns one object in two different columns."""
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        objectARepresentations = self._getActiveRepresentations()

        for pooler in self.poolers:
            pooler.reset()

        for patterns in objectA:
            for i in range(3):
                activeRepresentations = self._getActiveRepresentations()

                self.inferMultipleColumns(
                    feedforwardPatterns=patterns,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                if i > 0:
                    self.assertEqual(activeRepresentations, self._getActiveRepresentations())
                    self.assertEqual(objectARepresentations, self._getActiveRepresentations())


    def testLearnTwoObjectsInTwoColumnsNoCommonPattern(self):
        """Learns two objects in two different columns."""
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)
        objectB = self.generateObject(numPatterns=5, numCols=2)

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsA = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectB,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True,
        )
        activeRepresentationsB = self._getActiveRepresentations()

        for pooler in self.poolers:
            pooler.reset()

        # check inference for object A
        for patternsA in objectA:
            for _ in range(3):
                activeRepresentations = self._getActiveRepresentations()
                self.inferMultipleColumns(
                    feedforwardPatterns=patternsA,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                self.assertEqual(
                    activeRepresentationsA,
                    self._getActiveRepresentations()
                )

        for pooler in self.poolers:
            pooler.reset()

        # check inference for object B
        for patternsB in objectB:
            for _ in range(3):
                activeRepresentations = self._getActiveRepresentations()
                self.inferMultipleColumns(
                    feedforwardPatterns=patternsB,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices
                )

                self.assertEqual(
                    activeRepresentationsB,
                    self._getActiveRepresentations()
                )


    def testLearnTwoObjectsInTwoColumnsOneCommonPattern(self):
        """Learns two objects in two different columns, with a common pattern."""
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)
        objectB = self.generateObject(numPatterns=5, numCols=2)

        # second pattern in column 0 is shared
        objectB[1][0] = objectA[1][0]

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsA = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectB,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsB = self._getActiveRepresentations()

        # check inference for object A
        # for the first pattern, the distal predictions won't be correct
        # for the second one, the prediction will be unique thanks to the
        # distal predictions from the other column which has no ambiguity
        for pooler in self.poolers:
            pooler.reset()

        for patternsA in objectA:
            for _ in range(3):
                activeRepresentations = self._getActiveRepresentations()
                self.inferMultipleColumns(
                    feedforwardPatterns=patternsA,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                self.assertEqual(
                    activeRepresentationsA,
                    self._getActiveRepresentations()
                )

        for pooler in self.poolers:
            pooler.reset()

        # check inference for object B
        for patternsB in objectB:
            for _ in range(3):
                activeRepresentations = self._getActiveRepresentations()
                self.inferMultipleColumns(
                    feedforwardPatterns=patternsB,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices
                )

                self.assertEqual(
                    activeRepresentationsB,
                    self._getActiveRepresentations()
                )


    def testLearnTwoObjectsInTwoColumnsOneCommonPatternEmptyFirstInput(self):
        """Learns two objects in two different columns, with a common pattern."""
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)
        objectB = self.generateObject(numPatterns=5, numCols=2)

        # second pattern in column 0 is shared
        objectB[1][0] = objectA[1][0]

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsA = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectB,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        _ = self._getActiveRepresentations()

        # check inference for object A
        for pooler in self.poolers:
            pooler.reset()

        firstPattern = True
        for patternsA in objectA:
            activeRepresentations = self._getActiveRepresentations()
            if firstPattern:
                self.inferMultipleColumns(
                    feedforwardPatterns=[set(), patternsA[1]],
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                desiredRepresentation = [set(), activeRepresentationsA[1]]
            else:
                self.inferMultipleColumns(
                    feedforwardPatterns=patternsA,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                desiredRepresentation = activeRepresentationsA
            self.assertEqual(
                desiredRepresentation,
                self._getActiveRepresentations()
            )


    def testPersistence(self):
        """After learning, representation should persist in L2 without input."""
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        objectARepresentations = self._getActiveRepresentations()

        for pooler in self.poolers:
            pooler.reset()

        for patterns in objectA:
            for i in range(3):

                # replace third pattern for column 2 by empty pattern
                if i == 2:
                    patterns[1] = set()

                activeRepresentations = self._getActiveRepresentations()

                self.inferMultipleColumns(
                    feedforwardPatterns=patterns,
                    activeRepresentations=activeRepresentations,
                    neighborsIndices=neighborsIndices,
                )
                if i > 0:
                    self.assertEqual(activeRepresentations, self._getActiveRepresentations())
                    self.assertEqual(objectARepresentations, self._getActiveRepresentations())


    def testLateralDisambiguation(self):
        """Lateral disambiguation using a constant simulated distal input."""
        self.init(overrides={
            "lateralInputWidths": [self.inputWidth],
        })

        objectA = self.generateObject(numPatterns=5)
        lateralInputA = [[set()]] + [[self.generatePattern()] for _ in range(4)]
        self.learn(objectA,
                     lateralPatterns=lateralInputA,
                     numRepetitions=3,
                     randomOrder=True,
                     newObject=True)
        representationA = self._getActiveRepresentation()

        objectB = self.generateObject(numPatterns=5)
        objectB[3] = objectA[3]
        lateralInputB = [[set()]] + [[self.generatePattern()] for _ in range(4)]
        self.learn(objectB,
                     lateralPatterns=lateralInputB,
                     numRepetitions=3,
                     randomOrder=True,
                     newObject=True)
        representationB = self._getActiveRepresentation()

        self.assertNotEqual(representationA, representationB)
        # very small overlap
        self.assertLessEqual(len(representationA & representationB), 3)

        # no ambiguity with lateral input
        for pattern in objectA:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern, lateralInputs=lateralInputA[-1])
            self.assertEqual(
                self._getActiveRepresentation(),
                representationA,
                "The pooled representation for the first object is not stable"
            )

        # no ambiguity with lateral input
        for pattern in objectB:
            self.pooler.reset()
            self.infer(feedforwardPattern=pattern, lateralInputs=lateralInputB[-1])
            self.assertEqual(
                self._getActiveRepresentation(),
                representationB,
                "The pooled representation for the second object is not stable"
            )


    def testLateralContestResolved(self):
        """
        Infer an object via lateral disambiguation even if some other columns have
        similar ambiguity.

        """

        self.init(overrides={"lateralInputWidths": [self.inputWidth, self.inputWidth]})

        patterns = [self.generatePattern() for _ in range(3)]

        objectA = [patterns[0], patterns[1]]
        objectB = [patterns[1], patterns[2]]

        lateralInput1A = self.generatePattern()
        lateralInput2A = self.generatePattern()
        lateralInput1B = self.generatePattern()
        lateralInput2B = self.generatePattern()

        self.learn(objectA, lateralPatterns=[[lateralInput1A, lateralInput2A]]*2,
                             numRepetitions=3, newObject=True)
        representationA = set(self.pooler.getActiveCells())
        self.learn(objectB, lateralPatterns=[[lateralInput1B, lateralInput2B]]*2,
                             numRepetitions=3, newObject=True)
        _ = set(self.pooler.getActiveCells())

        self.pooler.reset()

        # This column will say A | B
        # One lateral column says A | B
        # Another lateral column says A
        self.infer(patterns[1], lateralInputs=[(), ()])
        self.infer(patterns[1], lateralInputs=[lateralInput1A | lateralInput1B, lateralInput2A])


        self.assertEqual(set(self.pooler.getActiveCells()), representationA)





    @unittest.skip("Fails, need to discuss")
    def testMultiColumnCompetition(self):
        """Competition between multiple conflicting lateral inputs."""
        self.init(numCols=4)
        neighborsIndices = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

        objectA = self.generateObject(numPatterns=5, numCols=4)
        objectB = self.generateObject(numPatterns=5, numCols=4)

        # second pattern in column 0 is shared
        objectB[1][0] = objectA[1][0]

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsA = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectB,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsB = self._getActiveRepresentations()

        # check inference for object A
        # for the first pattern, the distal predictions won't be correct
        # for the second one, the prediction will be unique thanks to the
        # distal predictions from the other column which has no ambiguity
        for pooler in self.poolers:
            pooler.reset()

        # sensed patterns will be mixed
        sensedPatterns = objectA[1][:-1] + [objectA[1][-1] | objectB[1][-1]]

        # feed sensed patterns first time
        # every one feels the correct object, except first column which feels
        # the union (reminder: lateral input are delayed)
        activeRepresentations = self._getActiveRepresentations()
        self.inferMultipleColumns(
            feedforwardPatterns=sensedPatterns,
            activeRepresentations=activeRepresentations,
            neighborsIndices=neighborsIndices,
        )
        firstSensedRepresentations = [
            activeRepresentationsA[0] | activeRepresentationsB[0],
            activeRepresentationsA[1],
            activeRepresentationsA[2],
            activeRepresentationsA[3] | activeRepresentationsB[3]
        ]
        self.assertEqual(
            firstSensedRepresentations,
            self._getActiveRepresentations()
        )

        # feed sensed patterns second time
        # the distal predictions are still ambiguous in C1, but disambiguated
        # in C4
        activeRepresentations = self._getActiveRepresentations()
        self.inferMultipleColumns(
            feedforwardPatterns=sensedPatterns,
            activeRepresentations=activeRepresentations,
            neighborsIndices=neighborsIndices,
        )
        secondSensedRepresentations = [
            activeRepresentationsA[0] | activeRepresentationsB[0],
            activeRepresentationsA[1],
            activeRepresentationsA[2],
            activeRepresentationsA[3]
        ]
        self.assertEqual(
            secondSensedRepresentations,
            self._getActiveRepresentations()
        )

        # feed sensed patterns third time
        # this time, it is all disambiguated
        activeRepresentations = self._getActiveRepresentations()
        self.inferMultipleColumns(
            feedforwardPatterns=sensedPatterns,
            activeRepresentations=activeRepresentations,
            neighborsIndices=neighborsIndices,
        )
        self.assertEqual(
            activeRepresentationsA,
            self._getActiveRepresentations()
        )


    def testMutualDisambiguationThroughUnions(self):
        """
        Learns three object in two different columns.

        Feed ambiguous sensations, A u B and B u C. The system should narrow down
        to B.
        """
        self.init(numCols=2)
        neighborsIndices = [[1], [0]]

        objectA = self.generateObject(numPatterns=5, numCols=2)
        objectB = self.generateObject(numPatterns=5, numCols=2)
        objectC = self.generateObject(numPatterns=5, numCols=2)

        # learn object
        self.learnMultipleColumns(
            objectA,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsA = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectB,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsB = self._getActiveRepresentations()

        # learn object
        self.learnMultipleColumns(
            objectC,
            numRepetitions=3,
            neighborsIndices=neighborsIndices,
            randomOrder=True,
            newObject=True
        )
        activeRepresentationsC = self._getActiveRepresentations()

        # create sensed patterns (ambiguous)
        sensedPatterns = [objectA[1][0] | objectB[1][0], objectB[2][1] | objectC[2][1]]

        for pooler in self.poolers:
            pooler.reset()

        # feed sensed patterns first time
        # the L2 representations should be ambiguous
        activeRepresentations = self._getActiveRepresentations()
        self.inferMultipleColumns(
            feedforwardPatterns=sensedPatterns,
            activeRepresentations=activeRepresentations,
            neighborsIndices=neighborsIndices,
        )
        firstRepresentations = [
            activeRepresentationsA[0] | activeRepresentationsB[0],
            activeRepresentationsB[1] | activeRepresentationsC[1]
        ]
        self.assertEqual(
            firstRepresentations,
            self._getActiveRepresentations()
        )

        # feed a second time, distal predictions should disambiguate
        activeRepresentations = self._getActiveRepresentations()
        self.inferMultipleColumns(
            feedforwardPatterns=sensedPatterns,
            activeRepresentations=activeRepresentations,
            neighborsIndices=neighborsIndices,
        )

        # check that representations are unique, being slightly tolerant
        self.assertLessEqual(
            len(self._getActiveRepresentations()[0] - activeRepresentationsB[0]),
            5,
        )

        self.assertLessEqual(
            len(self._getActiveRepresentations()[1] - activeRepresentationsB[1]),
            5,
        )

        self.assertGreaterEqual(
            len(self._getActiveRepresentations()[0] & activeRepresentationsB[0]),
            35,
        )

        self.assertGreaterEqual(
            len(self._getActiveRepresentations()[1] & activeRepresentationsB[1]),
            35,
        )


    def setUp(self):
        """
        Sets up the test.
        """
        # single column case
        self.pooler = None

        # multi column case
        self.poolers = []

        # create pattern machine
        self.proximalPatternMachine = PatternMachine(
            n=self.inputWidth,
            w=self.numOutputActiveBits,
            num=200,
            seed=self.seed
        )

        self.patternId = 0
        np.random.seed(self.seed)


    # Wrappers around ColumnPooler API

    def learn(self,
            feedforwardPatterns,
            lateralPatterns=None,
            numRepetitions=1,
            randomOrder=True,
            newObject=True):
        """
        Parameters:
        ----------------------------
        Learns a single object, with the provided patterns.

        @param     feedforwardPatterns     (list(set))
                         List of proximal input patterns

        @param     lateralPatterns             (list(list(iterable)))
                         List of distal input patterns, or None. If no lateral input is
                         used. The outer list is expected to have the same length as
                         feedforwardPatterns, whereas each inner list's length is the
                         number of cortical columns which are distally connected to the
                         pooler.

        @param     numRepetitions                (int)
                         Number of times the patterns will be fed

        @param     randomOrder                     (bool)
                         If true, the order of patterns will be shuffled at each
                         repetition

        """
        if newObject:
            self.pooler.mmClearHistory()
            self.pooler.reset()

        # set-up
        indices = list(range(len(feedforwardPatterns)))
        if lateralPatterns is None:
            lateralPatterns = [[] for _ in range(len(feedforwardPatterns))]

        for _ in range(numRepetitions):
            if randomOrder:
                np.random.shuffle(indices)

            for idx in indices:
                self.pooler.compute(sorted(feedforwardPatterns[idx]),
                                    [sorted(lateralPattern)
                                     for lateralPattern in lateralPatterns[idx]],
                                    learn=True)


    def infer(self,
                feedforwardPattern,
                lateralInputs=(),
                printMetrics=False):
        """
        Feeds a single pattern to the column pooler (as well as an eventual lateral
        pattern).

        Parameters:
        ----------------------------
        @param feedforwardPattern             (set)
             Input proximal pattern to the pooler

        @param lateralInputs                        (list(set))
             Input distal patterns to the pooler (one for each neighboring CC's)


        @param printMetrics                         (bool)
             If true, will print cell metrics

        """
        self.pooler.compute(sorted(feedforwardPattern),
                                    [sorted(lateralInput)
                                     for lateralInput in lateralInputs],
                                    learn=False)

        if printMetrics:
            print(self.pooler.mmPrettyPrintMetrics(self.pooler.mmGetDefaultMetrics()))


    # Helper functions

    def generatePattern(self):
        """
        Returns a random proximal input pattern.
        """
        pattern = self.proximalPatternMachine.get(self.patternId)
        self.patternId += 1
        return pattern


    def generateObject(self, numPatterns, numCols=1):
        """
        Creates a list of patterns, for a given object.

        If numCols > 1 is given, a list of list of patterns will be returned.
        """
        if numCols == 1:
            return [self.generatePattern() for _ in range(numPatterns)]

        else:
            patterns = []
            for _ in range(numPatterns):
                patterns.append([self.generatePattern() for _ in range(numCols)])
            return patterns


    def init(self, overrides=None, numCols=1):
        """
        Creates the column pooler with specified parameter overrides.

        Except for the specified overrides and problem-specific parameters, used
        parameters are implementation defaults.
        """
        params = {
            "inputWidth": self.inputWidth,
            "lateralInputWidths": [self.outputWidth]*(numCols-1),
            "cellCount": self.outputWidth,
            "sdrSize": self.numOutputActiveBits,
            "minThresholdProximal": 10,
            "sampleSizeProximal": 20,
            "connectedPermanenceProximal": 0.6,
            "initialDistalPermanence": 0.51,
            "activationThresholdDistal": 10,
            "sampleSizeDistal": 20,
            "connectedPermanenceDistal": 0.6,
            "seed": self.seed,
        }
        if overrides is None:
            overrides = {}
        params.update(overrides)

        if numCols == 1:
            self.pooler = MonitoredColumnPooler(**params)
        else:
            # TODO: We need a different seed for each pooler otherwise each one
            # outputs an identical representation. Use random seed for now but ideally
            # we would set different specific seeds for each pooler
            params['seed']=0
            self.poolers = [MonitoredColumnPooler(**params) for _ in range(numCols)]


    def _getActiveRepresentation(self):
        """
        Retrieves the current active representation in the pooler.
        """
        if self.pooler is None:
            raise ValueError("No pooler has been instantiated")

        return set(self.pooler.getActiveCells())


    # Multi-column testing

    def learnMultipleColumns(self,
                             feedforwardPatterns,
                             numRepetitions=1,
                             neighborsIndices=None,
                             randomOrder=True,
                             newObject=True):
        """
        Learns a single object, feeding it through the multiple columns.

        Parameters:
        ----------------------------
        Learns a single object, with the provided patterns.

        @param     feedforwardPatterns     (list(list(set)))
                         List of proximal input patterns (one for each pooler).


        @param     neighborsIndices            (list(list))
                         List of column indices each column received input from.

        @param     numRepetitions                (int)
                         Number of times the patterns will be fed

        @param     randomOrder                     (bool)
                         If true, the order of patterns will be shuffled at each
                         repetition

        """
        if newObject:
            for pooler in self.poolers:
                pooler.mmClearHistory()
                pooler.reset()

        # use different set of pattern indices to allow random orders
        indices = [list(range(len(feedforwardPatterns)))] * len(self.poolers)
        prevActiveCells = [set() for _ in range(len(self.poolers))]

        # by default, all columns are neighbors
        if neighborsIndices is None:
            neighborsIndices = [
                list(range(i)) + list(range(i+1, len(self.poolers)))
                for i in range(len(self.poolers))
            ]

        for _ in range(numRepetitions):

            # independently shuffle pattern orders if necessary
            if randomOrder:
                for idx in indices:
                    np.random.shuffle(idx)

            for i in range(len(indices[0])):
                # Train each column
                for col, pooler in enumerate(self.poolers):
                    # get union of relevant lateral representations
                    lateralInputs = [sorted(activeCells) for presynapticCol, activeCells in enumerate(prevActiveCells)
                                     if col != presynapticCol]

                    pooler.compute(sorted(feedforwardPatterns[indices[col][i]][col]), lateralInputs, learn=True)

                prevActiveCells = self._getActiveRepresentations()


    def inferMultipleColumns(self,
                             feedforwardPatterns,
                             activeRepresentations,
                             neighborsIndices=None,
                             printMetrics=False,
                             reset=False):
        """
        Feeds a single pattern to the column pooler (as well as an eventual lateral
        pattern).

        Parameters:
        ----------------------------
        @param feedforwardPattern             (list(set))
                     Input proximal patterns to the pooler (one for each column)

        @param activeRepresentations        (list(set))
                     Active representations in the columns at the previous step.

        @param neighborsIndices                 (list(list))
                     List of column indices each column received input from.

        @param printMetrics                         (bool)
                     If true, will print cell metrics

        """
        if reset:
            for pooler in self.poolers:
                pooler.reset()

        # by default, all columns are neighbors
        if neighborsIndices is None:
            neighborsIndices = [
                list(range(i)) + list(range(i+1, len(self.poolers)))
                for i in range(len(self.poolers))
            ]

        for col, pooler in enumerate(self.poolers):
            # get union of relevant lateral representations
            lateralInputs = [sorted(activeCells) for presynapticCol, activeCells in enumerate(activeRepresentations) if col != presynapticCol]

            pooler.compute(sorted(feedforwardPatterns[col]), lateralInputs, learn=False)

        if printMetrics:
            for pooler in self.poolers:
                print(pooler.mmPrettyPrintMetrics(pooler.mmGetDefaultMetrics()))


    def _getActiveRepresentations(self):
        """
        Retrieves the current active representations in the poolers.
        """
        if len(self.poolers) == 0:
            raise ValueError("No pooler has been instantiated")

        return [set(pooler.getActiveCells()) for pooler in self.poolers]



if __name__ == "__main__":
    unittest.main()
