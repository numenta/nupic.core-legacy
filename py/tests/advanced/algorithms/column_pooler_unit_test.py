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

from htm.advanced.algorithms.column_pooler import ColumnPooler
from htm.advanced.algorithms.monitor_mixin.column_pooler_mixin import ColumnPoolerMonitorMixin


class MonitoredColumnPooler(ColumnPoolerMonitorMixin, ColumnPooler):
    pass


class ColumnPoolerTest(unittest.TestCase):
    """
    Simplistic tests of the ColumnPooler region, focusing on underlying
    implementation.
    """

    def _initializeDefaultPooler(self, **kwargs):
        """Initialize and return a default ColumnPooler """

        args = {
            "inputWidth": 2048 * 8,
            "cellCount": 2048,
        }

        args.update(kwargs)

        return ColumnPooler(**args)


    def testConstructor(self):
        """Create a simple instance and test the constructor."""

        pooler = self._initializeDefaultPooler()

        self.assertEqual(pooler.numberOfCells(), 2048, "Incorrect number of cells")

        self.assertEqual(pooler.numberOfInputs(), 16384, "Incorrect number of inputs")

        self.assertEqual(
            pooler.numberOfProximalSynapses(list(range(2048))),
            0,
            "Should be no synapses on initialization"
        )

        self.assertEqual(
            pooler.numberOfConnectedProximalSynapses(list(range(2048))),
            0,
            "Should be no connected synapses on initialization"
        )


    def testInitialNullInputLearnMode(self):
        """Tests with no input in the beginning. """

        pooler = self._initializeDefaultPooler()

        # Should be no active cells in beginning
        self.assertEqual(len(pooler.getActiveCells()), 0, "Incorrect number of active cells")

        # After computing with no input should have 40 active cells
        pooler.compute(feedforwardInput=(), learn=True)
        objectSDR1 = set(pooler.getActiveCells())
        self.assertEqual(len(objectSDR1), 40, "Incorrect number of active cells")

        # Should be no active cells after reset
        pooler.reset()
        self.assertEqual(len(pooler.getActiveCells()), 0, "Incorrect number of active cells")

        # Computing again with no input should lead to different 40 active cells
        pooler.compute(feedforwardInput=(), learn=True)
        objectSDR2 = set(pooler.getActiveCells())
        self.assertEqual(len(objectSDR2), 40,
            "Incorrect number of active cells")
        self.assertLess(len(objectSDR1 & objectSDR2), 5, "SDRs not sufficiently different")


    def testInitialProximalLearning(self):
        """Tests the first few steps of proximal learning. """

        pooler = MonitoredColumnPooler(
            inputWidth=2048 * 8,
            cellCount=2048
        )


        # Get initial activity
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=True)
        self.assertEqual(len(pooler.getActiveCells()), 40, "Incorrect number of active cells")
        objectSDR = set(pooler.getActiveCells())

        # Ensure we've added correct number synapses on the active cells
        self.assertEqual(
            pooler.mmGetTraceNumProximalSynapses().data[-1],
            40*20,
            "Incorrect number of nonzero permanences on active cells"
        )

        # Ensure they are all connected
        self.assertEqual(
            pooler.numberOfConnectedProximalSynapses(pooler.getActiveCells()),
            40*20,
            "Incorrect number of connected synapses on active cells"
        )

        # As multiple different feedforward inputs come in, the same set of cells
        # should be active.
        pooler.compute(feedforwardInput=list(range(100, 140)), learn=True)
        self.assertEqual(objectSDR, set(pooler.getActiveCells()), "Activity is not consistent for same input")

        # Ensure we've added correct number of new synapses on the active cells
        self.assertEqual(
            pooler.mmGetTraceNumProximalSynapses().data[-1],
            40*40,
            "Incorrect number of nonzero permanences on active cells"
        )

        # Ensure they are all connected
        self.assertEqual(
            pooler.numberOfConnectedProximalSynapses(pooler.getActiveCells()),
            40*40,
            "Incorrect number of connected synapses on active cells"
        )

        # If there is no feedforward input we should still get the same set of
        # active cells
        pooler.compute(feedforwardInput=(), learn=True)
        self.assertEqual(objectSDR, set(pooler.getActiveCells()), "Activity is not consistent for same input")

        # Ensure we do actually add the number of synapses we want

        # In "learn new object mode", given a familiar feedforward input after reset
        # we should not get the same set of active cells
        pooler.reset()
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=True)
        self.assertNotEqual(objectSDR, set(pooler.getActiveCells()), "Activity should not be consistent for same input after reset")
        self.assertEqual(len(pooler.getActiveCells()), 40, "Incorrect number of active cells after reset")


    def testInitialInference(self):
        """Tests inference after learning one pattern. """

        pooler = self._initializeDefaultPooler()

        # Learn one pattern
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=True)
        objectSDR = set(pooler.getActiveCells())

        # Form internal distal connections
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=True)

        # Inferring on same pattern should lead to same result
        pooler.reset()
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=False)
        self.assertEqual(objectSDR, set(pooler.getActiveCells()), "Inference on pattern after learning it is incorrect")

        # Inferring with no inputs should maintain same pattern
        pooler.compute(feedforwardInput=(), learn=False)
        self.assertEqual(objectSDR, set(pooler.getActiveCells()), "Inference doesn't maintain activity with no input.")


    def testShortInferenceSequence(self):
        """Tests inference after learning two objects with two patterns. """

        pooler = self._initializeDefaultPooler()

        # Learn object one
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=True)
        object1SDR = set(pooler.getActiveCells())

        pooler.compute(feedforwardInput=list(range(100, 140)), learn=True)
        self.assertEqual(object1SDR, set(pooler.getActiveCells()), "Activity for second pattern is incorrect")

        # Learn object two
        pooler.reset()
        pooler.compute(feedforwardInput=list(range(1000, 1040)), learn=True)
        object2SDR = set(pooler.getActiveCells())

        pooler.compute(feedforwardInput=list(range(1100, 1140)), learn=True)
        self.assertEqual(object2SDR, set(pooler.getActiveCells()), "Activity for second pattern is incorrect")

        # Inferring on patterns in first object should lead to same result, even
        # after gap
        pooler.reset()
        pooler.compute(feedforwardInput=list(range(100, 140)), learn=False)
        self.assertEqual(object1SDR, set(pooler.getActiveCells()), "Inference on pattern after learning it is incorrect")

        # Inferring with no inputs should maintain same pattern
        pooler.compute(feedforwardInput=(), learn=False)
        self.assertEqual(object1SDR, set(pooler.getActiveCells()), "Inference doesn't maintain activity with no input.")

        pooler.reset()
        pooler.compute(feedforwardInput=list(range(0, 40)), learn=False)
        self.assertEqual(object1SDR, set(pooler.getActiveCells()), "Inference on pattern after learning it is incorrect")

        # Inferring on patterns in second object should lead to same result, even
        # after gap
        pooler.reset()
        pooler.compute(feedforwardInput=list(range(1100, 1140)), learn=False)
        self.assertEqual(object2SDR, set(pooler.getActiveCells()), "Inference on pattern after learning it is incorrect")

        # Inferring with no inputs should maintain same pattern
        pooler.compute(feedforwardInput=(), learn=False)
        self.assertEqual(object2SDR, set(pooler.getActiveCells()), "Inference doesn't maintain activity with no input.")

        pooler.reset()
        pooler.compute(feedforwardInput=list(range(1000, 1040)), learn=False)
        self.assertEqual(object2SDR, set(pooler.getActiveCells()), "Inference on pattern after learning it is incorrect")


    def testProximalLearning_SampleSize(self):
        """
        During learning, cells should attempt to have sampleSizeProximal
        active proximal synapses.

        """
        pooler = ColumnPooler(
            inputWidth=2048 * 8,
            initialProximalPermanence=0.60,
            connectedPermanenceProximal=0.50,
            sampleSizeProximal=10,
            synPermProximalDec=0,
        )

        feedforwardInput1 = list(range(10))

        pooler.compute(feedforwardInput1, learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 10, "Should connect to every active input bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 10, "Each synapse should be marked as connected.")

            presynapticCells, permanences = self._rowNonZeros(pooler, cell)

            self.assertEqual(set(presynapticCells), set(feedforwardInput1), "Should connect to every active input bit.")
            for perm in permanences:
                self.assertAlmostEqual(perm, 0.60, msg="Should use 'initialProximalPermanence'.")

        pooler.compute(list(range(10, 20)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 20, "Should connect to every active input bit.")

        pooler.compute(list(range(15, 25)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 25,
                                             ("Should connect to every active input bit "
                                                "that it's not yet connected to."))

        pooler.compute(list(range(0, 30)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 25, "Should not grow more synapses if it had lots active.")

        pooler.compute(list(range(23, 30)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 30, "Should grow as many as it can.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 30, "Each synapse should be marked as connected.")

    def _rowNonZeros(self, pooler, cell):
        """
        Answer a tuple of preseynaptic cells and their permanences for cell.
        """
        proximalPermanences = pooler.proximalPermanences
        segments = proximalPermanences.segmentsForCell(cell)
        segment = segments[0]
        presynapticCells = [proximalPermanences.presynapticCellForSynapse(synapse) for synapse in proximalPermanences.synapsesForSegment(segment)]
        permanences = [proximalPermanences.permanenceForSynapse(synapse) for synapse in proximalPermanences.synapsesForSegment(segment)]
        return presynapticCells, permanences

    def testProximalLearning_NoSampling(self):
        """
        With sampleSize -1, during learning each cell should connect to every
        active bit.
        """
        pooler = ColumnPooler(
            inputWidth=2048 * 8,
            initialProximalPermanence=0.60,
            connectedPermanenceProximal=0.50,
            sampleSizeProximal=-1,
            synPermProximalDec=0,
        )

        feedforwardInput1 = list(range(10))

        pooler.compute(feedforwardInput1, learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 10, "Should connect to every active input bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 10, "Each synapse should be marked as connected.")

            presynapticCells, permanences = self._rowNonZeros(pooler, cell)

            self.assertEqual(set(presynapticCells), set(feedforwardInput1), "Should connect to every active input bit.")
            for perm in permanences:
                self.assertAlmostEqual(perm, 0.60, msg="Should use 'initialProximalPermanence'.")

        pooler.compute(list(range(30)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 30, "Should grow synapses to every unsynapsed active bit.")

        pooler.compute(list(range(25, 30)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 30, "Every bit is synapsed so nothing else should grow.")

        pooler.compute(list(range(125, 130)), learn=True)

        for cell in pooler.getActiveCells():
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 35, "Should grow synapses to every unsynapsed active bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 35, "Each synapse should be marked as connected.")


    def testProximalLearning_InitiallyDisconnected(self):
        """
        If the initialProximalPermanence is below the connectedPermanence, new
        synapses should not be marked as connected.

        """
        pooler = self._initializeDefaultPooler(
            sdrSize=40,
            initialProximalPermanence=0.45,
            connectedPermanenceProximal=0.50,
            sampleSizeProximal=10,
        )

        feedforwardInput = list(range(10))

        pooler.compute(feedforwardInput, learn=True)

        activeCells = pooler.getActiveCells()
        self.assertEqual(len(activeCells), 40)

        for cell in activeCells:
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 10, "Should connect to every active input bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 0,
                             "The synapses shouldn't have a high enough permanence"
                             " to be connected.")


    def testProximalLearning_ReinforceExisting(self):
        """
        When a cell has a synapse to an active input bit, increase its permanence by
        'synPermProximalInc'.

        """
        pooler = self._initializeDefaultPooler(
            sdrSize=40,
            initialProximalPermanence=0.45,
            connectedPermanenceProximal=0.50,
            sampleSizeProximal=10,
            synPermProximalInc=0.1,
            synPermProximalDec=0.0,
        )

        # Grow some synapses.
        pooler.compute(list(range(0, 10)), learn=True)
        pooler.compute(list(range(10, 20)), learn=True)

        # Reinforce some of them.
        pooler.compute(list(range(0, 15)), learn=True)

        activeCells = pooler.getActiveCells()
        self.assertEqual(len(activeCells), 40)

        for cell in activeCells:
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 20, "Should connect to every active input bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 15, "Each reinforced synapse should be marked as connected.")

            presynapticCells, permanences = self._rowNonZeros(pooler, cell)

            d = dict(list(zip(presynapticCells, permanences)))
            for presynapticCell in range(0, 15):
                perm = d[presynapticCell]
                self.assertAlmostEqual(
                    perm, 0.55,
                    msg=("Should have permanence of 'initialProximalPermanence'" " + 'synPermProximalInc'."))
            for presynapticCell in range(15, 20):
                perm = d[presynapticCell]
                self.assertAlmostEqual(
                    perm, 0.45,
                    msg="Should have permanence of 'initialProximalPermanence'")


    def testProximalLearning_PunishExisting(self):
        """
        When a cell has a synapse to an inactive input bit, decrease its permanence
        by 'synPermProximalDec'.

        """
        pooler = self._initializeDefaultPooler(
            sdrSize=40,
            initialProximalPermanence=0.55,
            connectedPermanenceProximal=0.50,
            sampleSizeProximal=10,
            synPermProximalInc=0.0,
            synPermProximalDec=0.1,
        )

        # Grow some synapses.
        pooler.compute(list(range(0, 10)), learn=True)

        # Punish some of them.
        pooler.compute(list(range(0, 5)), learn=True)

        activeCells = pooler.getActiveCells()
        self.assertEqual(len(activeCells), 40)

        for cell in activeCells:
            self.assertEqual(pooler.numberOfProximalSynapses([cell]), 10, "Should connect to every active input bit.")
            self.assertEqual(pooler.numberOfConnectedProximalSynapses([cell]), 5,
                             "Each punished synapse should no longer be marked as"
                             " connected.")

            presynapticCells, permanences = self._rowNonZeros(pooler, cell)

            d = dict(list(zip(presynapticCells, permanences)))
            for presynapticCell in range(0, 5):
                perm = d[presynapticCell]
                self.assertAlmostEqual(
                    perm, 0.55,
                    msg="Should have permanence of 'initialProximalPermanence'")
            for presynapticCell in range(5, 10):
                perm = d[presynapticCell]
                self.assertAlmostEqual(
                    perm, 0.45,
                    msg=("Should have permanence of 'initialProximalPermanence'"
                             " - 'synPermProximalDec'."))


    def testLearningWithLateralInputs(self):
        """
        With lateral inputs from other columns, test that some distal segments are
        learned on a stable set of SDRs for each new feed forward object
        """
        pooler = self._initializeDefaultPooler(lateralInputWidths=[512])

        # Object 1
        lateralInput1 = list(range(100, 140))
        pooler.compute(list(range(0, 40)), [lateralInput1], learn=True)

        # Get initial SDR for first object from pooler.
        activeCells = pooler.getActiveCells()

        # Cells corresponding to that initial SDR should have started learning on
        # their distal segments.
        self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                         40,
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                         40*20,
                         "Incorrect number of synapses after learning")

        # When the cells have been active for another timestep, they should grow
        # internal distal connections, using a new segment on each cell.
        pooler.compute(list(range(40, 80)), [lateralInput1], learn=True)
        self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                         80,
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                         80*20,
                         "Incorrect number of synapses after learning")

        # Cells corresponding to that initial SDR should continue to learn new
        # synapses on that same set of segments. There should be no segments on any
        # other cells.
        pooler.compute(list(range(80, 120)), [lateralInput1], learn=True)

        self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                         80,
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSegments(list(range(2048))),
                         80,
                         "Extra segments on other cells after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                         80*20,
                         "Incorrect number of synapses after learning")


        # Object 2
        pooler.reset()
        lateralInput2 = list(range(200, 240))
        pooler.compute(list(range(120, 160)), [lateralInput2], learn=True)

        # Get initial SDR for second object from pooler.
        activeCellsObject2 = set(pooler.getActiveCells())
        uniqueCellsObject2 = set(activeCellsObject2) - set(activeCells)
        numCommonCells = len(set(activeCells).intersection(set(activeCellsObject2)))


        # Cells corresponding to that initial SDR should have started learning
        # on their distal segments.
        self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                         len(uniqueCellsObject2),
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                         len(uniqueCellsObject2)*20,
                         "Incorrect number of synapses after learning")
        self.assertLess(numCommonCells, 5, "Too many common cells across objects")

        # When the cells have been active for another timestep, they should grow
        # internal distal connections, using a new segment on each cell.
        pooler.compute(list(range(160, 200)), [lateralInput2], learn=True)
        self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                         len(uniqueCellsObject2)*2,
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                         len(uniqueCellsObject2)*2*20,
                         "Incorrect number of synapses after learning")

        # Cells corresponding to that initial SDR should continue to learn new
        # synapses on that same set of segments. There should be no segments on any
        # other cells.
        pooler.compute(list(range(200, 240)), [lateralInput2], learn=True)
        self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                         len(uniqueCellsObject2)*2,
                         "Incorrect number of segments after learning")
        self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                         len(uniqueCellsObject2)*2*20,
                         "Incorrect number of synapses after learning")


    def testInferenceWithLateralInputs(self):
        """
        After learning two objects, test that inference behaves as expected in
        a variety of scenarios.
        """
        pooler = self._initializeDefaultPooler(
            lateralInputWidths=[512, 512],
        )

        # Feed-forward representations:
        # Object 1 = union(range(0,40), range(40,80), range(80,120))
        # Object 2 = union(range(120, 160), range(160,200), range(200,240))
        feedforwardInputs = [
            [list(range(0, 40)), list(range(40, 80)), list(range(80, 120))],
            [list(range(120, 160)), list(range(160, 200)), list(range(200, 240))]
        ]

        # Lateral representations:
        # Object 1, Col 1 = range(200,240)
        # Object 2, Col 1 = range(240,280)
        # Object 1, Col 2 = range(100,140)
        # Object 2, Col 2 = range(140,180)
        objectLateralInputs = [
            [list(range(200, 240)), list(range(100, 140))],    # Object 1
            [list(range(240, 280)), list(range(140, 180))],    # Object 2
        ]

        # Train pooler on two objects, three iterations per object
        objectRepresentations = []
        for obj in range(2):
            pooler.reset()
            for _ in range(3): # three iterations
                for f in range(3): # three features per object
                    pooler.compute(feedforwardInputs[obj][f], objectLateralInputs[obj], learn=True)

            objectRepresentations += [set(pooler.getActiveCells())]

        # With no lateral support, BU for O1 feature 0 + O2 feature 1.
        # End up with object representations for O1+O2.
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])), learn=False)
        self.assertEqual(set(pooler.getActiveCells()), 
                         objectRepresentations[0] | objectRepresentations[1],
                         "Incorrect object representations - expecting union of objects")

        # If you now get no input, should maintain the representation
        pooler.compute(feedforwardInput=(), learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                        objectRepresentations[0] | objectRepresentations[1],
                        "Incorrect object representations - expecting union is maintained")


        # Test case where you have two objects in bottom up representation, but
        # only one in lateral. In this case the laterally supported object
        # should dominate.

        # Test lateral from first column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[objectLateralInputs[0][0], ()],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                         objectRepresentations[0],
                         "Incorrect object representations - expecting single object")

        # Test lateral from second column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[(), objectLateralInputs[0][1]],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()), 
                         objectRepresentations[0],
                         "Incorrect object representations - expecting single object")

        # Test lateral from both columns
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=objectLateralInputs[0],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                            objectRepresentations[0],
                            "Incorrect object representations - expecting single object")


        # Test case where you have bottom up for O1, and lateral for O2. In
        # this case the bottom up one, O1, should dominate.
        pooler.reset()
        pooler.compute(feedforwardInput=feedforwardInputs[0][0],
                         lateralInputs=objectLateralInputs[1],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                         objectRepresentations[0],
                         "Incorrect object representations - expecting first object")

        # Test case where you have BU support O1+O2 with no lateral input Then see
        # no input but get lateral support for O1. Should converge to O1 only.
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[],
                         learn=False)

        # No bottom input, but lateral support for O1
        pooler.compute(feedforwardInput=(), lateralInputs=objectLateralInputs[0], learn=False)

        self.assertEqual(set(pooler.getActiveCells()),
                        objectRepresentations[0],
                        "Incorrect object representations - expecting first object")


        # TODO: more tests we could write:
        # Test case where you have two objects in bottom up representation, and
        # same two in lateral. End up with both active.

        # Test case where you have O1, O2 in bottom up representation, but
        # O1, O3 in lateral. In this case should end up with O1.

        # Test case where you have BU support for two objects, less than adequate
        # lateral support (below threshold) for one of them. Should end up with both
        # BU objects.


    def testInferenceWithChangingLateralInputs0(self):
        """
        # Test case where the lateral inputs change while learning an object.
        # The same distal segments should continue to sample from the new inputs.
        # During inference any of these lateral inputs should cause the pooler
        # to disambiguate appropriately object 0.
        """
        pooler = self._initializeDefaultPooler( lateralInputWidths=[512, 512])

        # Feed-forward representations:
        # Object 1 = union(range(0,40), range(40,80), range(80,120))
        # Object 2 = union(range(120, 160), range(160,200), range(200,240))
        feedforwardInputs = [
            [list(range(0, 40)), list(range(40, 80)), list(range(80, 120))],
            [list(range(120, 160)), list(range(160, 200)), list(range(200, 240))]
        ]

        # Lateral representations:
        # Object 1, Col 1 = range(200,240)
        # Object 2, Col 1 = range(240,280)
        # Object 1, Col 2 = range(100,140)
        # Object 2, Col 2 = range(140,180)
        objectLateralInputs = [
            [list(range(200, 240)), list(range(100, 140))],    # Object 1
            [list(range(240, 280)), list(range(140, 180))],    # Object 2
        ]

        # Train pooler on two objects. For each object we go through three
        # iterations using just lateral input from first column. Then repeat with
        # second column.
        objectRepresentations = []
        for obj in range(2):
            pooler.reset()
            for col in range(2):
                lateralInputs = [[], []]
                lateralInputs[col] = objectLateralInputs[obj][col]
                for _ in range(3): # three iterations
                    for f in range(3): # three features per object
                        pooler.compute(feedforwardInputs[obj][f], lateralInputs, learn=True)
            objectRepresentations += [set(pooler.getActiveCells())]

        # We want to ensure that the learning for each cell happens on one distal
        # segment only. Some cells could in theory be common across both
        # representations so we just check the unique ones.
        # TODO: this test currently fails due to NuPIC issue #3268
        # uniqueCells = objectRepresentations[0].symmetric_difference(objectRepresentations[1])
        # connections = pooler.tm.connections
        # for cell in uniqueCells:
        #     self.assertEqual(connections.segmentsForCell(cell), 1,
        #                                        "Too many segments")

        # Test case where both objects are present in bottom up representation, but
        # only one in lateral. In this case the laterally supported object
        # should dominate after second iteration.

        # Test where lateral input is from first column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[objectLateralInputs[0][0], ()],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                         objectRepresentations[0],
                         "Incorrect object representations - expecting single object")

        # Test lateral from second column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[(), objectLateralInputs[0][1]],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                        objectRepresentations[0],
                        "Incorrect object representations - expecting single object")

    def testInferenceWithChangingLateralInputs1(self):
        """
        # Test case where the lateral inputs change while learning an object.
        # The same distal segments should continue to sample from the new inputs.
        # During inference any of these lateral inputs should cause the pooler
        # to disambiguate appropriately object 1.
        """
        pooler = self._initializeDefaultPooler( lateralInputWidths=[512, 512])

        # Feed-forward representations:
        # Object 1 = union(range(0,40), range(40,80), range(80,120))
        # Object 2 = union(range(120, 160), range(160,200), range(200,240))
        feedforwardInputs = [
            [list(range(0, 40)), list(range(40, 80)), list(range(80, 120))],
            [list(range(120, 160)), list(range(160, 200)), list(range(200, 240))]
        ]

        # Lateral representations:
        # Object 1, Col 1 = range(200,240)
        # Object 2, Col 1 = range(240,280)
        # Object 1, Col 2 = range(100,140)
        # Object 2, Col 2 = range(140,180)
        objectLateralInputs = [
            [list(range(200, 240)), list(range(100, 140))],    # Object 1
            [list(range(240, 280)), list(range(140, 180))],    # Object 2
        ]

        # Train pooler on two objects. For each object we go through three
        # iterations using just lateral input from first column. Then repeat with
        # second column.
        objectRepresentations = []
        for obj in range(2):
            pooler.reset()
            for col in range(2):
                lateralInputs = [[], []]
                lateralInputs[col] = objectLateralInputs[obj][col]
                for _ in range(3): # three iterations
                    for f in range(3): # three features per object
                        pooler.compute(feedforwardInputs[obj][f], lateralInputs, learn=True)
            objectRepresentations += [set(pooler.getActiveCells())]

        # We want to ensure that the learning for each cell happens on one distal
        # segment only. Some cells could in theory be common across both
        # representations so we just check the unique ones.
        # TODO: this test currently fails due to NuPIC issue #3268
        # uniqueCells = objectRepresentations[0].symmetric_difference(objectRepresentations[1])
        # connections = pooler.tm.connections
        # for cell in uniqueCells:
        #     self.assertEqual(connections.segmentsForCell(cell), 1,
        #                                        "Too many segments")

        # Test case where both objects are present in bottom up representation, but
        # only one in lateral. In this case the laterally supported object
        # should dominate after second iteration.

        # Test where lateral input is from first column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[objectLateralInputs[1][0], ()],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                         objectRepresentations[1],
                         "Incorrect object representations - expecting single object")

        # Test lateral from second column
        pooler.reset()
        pooler.compute(feedforwardInput=list(set(feedforwardInputs[0][0]) | set(feedforwardInputs[1][1])),
                         lateralInputs=[(), objectLateralInputs[1][1]],
                         learn=False)
        self.assertEqual(set(pooler.getActiveCells()),
                        objectRepresentations[1],
                        "Incorrect object representations - expecting single object")



if __name__ == "__main__":
    unittest.main()
