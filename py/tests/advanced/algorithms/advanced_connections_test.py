# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Frederick C. Rotbart 
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

from htm.bindings.sdr import SDR
from htm.bindings.math import Random
from htm.advanced.algorithms.connections import Connections

import numpy as np

from random import shuffle

NUM_CELLS = 4096

class ConnectionsTest(unittest.TestCase):
  
    def _getPresynapticCells(self, connections, segment, threshold):
        """
        Return a set of presynaptic cells that have synapses to segment.
        """
        return set([connections.presynapticCellForSynapse(synapse) for synapse in connections.synapsesForSegment(segment) 
                    if connections.permanenceForSynapse(synapse) >= threshold])

    def testAdaptShouldNotRemoveSegments(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.51) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            self.assertEqual(len(segments), 1, "Segments were destroyed.")
            segment = segments[0]
            connections.adaptSegment(segment, inputSDR, 0.1, 0.001, False)
        
            segments = connections.segmentsForCell(cell)
            self.assertEqual(len(segments), 1, "Segments were destroyed.")
            segment = segments[0]
    
    def testAdaptShouldRemoveSegments(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.51) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            self.assertEqual(len(segments), 1, "Segments were prematurely destroyed.")
            segment = segments[0]
            connections.adaptSegment(segment, inputSDR, 0.1, 0.001, True)
            segments = connections.segmentsForCell(cell)
            self.assertEqual(len(segments), 0, "Segments were not destroyed.")
    
    def testAdaptShouldIncrementSynapses(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        presynaptic_input_set = set(presynaptic_input)
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.51) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)          
            connections.adaptSegment(segment, inputSDR, 0.1, 0.001, True)
        
            presynamptic_cells = self._getPresynapticCells(connections, segment, 0.2)
            self.assertEqual(presynamptic_cells, presynaptic_input_set, "Missing synapses")
        
            presynamptic_cells = self._getPresynapticCells(connections, segment, 0.3)
            self.assertEqual(presynamptic_cells, set(), "Too many synapses")
    
    def testAdaptShouldDecrementSynapses(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        presynaptic_input_set = set(presynaptic_input)
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.51) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)
              
            connections.adaptSegment(segment, inputSDR, 0.1, 0.0, False)
        
            presynamptic_cells = self._getPresynapticCells(connections, segment, 0.2)
            self.assertEqual(presynamptic_cells, presynaptic_input_set, "Missing synapses")
        
        presynaptic_input1 = list(range(0, 5))
        presynaptic_input_set1 = set(presynaptic_input1)
        inputSDR.sparse = presynaptic_input1
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            connections.adaptSegment(segment, inputSDR, 0.0, 0.1, False)
        
        
            presynamptic_cells = self._getPresynapticCells(connections, segment, 0.2)
            self.assertEqual(presynamptic_cells, presynaptic_input_set1, "Too many synapses")
        
            presynamptic_cells = self._getPresynapticCells(connections, segment, 0.1)
            self.assertEqual(presynamptic_cells, presynaptic_input_set, "Missing synapses")
    
    
    def testNumSynapses(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.3) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
          
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)
              
            connections.adaptSegment(segment, inputSDR, 0.1, 0.0, False)
        
            num_synapses = connections.numSynapses(segment)
            self.assertEqual(num_synapses, len(presynaptic_input), "Missing synapses")
            
        self.assertEqual(connections.numSynapses(), len(presynaptic_input) * 40, "Missing synapses")
      
    
    def testNumConnectedSynapses(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        connections = Connections(NUM_CELLS, 0.2) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)
              
            connections.adaptSegment(segment, inputSDR, 0.1, 0.0, False)
        
            connected_synapses = connections.numConnectedSynapses(segment)
            self.assertEqual(connected_synapses, len(presynaptic_input), "Missing synapses")
        
        presynaptic_input1 = list(range(0, 5))
        inputSDR.sparse = presynaptic_input1
        
        total_connected = 0
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            connections.adaptSegment(segment, inputSDR, 0.0, 0.1, False)
        
            connected_synapses = connections.numConnectedSynapses(segment)
            self.assertEqual(connected_synapses, len(presynaptic_input1), "Missing synapses")
            
            total_connected += connected_synapses
        
            connected_synapses = connections.numSynapses(segment)
            self.assertEqual(connected_synapses, len(presynaptic_input), "Missing synapses")
        
        self.assertEqual(total_connected, len(presynaptic_input1) * 40, "Missing synapses")
    
    def testComputeActivity(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input = list(range(0, 10))
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        l = len(presynaptic_input)
        
        connections = Connections(NUM_CELLS, 0.51, False) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        numActiveConnectedSynapsesForSegment = connections.computeActivity(inputSDR, False)
        for count in numActiveConnectedSynapsesForSegment:
            self.assertEqual(count, 0, "Segment should not be active")
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)
            
        numActiveConnectedSynapsesForSegment = connections.computeActivity(inputSDR, False)
        for count in numActiveConnectedSynapsesForSegment:
            self.assertEqual(count, 0, "Segment should not be active")
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]        
            connections.adaptSegment(segment, inputSDR, 0.5, 0.0, False)
            
        active_cells_set = set(active_cells)
        numActiveConnectedSynapsesForSegment = connections.computeActivity(inputSDR, False)
        for cell, count in enumerate(numActiveConnectedSynapsesForSegment):
            if cell in active_cells_set:
                self.assertEqual(count, l, "Segment should be active")
            else:
                self.assertEqual(count, 0, "Segment should not be active")
          
    def _learn(self, connections, active_cells, presynaptic_input):
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input
        
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            for c in presynaptic_input:
                connections.createSynapse(segment, c, 0.1)
            
        for cell in active_cells:
            segments = connections.segmentsForCell(cell)
            segment = segments[0]
            connections.adaptSegment(segment, inputSDR, 0.5, 0.0, False)
    
    def testComputeActivityUnion(self):
        """
        Test that connections are generated on predefined segments.
        """
        random = Random(1981)
        active_cells = np.array(random.sample(np.arange(0, NUM_CELLS, 1, dtype="uint32"), 40), dtype="uint32")
        active_cells.sort()
        
        presynaptic_input1 = list(range(0, 10))
        presynaptic_input2 = list(range(10, 20))
         
        connections = Connections(NUM_CELLS, 0.51, False) 
        for i in range(NUM_CELLS):
            connections.createSegment(i, 1)
        
        self._learn(connections, active_cells, presynaptic_input1)
        self._learn(connections, active_cells, presynaptic_input2)
        
        numSynapses = connections.numSynapses()
        self.assertNotEqual(numSynapses, 40, "There should be a synapse for each presynaptic cell")
        
        active_cells_set = set(active_cells)
        inputSDR = SDR(1024)
        inputSDR.sparse = presynaptic_input1
        
        numActiveConnectedSynapsesForSegment = connections.computeActivity(inputSDR, False)
        for cell, count in enumerate(numActiveConnectedSynapsesForSegment):
            if cell in active_cells_set:
                self.assertNotEqual(count, 0, "Segment should be active")
        
        inputSDR.sparse = presynaptic_input2
        numActiveConnectedSynapsesForSegment = connections.computeActivity(inputSDR, False)
        for cell, count in enumerate(numActiveConnectedSynapsesForSegment):
            if cell in active_cells_set:
                self.assertNotEqual(count, 0, "Segment should be active")
                
    def testMapSegmentsToCell(self):
        """
        Test that cells are correspond to segments.
        """
        connections = Connections(20, 0.2) 
        segments = [connections.createSegment(i, 1) for i in range(19, -1, -1)]
        self.assertEqual(list(range(20)), segments, "Segments were not allocated continuously")

        cells_should_be = list(range(19, -1, -1)) # Segment 19 belongs to cell 0, etc.
        cells = list(connections.mapSegmentsToCells(segments))
        self.assertEqual(cells_should_be, cells)
        
    def testSortSegmentsByCell(self):
        """
        Test that segments are sorted by cell.
        """
        connections = Connections(20, 0.2) 
        segments = [connections.createSegment(i, 1) for i in range(19, -1, -1)]
        self.assertEqual(list(range(20)), segments, "Segments were not allocated continuously")
        
        segments_should_be = list(range(19, -1, -1)) # Segment 19 belongs to cell 0, etc.
        shuffle(segments)
        sorted_segments = list(connections.sortSegmentsByCell(np.array(segments)))
        self.assertEqual(segments_should_be, sorted_segments, "Segments were not sorted by cell")
        
    def testFilterUnsortedSegmentsByCell(self):
        """
        Test that segments are filtered by cell.
        """
        connections = Connections(20, 0.2) 
        segments = [connections.createSegment(i, 1) for i in range(19, -1, -1)]
        self.assertEqual(list(range(20)), segments, "Segments were not allocated continuously")
        
        segments_should_be = [19, 10, 0] # Segment 19 belongs to cell 0, etc.
        filtered_segments = list(connections.filterSegmentsByCell(np.array(segments), [0,9,19], False))
        self.assertEqual(segments_should_be, filtered_segments, "Segments were not filtered correctly")
        
    def testFilterSortedSegmentsByCell(self):
        """
        Test that segments are filtered by cell.
        """
        connections = Connections(20, 0.2) 
        segments = [connections.createSegment(i, 1) for i in range(19, -1, -1)]
        self.assertEqual(list(range(20)), segments, "Segments were not allocated continuously")
        
        sorted_segments = connections.sortSegmentsByCell(np.array(segments))
        segments_should_be = [19, 10, 0] # Segment 19 belongs to cell 0, etc.
        filtered_segments = list(connections.filterSegmentsByCell(sorted_segments, [0,9,19], True))
        self.assertEqual(segments_should_be, filtered_segments, "Segments were not filtered correctly")


if __name__ == "__main__":
    unittest.main()
