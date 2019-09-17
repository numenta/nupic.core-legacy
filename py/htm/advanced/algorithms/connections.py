# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Frederick C. Rotbart
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

from htm.bindings.algorithms import Connections as CPPConnections
import numpy as np

class Connections(CPPConnections):
    
    def numConnectedSynapsesForCells(self, cells):
        """
        Return the number of connected synapses in the connection for the list of cells.
        """
        n = 0
        for cell in cells:
            segments = self.segmentsForCell(cell)
            for segment in segments:
                n += self.numConnectedSynapses(segment)
        return n
    
    def numSynapsesForCells(self, cells):
        """
        Return the number of connected synapses in the connection for the list of cells.
        """
        n = 0
        for cell in cells:
            segments = self.segmentsForCell(cell)
            for segment in segments:
                n += self.numSynapses(segment)
        return n
    
    def numSegmentsWithSynapses(self, cells):
        """
        Return the number of segments in the connection that have at least one synapse for the list of cells.
        """
        n = 0
        for cell in cells:
            segments = self.segmentsForCell(cell)
            for segment in segments:
                if self.numSynapses(segment) > 0:
                    n += 1
        return n
    
    def sortSegmentsByCell(self, segments):
        """
        Sort an array of segments by cell in increasing order.
        
        @param segments
            The segment array.
            
        @return:
            A sorted segment array
        """
        cells = [self.cellForSegment(s) for s in segments]
        cells_args = np.argsort(cells)
        return segments[cells_args]
    
    def filterSegmentsByCell(self, segments, cells, assumeSorted=False):
        """
        Return the subset of segments that are on the provided cells.
        
        @param segments
            The segments to filter. Must be sorted by cell.
        
        @param cells
            The cells whose segments we want to keep. Must be sorted.
        
        """
        if not assumeSorted:
            segments = self.sortSegmentsByCell(segments)

        mask = np.isin([self.cellForSegment(s) for s in segments], cells)
        return segments[mask]   

    def mapSegmentsToCells(self, segments):
        """
        Get the cell for each provided segment.
        
        @param segments
            The segments to query
        
        @param cells
            Output array with the same length as 'segments'
        """
        return np.array([self.cellForSegment(s) for s in segments], dtype=np.uint32)

    def growSynapsesToSample(self, segment, growthCandidates, maxNew, initialPermanence, rng):
        """
        For each specified segments, grow synapses to a random subset of the
        inputs that aren't already connected to the segment.
        *
        @param segment
        The segment to modify
        *
        @param inputs
        The inputs to sample
        *
        @param sampleSize
        The number of synapses to attempt to grow per segment
        *
        @param initialPermanence
        The permanence for each added synapse
        *
        @param rng
        Random number generator
        """
        presynamptic_cells = [self.presynapticCellForSynapse(synapse) for synapse in self.synapsesForSegment(segment)]
        active_cells_without_synapses = growthCandidates[np.isin(growthCandidates, presynamptic_cells, invert=True)]
        if len(active_cells_without_synapses) > maxNew:
            active_cells_without_synapses = rng.sample(np.asarray(active_cells_without_synapses, dtype="uint32"), maxNew)
        
        for c in active_cells_without_synapses:
            self.createSynapse(segment, c, initialPermanence)
    
    def getSegmentCounts(self, cells):
        """
        Get the number of segments on each of the provided cells.
        
        @param cells
        The cells to check
        
        @param counts
        Output array with the same length as 'cells'
        """
        return np.array([self.numSegments(cell) for cell in cells], dtype=np.uint32)
        
    def computeActiveSegments(self, presynapticCells, activationThreshold):
        """
        Compute the segments whose number of active synapses is greater or equal to activationThreshold
        for a vector of active presynaptic cells
        
        @param SDR presynapticCells
        The cells to check
        
        @param int activationThreshold
        The threshold that number of synapses must reach
        
        @return list
        List of segments with greater or equal number of sysnapses to activationThreshold 
        """
        overlaps = self.computeActivity(presynapticCells, False)
        return np.flatnonzero(overlaps >= activationThreshold)

