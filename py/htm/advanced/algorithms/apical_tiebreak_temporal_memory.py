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

"""An implementation of TemporalMemory"""

import numpy as np
from htm.bindings.math import Random
from htm.advanced.support import numpy_helpers as np2
from .connections import Connections
from htm.bindings.sdr import SDR




class ApicalTiebreakTemporalMemory(object):
    """
    A generalized Temporal Memory with apical dendrites that add a "tiebreak".

    Basal connections are used to implement traditional Temporal Memory.

    The apical connections are used for further disambiguation. If multiple cells
    in a minicolumn have active basal segments, each of those cells is predicted,
    unless one of them also has an active apical segment, in which case only the
    cells with active basal and apical segments are predicted.

    In other words, the apical connections have no effect unless the basal input
    is a union of SDRs (e.g. from bursting minicolumns).

    This class is generalized in two ways:

    - This class does not specify when a 'timestep' begins and ends. It exposes
        two main methods: 'depolarizeCells' and 'activateCells', and callers or
        subclasses can introduce the notion of a timestep.
    - This class is unaware of whether its 'basalInput' or 'apicalInput' are from
        internal or external cells. They are just cell numbers. The caller knows
        what these cell numbers mean, but the TemporalMemory doesn't.
    """

    def __init__(self,
                 columnCount=2048,
                 basalInputSize=0,
                 apicalInputSize=0,
                 cellsPerColumn=32,
                 activationThreshold=13,
                 reducedBasalThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.1,
                 basalPredictedSegmentDecrement=0.0,
                 apicalPredictedSegmentDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 maxSegmentsPerCell=255,
                 seed=42):
        """
        @param columnCount (int)
        The number of minicolumns

        @param basalInputSize (sequence)
        The number of bits in the basal input

        @param apicalInputSize (int)
        The number of bits in the apical input

        @param cellsPerColumn (int)
        Number of cells per column

        @param activationThreshold (int)
        If the number of active connected synapses on a segment is at least this
        threshold, the segment is said to be active.

        @param reducedBasalThreshold (int)
        The activation threshold of basal (lateral) segments for cells that have
        active apical segments. If equal to activationThreshold (default),
        this parameter has no effect.

        @param initialPermanence (float)
        Initial permanence of a new synapse

        @param connectedPermanence (float)
        If the permanence value for a synapse is greater than this value, it is said
        to be connected.

        @param minThreshold (int)
        If the number of potential synapses active on a segment is at least this
        threshold, it is said to be "matching" and is eligible for learning.

        @param sampleSize (int)
        How much of the active SDR to sample with synapses.

        @param permanenceIncrement (float)
        Amount by which permanences of synapses are incremented during learning.

        @param permanenceDecrement (float)
        Amount by which permanences of synapses are decremented during learning.

        @param basalPredictedSegmentDecrement (float)
        Amount by which segments are punished for incorrect predictions.

        @param apicalPredictedSegmentDecrement (float)
        Amount by which segments are punished for incorrect predictions.

        @param maxSynapsesPerSegment
        The maximum number of synapses per segment.

        @param maxSegmentsPerCell
        The maximum number of segments per cell.
        
        @param seed (int)
        Seed for the random number generator.
        """

        self.columnCount = columnCount
        self.cellsPerColumn = cellsPerColumn
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.reducedBasalThreshold = reducedBasalThreshold
        self.minThreshold = minThreshold
        self.sampleSize = sampleSize
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
        self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement
        self.activationThreshold = activationThreshold
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell

        self.basalConnections = Connections(columnCount*cellsPerColumn, connectedPermanence, False)
        self.apicalConnections = Connections(columnCount*cellsPerColumn, connectedPermanence, False)
        
        self.rng = Random(seed)
        self.activeCells = np.empty(0, dtype="uint32")
        self.winnerCells = np.empty(0, dtype="uint32")
        self.predictedCells = np.empty(0, dtype="uint32")
        self.predictedActiveCells = np.empty(0, dtype="uint32")
        self.activeBasalSegments = np.empty(0, dtype="uint32")
        self.activeApicalSegments = np.empty(0, dtype="uint32")
        self.matchingBasalSegments = np.empty(0, dtype="uint32")
        self.matchingApicalSegments = np.empty(0, dtype="uint32")
        self.basalPotentialOverlaps = np.empty(0, dtype="int32")
        self.apicalPotentialOverlaps = np.empty(0, dtype="int32")
        
        self.basalInputSize = basalInputSize
        self.apicalInputSize = apicalInputSize

        self.useApicalTiebreak=True
        self.useApicalModulationBasalThreshold=True

    def reset(self):
        """
        Clear all cell and segment activity.
        """
        self.activeCells = np.empty(0, dtype="uint32")
        self.winnerCells = np.empty(0, dtype="uint32")
        self.predictedCells = np.empty(0, dtype="uint32")
        self.predictedActiveCells = np.empty(0, dtype="uint32")
        self.activeBasalSegments = np.empty(0, dtype="uint32")
        self.activeApicalSegments = np.empty(0, dtype="uint32")
        self.matchingBasalSegments = np.empty(0, dtype="uint32")
        self.matchingApicalSegments = np.empty(0, dtype="uint32")
        self.basalPotentialOverlaps = np.empty(0, dtype="int32")
        self.apicalPotentialOverlaps = np.empty(0, dtype="int32")


    def depolarizeCells(self, basalInput, apicalInput, learn):
        """
        Calculate predictions.

        @param basalInput SDR
        List of active input bits for the basal dendrite segments

        @param apicalInput SDR
        List of active input bits for the apical dendrite segments

        @param learn (bool)
        Whether learning is enabled. Some TM implementations may depolarize cells
        differently or do segment activity bookkeeping when learning is enabled.
        """
        activeApicalSegments, matchingApicalSegments, apicalPotentialOverlaps = self._calculateApicalSegmentActivity(apicalInput)

        if learn or self.useApicalModulationBasalThreshold == False:
            reducedBasalThresholdCells = ()
        else:
            reducedBasalThresholdCells = self.apicalConnections.mapSegmentsToCells(activeApicalSegments)

        activeBasalSegments, matchingBasalSegments, basalPotentialOverlaps = self._calculateBasalSegmentActivity(basalInput, reducedBasalThresholdCells)

        predictedCells = self._calculatePredictedCells(activeBasalSegments, activeApicalSegments)

        self.predictedCells = predictedCells
        self.activeBasalSegments = activeBasalSegments
        self.activeApicalSegments = activeApicalSegments
        self.matchingBasalSegments = matchingBasalSegments
        self.matchingApicalSegments = matchingApicalSegments
        self.basalPotentialOverlaps = basalPotentialOverlaps
        self.apicalPotentialOverlaps = apicalPotentialOverlaps


    def activateCells(self,
                        activeColumns,
                        basalReinforceCandidates,
                        apicalReinforceCandidates,
                        basalGrowthCandidates,
                        apicalGrowthCandidates,
                        learn=True):
        """
        Activate cells in the specified columns, using the result of the previous
        'depolarizeCells' as predictions. Then learn.

        @param activeColumns (numpy array)
        List of active columns

        @param basalReinforceCandidates SDR
        List of bits that the active cells may reinforce basal synapses to.

        @param apicalReinforceCandidates SDR
        List of bits that the active cells may reinforce apical synapses to.

        @param basalGrowthCandidates (numpy array)
        List of bits that the active cells may grow new basal synapses to.

        @param apicalGrowthCandidates (numpy array)
        List of bits that the active cells may grow new apical synapses to

        @param learn (bool)
        Whether to grow / reinforce / punish synapses
        """

        # Calculate active cells
        correctPredictedCells, burstingColumns = np2.setCompare(self.predictedCells, activeColumns, self.predictedCells // self.cellsPerColumn, rightMinusLeft=True)
        newActiveCells = np.concatenate((correctPredictedCells, np2.getAllCellsInColumns(burstingColumns, self.cellsPerColumn)))

        # Calculate learning
        (learningActiveBasalSegments,
         learningMatchingBasalSegments,
         basalSegmentsToPunish,
         newBasalSegmentCells,
         learningCells) = self._calculateBasalLearning(activeColumns, burstingColumns, correctPredictedCells)

        (learningActiveApicalSegments,
         learningMatchingApicalSegments,
         apicalSegmentsToPunish,
         newApicalSegmentCells) = self._calculateApicalLearning(learningCells, activeColumns)

        # Learn
        if learn:
            # Learn on existing segments
            for learningSegments in (learningActiveBasalSegments, learningMatchingBasalSegments):
                self._learn(self.basalConnections, learningSegments, basalReinforceCandidates, basalGrowthCandidates, self.basalPotentialOverlaps)

            for learningSegments in (learningActiveApicalSegments, learningMatchingApicalSegments):
                self._learn(self.apicalConnections, learningSegments, apicalReinforceCandidates, apicalGrowthCandidates, self.apicalPotentialOverlaps)

            # Punish incorrect predictions
            if self.basalPredictedSegmentDecrement != 0.0:
                for segment in basalSegmentsToPunish:
                    self.basalConnections.adaptSegment(segment, basalReinforceCandidates, -self.basalPredictedSegmentDecrement, 0.0, False)

            if self.apicalPredictedSegmentDecrement != 0.0:
                for segment in apicalSegmentsToPunish:
                    self.apicalConnections.adaptSegment(segment, apicalReinforceCandidates, -self.apicalPredictedSegmentDecrement, 0.0, False)

            # Grow new segments
            if len(basalGrowthCandidates) > 0:
                self._learnOnNewSegments(self.basalConnections, newBasalSegmentCells, basalGrowthCandidates)

            if len(apicalGrowthCandidates) > 0:
                self._learnOnNewSegments(self.apicalConnections, newApicalSegmentCells, apicalGrowthCandidates)

        # Save the results
        newActiveCells.sort()
        learningCells.sort()
        self.activeCells = newActiveCells
        self.winnerCells = learningCells
        self.predictedActiveCells = correctPredictedCells


    def _calculateBasalLearning(self, activeColumns, burstingColumns, correctPredictedCells):
        """
        Basic Temporal Memory learning. Correctly predicted cells always have
        active basal segments, and we learn on these segments. In bursting
        columns, we either learn on an existing basal segment, or we grow a new one.

        The only influence apical dendrites have on basal learning is: the apical
        dendrites influence which cells are considered "predicted". So an active
        apical dendrite can prevent some basal segments in active columns from
        learning.

        @param correctPredictedCells (numpy array)
        @param burstingColumns (numpy array)
        @param activeBasalSegments (numpy array)

        @return (tuple)
        - learningActiveBasalSegments (numpy array)
            Active basal segments on correct predicted cells

        - learningMatchingBasalSegments (numpy array)
            Matching basal segments selected for learning in bursting columns

        - basalSegmentsToPunish (numpy array)
            Basal segments that should be punished for predicting an inactive column

        - newBasalSegmentCells (numpy array)
            Cells in bursting columns that were selected to grow new basal segments

        - learningCells (numpy array)
            Cells that have learning basal segments or are selected to grow a basal
            segment
        """

        # Correctly predicted columns
        learningActiveBasalSegments = self.basalConnections.filterSegmentsByCell(self.activeBasalSegments, correctPredictedCells)

        cellsForMatchingBasal = self.basalConnections.mapSegmentsToCells(self.matchingBasalSegments)
        matchingCells = np.unique(cellsForMatchingBasal)

        matchingCellsInBurstingColumns, burstingColumnsWithNoMatch = np2.setCompare(matchingCells, burstingColumns, matchingCells // self.cellsPerColumn, rightMinusLeft=True)

        learningMatchingBasalSegments = self._chooseBestSegmentPerColumn(self.basalConnections, matchingCellsInBurstingColumns, self.matchingBasalSegments, self.basalPotentialOverlaps)
        newBasalSegmentCells = self._getCellsWithFewestSegments( self.basalConnections, burstingColumnsWithNoMatch)

        learningCells = np.concatenate(
            (correctPredictedCells,
             self.basalConnections.mapSegmentsToCells(learningMatchingBasalSegments),
             newBasalSegmentCells))

        # Incorrectly predicted columns
        correctMatchingBasalMask = np.isin(cellsForMatchingBasal // self.cellsPerColumn, activeColumns)

        basalSegmentsToPunish = self.matchingBasalSegments[~correctMatchingBasalMask]

        return (learningActiveBasalSegments,
                learningMatchingBasalSegments,
                basalSegmentsToPunish,
                newBasalSegmentCells,
                learningCells)


    def _calculateApicalLearning(self, learningCells, activeColumns):
        """
        Calculate apical learning for each learning cell.

        The set of learning cells was determined completely from basal segments.
        Do all apical learning on the same cells.

        Learn on any active segments on learning cells. For cells without active
        segments, learn on the best matching segment. For cells without a matching
        segment, grow a new segment.

        @param learningCells (numpy array)
        @param correctPredictedCells (numpy array)
        @param activeApicalSegments (numpy array)
        @param matchingApicalSegments (numpy array)
        @param apicalPotentialOverlaps (numpy array)

        @return (tuple)
        - learningActiveApicalSegments (numpy array)
            Active apical segments on correct predicted cells

        - learningMatchingApicalSegments (numpy array)
            Matching apical segments selected for learning in bursting columns

        - apicalSegmentsToPunish (numpy array)
            Apical segments that should be punished for predicting an inactive column

        - newApicalSegmentCells (numpy array)
            Cells in bursting columns that were selected to grow new apical segments
        """

        # Cells with active apical segments
        learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(self.activeApicalSegments, learningCells)

        # Cells with matching apical segments
        learningCellsWithoutActiveApical = np.setdiff1d(learningCells, self.apicalConnections.mapSegmentsToCells(learningActiveApicalSegments))
        cellsForMatchingApical = self.apicalConnections.mapSegmentsToCells(self.matchingApicalSegments)
        learningCellsWithMatchingApical = np.intersect1d(learningCellsWithoutActiveApical, cellsForMatchingApical)
        learningMatchingApicalSegments = self._chooseBestSegmentPerCell(self.apicalConnections, learningCellsWithMatchingApical, self.matchingApicalSegments, self.apicalPotentialOverlaps)

        # Cells that need to grow an apical segment
        newApicalSegmentCells = np.setdiff1d(learningCellsWithoutActiveApical, learningCellsWithMatchingApical)

        # Incorrectly predicted columns
        correctMatchingApicalMask = np.isin(cellsForMatchingApical // self.cellsPerColumn, activeColumns)

        apicalSegmentsToPunish = self.matchingApicalSegments[~correctMatchingApicalMask]

        return (learningActiveApicalSegments,
                learningMatchingApicalSegments,
                apicalSegmentsToPunish,
                newApicalSegmentCells)


    def _calculateApicalSegmentActivity(self, activeInput):
        """
        Calculate the active and matching apical segments for this timestep.

        @param activeInput SDR

        @return (tuple)
        - activeSegments (numpy array)
            Dendrite segments with enough active connected synapses to cause a
            dendritic spike

        - matchingSegments (numpy array)
            Dendrite segments with enough active potential synapses to be selected for
            learning in a bursting column

        - potentialOverlaps (numpy array)
            The number of active potential synapses for each segment.
            Includes counts for active, matching, and nonmatching segments.
        """

        # Active
        overlaps,potentialOverlaps = self.apicalConnections.computeActivityFull(activeInput, False)
        activeSegments = np.flatnonzero(overlaps >= self.activationThreshold)

        # Matching
        matchingSegments = np.flatnonzero(potentialOverlaps >= self.minThreshold)

        return activeSegments, matchingSegments, potentialOverlaps


    def _calculateBasalSegmentActivity(self, activeInput, reducedBasalThresholdCells):
        """
        Calculate the active and matching basal segments for this timestep.

        The difference with _calculateApicalSegmentActivity is that cells
        with active apical segments (collected in reducedBasalThresholdCells) have
        a lower activation threshold for their basal segments (set by
        reducedBasalThreshold parameter).

        @param activeInput SDR
        @param reducedBasalThresholdCells (numpy array)

        @return (tuple)
        - activeSegments (numpy array)
            Dendrite segments with enough active connected synapses to cause a
            dendritic spike

        - matchingSegments (numpy array)
            Dendrite segments with enough active potential synapses to be selected for
            learning in a bursting column

        - potentialOverlaps (numpy array)
            The number of active potential synapses for each segment.
            Includes counts for active, matching, and nonmatching segments.
        """
        # Active apical segments lower the activation threshold for basal (lateral) segments
        overlaps,potentialOverlaps = self.basalConnections.computeActivityFull(activeInput, False)
        outrightActiveSegments = np.flatnonzero(overlaps >= self.activationThreshold)
        
        if self.reducedBasalThreshold != self.activationThreshold and len(reducedBasalThresholdCells) > 0:
            potentiallyActiveSegments = np.flatnonzero((overlaps < self.activationThreshold) & (overlaps >= self.reducedBasalThreshold))
            cellsOfCASegments = self.basalConnections.mapSegmentsToCells(potentiallyActiveSegments)
            # apically active segments are condit. active segments from apically active cells
            conditionallyActiveSegments = potentiallyActiveSegments[np.isin(cellsOfCASegments, reducedBasalThresholdCells)]
            activeSegments = np.concatenate((outrightActiveSegments, conditionallyActiveSegments))
        else:
            activeSegments = outrightActiveSegments

        # Matching
        matchingSegments = np.flatnonzero(potentialOverlaps >= self.minThreshold)

        return activeSegments, matchingSegments, potentialOverlaps


    def _calculatePredictedCells(self, activeBasalSegments, activeApicalSegments):
        """
        Calculate the predicted cells, given the set of active segments.

        An active basal segment is enough to predict a cell.
        An active apical segment is *not* enough to predict a cell.

        When a cell has both types of segments active, other cells in its minicolumn
        must also have both types of segments to be considered predictive.

        @param activeBasalSegments (numpy array)
        @param activeApicalSegments (numpy array)

        @return (numpy array)
        """

        cellsForBasalSegments = self.basalConnections.mapSegmentsToCells(activeBasalSegments)
        if self.useApicalTiebreak == False:
            predictedCells = cellsForBasalSegments

        else:
            cellsForApicalSegments = self.apicalConnections.mapSegmentsToCells(activeApicalSegments)
    
            fullyDepolarizedCells = np.intersect1d(cellsForBasalSegments, cellsForApicalSegments)
            partlyDepolarizedCells = np.setdiff1d(cellsForBasalSegments, fullyDepolarizedCells)
    
            inhibitedMask = np.isin(partlyDepolarizedCells // self.cellsPerColumn, fullyDepolarizedCells // self.cellsPerColumn)
            predictedCells = np.append(fullyDepolarizedCells, partlyDepolarizedCells[~inhibitedMask])

        return predictedCells

    def _learn(self, connections, learningSegments, activeInput, growthCandidates, potentialOverlaps):
        """
        Adjust synapse permanences, grow new synapses, and grow new segments.

        @param learningActiveSegments (numpy array)
        @param learningMatchingSegments (numpy array)
        @param activeInput SDR
        @param growthCandidates (numpy array)
        @param potentialOverlaps (numpy array)
        """

        # Learn on existing segments
        for segment in learningSegments:
            connections.adaptSegment(segment, activeInput, self.permanenceIncrement, self.permanenceDecrement, False)

            # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
            # grow per segment. "maxNew" might be a number or it might be a list of
            # numbers.
            if self.sampleSize == -1:
                maxNew = len(growthCandidates)
            else:
                maxNew = self.sampleSize - potentialOverlaps[segment]
    
            if self.maxSynapsesPerSegment != -1:
                synapseCounts = connections.numSynapses(segment)
                numSynapsesToReachMax = self.maxSynapsesPerSegment - synapseCounts
                maxNew = min(maxNew, numSynapsesToReachMax)
            if maxNew > 0:
                connections.growSynapsesToSample(segment, growthCandidates, maxNew, self.initialPermanence, self.rng)


    def _learnOnNewSegments(self, connections, newSegmentCells, growthCandidates):

        numNewSynapses = len(growthCandidates)

        if self.sampleSize != -1:
            numNewSynapses = min(numNewSynapses, self.sampleSize)

        if self.maxSynapsesPerSegment != -1:
            numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)
            
        for cell in  newSegmentCells:
            newSegment = connections.createSegment(cell, self.maxSegmentsPerCell)
            connections.growSynapsesToSample(newSegment, growthCandidates, numNewSynapses, self.initialPermanence, self.rng)


    def _chooseBestSegmentPerCell(self,
                                connections,
                                cells,
                                allMatchingSegments,
                                potentialOverlaps):
        """
        For each specified cell, choose its matching segment with largest number
        of active potential synapses. When there's a tie, the first segment wins.

        @param connections (SparseMatrixConnections)
        @param cells (numpy array)
        @param allMatchingSegments (numpy array)
        @param potentialOverlaps (numpy array)

        @return (numpy array)
        One segment per cell
        """

        candidateSegments = connections.filterSegmentsByCell(allMatchingSegments, cells)

        # Narrow it down to one pair per cell.
        onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments], connections.mapSegmentsToCells(candidateSegments))
        learningSegments = candidateSegments[onePerCellFilter]

        return learningSegments


    def _chooseBestSegmentPerColumn(self, connections, matchingCells, allMatchingSegments, potentialOverlaps):
        """
        For all the columns covered by 'matchingCells', choose the column's matching
        segment with largest number of active potential synapses. When there's a
        tie, the first segment wins.

        @param connections (SparseMatrixConnections)
        @param matchingCells (numpy array)
        @param allMatchingSegments (numpy array)
        @param potentialOverlaps (numpy array)
        """

        candidateSegments = connections.filterSegmentsByCell(allMatchingSegments, matchingCells)

        # Narrow it down to one segment per column.
        cellScores = potentialOverlaps[candidateSegments]
        columnsForCandidates = (connections.mapSegmentsToCells(candidateSegments) // self.cellsPerColumn)
        onePerColumnFilter = np2.argmaxMulti(cellScores, columnsForCandidates)

        learningSegments = candidateSegments[onePerColumnFilter]

        return learningSegments


    def _getCellsWithFewestSegments(self, connections, columns):
        """
        For each column, get the cell that has the fewest total basal segments.
        Break ties randomly.

        @param connections (SparseMatrixConnections)
        @param rng (Random)
        @param columns (numpy array) Columns to check

        @return (numpy array)
        One cell for each of the provided columns
        """
        candidateCells = np2.getAllCellsInColumns(columns, self.cellsPerColumn)

        # Arrange the segment counts into one row per minicolumn.
        segmentCounts = np.reshape(connections.getSegmentCounts(candidateCells), newshape=(len(columns), self.cellsPerColumn))

        # Filter to just the cells that are tied for fewest in their minicolumn.
        minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
        candidateCells = candidateCells[np.flatnonzero(segmentCounts == minSegmentCounts)]

        # Filter to one cell per column, choosing randomly from the minimums.
        # To do the random choice, add a random offset to each index in-place, using
        # casting to floor the result.
        _, onePerColumnFilter, numCandidatesInColumns = np.unique(candidateCells // self.cellsPerColumn, return_index=True, return_counts=True)

        offsetPercents = np.empty(len(columns), dtype="float64")
        self.rng.initializeReal64Array(offsetPercents)

        np.add(onePerColumnFilter, offsetPercents*numCandidatesInColumns, out=onePerColumnFilter, casting="unsafe")

        return candidateCells[onePerColumnFilter]


    def getActiveCells(self):
        """
        @return (numpy array)
        Active cells
        """
        return self.activeCells


    def getPredictedActiveCells(self):
        """
        @return (numpy array)
        Active cells that were correctly predicted
        """
        return self.predictedActiveCells


    def getWinnerCells(self):
        """
        @return (numpy array)
        Cells that were selected for learning
        """
        return self.winnerCells


    def getActiveBasalSegments(self):
        """
        @return (numpy array)
        Active basal segments for this timestep
        """
        return self.activeBasalSegments


    def getActiveApicalSegments(self):
        """
        @return (numpy array)
        Matching basal segments for this timestep
        """
        return self.activeApicalSegments


    def numberOfColumns(self):
        """ Returns the number of columns in this layer.

        @return (int) Number of columns
        """
        return self.columnCount


    def numberOfCells(self):
        """
        Returns the number of cells in this layer.

        @return (int) Number of cells
        """
        return self.numberOfColumns() * self.cellsPerColumn


    def getCellsPerColumn(self):
        """
        Returns the number of cells per column.

        @return (int) The number of cells per column.
        """
        return self.cellsPerColumn


    def getActivationThreshold(self):
        """
        Returns the activation threshold.
        @return (int) The activation threshold.
        """
        return self.activationThreshold


    def setActivationThreshold(self, activationThreshold):
        """
        Sets the activation threshold.
        @param activationThreshold (int) activation threshold.
        """
        self.activationThreshold = activationThreshold


    def getReducedBasalThreshold(self):
        """
        Returns the reduced basal activation threshold for apically active cells.
        @return (int) The activation threshold.
        """
        return self.reducedBasalThreshold


    def setReducedBasalThreshold(self, reducedBasalThreshold):
        """
        Sets the reduced basal activation threshold for apically active cells.
        @param reducedBasalThreshold (int) activation threshold.
        """
        self.reducedBasalThreshold = reducedBasalThreshold


    def getInitialPermanence(self):
        """
        Get the initial permanence.
        @return (float) The initial permanence.
        """
        return self.initialPermanence


    def setInitialPermanence(self, initialPermanence):
        """
        Sets the initial permanence.
        @param initialPermanence (float) The initial permanence.
        """
        self.initialPermanence = initialPermanence


    def getMinThreshold(self):
        """
        Returns the min threshold.
        @return (int) The min threshold.
        """
        return self.minThreshold


    def setMinThreshold(self, minThreshold):
        """
        Sets the min threshold.
        @param minThreshold (int) min threshold.
        """
        self.minThreshold = minThreshold


    def getSampleSize(self):
        """
        Gets the sampleSize.
        @return (int)
        """
        return self.sampleSize


    def setSampleSize(self, sampleSize):
        """
        Sets the sampleSize.
        @param sampleSize (int)
        """
        self.sampleSize = sampleSize


    def getPermanenceIncrement(self):
        """
        Get the permanence increment.
        @return (float) The permanence increment.
        """
        return self.permanenceIncrement


    def setPermanenceIncrement(self, permanenceIncrement):
        """
        Sets the permanence increment.
        @param permanenceIncrement (float) The permanence increment.
        """
        self.permanenceIncrement = permanenceIncrement


    def getPermanenceDecrement(self):
        """
        Get the permanence decrement.
        @return (float) The permanence decrement.
        """
        return self.permanenceDecrement


    def setPermanenceDecrement(self, permanenceDecrement):
        """
        Sets the permanence decrement.
        @param permanenceDecrement (float) The permanence decrement.
        """
        self.permanenceDecrement = permanenceDecrement


    def getBasalPredictedSegmentDecrement(self):
        """
        Get the predicted segment decrement.
        @return (float) The predicted segment decrement.
        """
        return self.basalPredictedSegmentDecrement


    def setBasalPredictedSegmentDecrement(self, basalPredictedSegmentDecrement):
        """
        Sets the predicted segment decrement.
        @param predictedSegmentDecrement (float) The predicted segment decrement.
        """
        self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement


    def getApicalPredictedSegmentDecrement(self):
        """
        Get the predicted segment decrement.
        @return (float) The predicted segment decrement.
        """
        return self.apicalPredictedSegmentDecrement


    def setApicalPredictedSegmentDecrement(self, apicalPredictedSegmentDecrement):
        """
        Sets the predicted segment decrement.
        @param predictedSegmentDecrement (float) The predicted segment decrement.
        """
        self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement


    def getConnectedPermanence(self):
        """
        Get the connected permanence.
        @return (float) The connected permanence.
        """
        return self.connectedPermanence


    def setConnectedPermanence(self, connectedPermanence):
        """
        Sets the connected permanence.
        @param connectedPermanence (float) The connected permanence.
        """
        self.connectedPermanence = connectedPermanence


    def getUseApicalTieBreak(self):
        """
        Get whether we actually use apical tie-break.
        @return (Bool) Whether apical tie-break is used.
        """
        return self.useApicalTiebreak


    def setUseApicalTiebreak(self, useApicalTiebreak):
        """
        Sets whether we actually use apical tie-break.
        @param useApicalTiebreak (Bool) Whether apical tie-break is used.
        """
        self.useApicalTiebreak = useApicalTiebreak


    def getUseApicalModulationBasalThreshold(self):
        """
        Get whether we actually use apical modulation of basal threshold.
        @return (Bool) Whether apical modulation is used.
        """
        return self.useApicalModulationBasalThreshold


    def setUseApicalModulationBasalThreshold(self, useApicalModulationBasalThreshold):
        """
        Sets whether we actually use apical modulation of basal threshold.
        @param useApicalModulationBasalThreshold (Bool) Whether apical modulation is used.
        """
        self.useApicalModulationBasalThreshold = useApicalModulationBasalThreshold



class ApicalTiebreakPairMemory(ApicalTiebreakTemporalMemory):
    """
    Pair memory with apical tiebreak.
    """

    def compute(self,
                activeColumns,
                basalInput,
                apicalInput=(),
                basalGrowthCandidates=None,
                apicalGrowthCandidates=None,
                learn=True):
        """
        Perform one timestep. Use the basal and apical input to form a set of
        predictions, then activate the specified columns, then learn.

        @param activeColumns (numpy array)
        List of active columns

        @param basalInput (numpy array)
        List of active input bits for the basal dendrite segments

        @param apicalInput (numpy array)
        List of active input bits for the apical dendrite segments

        @param basalGrowthCandidates (numpy array or None)
        List of bits that the active cells may grow new basal synapses to.
        If None, the basalInput is assumed to be growth candidates.

        @param apicalGrowthCandidates (numpy array or None)
        List of bits that the active cells may grow new apical synapses to
        If None, the apicalInput is assumed to be growth candidates.

        @param learn (bool)
        Whether to grow / reinforce / punish synapses
        """
        activeColumns = np.asarray(activeColumns)

        apicalInputSDR = SDR(self.apicalInputSize)
        apicalInputSDR.sparse = apicalInput
        
        basalInputSDR = SDR(self.basalInputSize)
        basalInputSDR.sparse = basalInput

        if basalGrowthCandidates is None:
            basalGrowthCandidates = basalInput
        basalGrowthCandidates = np.asarray(basalGrowthCandidates)

        if apicalGrowthCandidates is None:
            apicalGrowthCandidates = apicalInput
        apicalGrowthCandidates = np.asarray(apicalGrowthCandidates)

        self.depolarizeCells(basalInputSDR, apicalInputSDR, learn)
        
        apicalInputSDR.sparse = apicalInput
        basalInputSDR.sparse = basalInput
        self.activateCells(activeColumns, basalInputSDR, apicalInputSDR, basalGrowthCandidates, apicalGrowthCandidates, learn)


    def getPredictedCells(self):
        """
        @return (numpy array)
        Cells that were predicted for this timestep
        """
        return self.predictedCells


    def getBasalPredictedCells(self):
        """
        @return (numpy array)
        Cells with active basal segments
        """
        return np.unique(self.basalConnections.mapSegmentsToCells(self.activeBasalSegments))


    def getApicalPredictedCells(self):
        """
        @return (numpy array)
        Cells with active apical segments
        """
        return np.unique(self.apicalConnections.mapSegmentsToCells(self.activeApicalSegments))




class ApicalTiebreakSequenceMemory(ApicalTiebreakTemporalMemory):
    """
    Sequence memory with apical tiebreak.
    """

    def __init__(self,
                 columnCount=2048,
                 apicalInputSize=0,
                 cellsPerColumn=32,
                 activationThreshold=13,
                 reducedBasalThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.1,
                 basalPredictedSegmentDecrement=0.0,
                 apicalPredictedSegmentDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 seed=42):
        params = {
            "columnCount": columnCount,
            "basalInputSize": columnCount * cellsPerColumn,
            "apicalInputSize": apicalInputSize,
            "cellsPerColumn": cellsPerColumn,
            "activationThreshold": activationThreshold,
            "reducedBasalThreshold": reducedBasalThreshold,
            "initialPermanence": initialPermanence,
            "connectedPermanence": connectedPermanence,
            "minThreshold": minThreshold,
            "sampleSize": sampleSize,
            "permanenceIncrement": permanenceIncrement,
            "permanenceDecrement": permanenceDecrement,
            "basalPredictedSegmentDecrement": basalPredictedSegmentDecrement,
            "apicalPredictedSegmentDecrement": apicalPredictedSegmentDecrement,
            "maxSynapsesPerSegment": maxSynapsesPerSegment,
            "seed": seed,
        }

        super(ApicalTiebreakSequenceMemory, self).__init__(**params)

        self.prevApicalInput = np.empty(0, dtype="uint32")
        self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")
        self.prevPredictedCells = np.empty(0, dtype="uint32")


    def reset(self):
        """
        Clear all cell and segment activity.
        """
        super(ApicalTiebreakSequenceMemory, self).reset()

        self.prevApicalInput = np.empty(0, dtype="uint32")
        self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")
        self.prevPredictedCells = np.empty(0, dtype="uint32")


    def compute(self,
                activeColumns,
                apicalInput=(),
                apicalGrowthCandidates=None,
                learn=True):
        """
        Perform one timestep. Activate the specified columns, using the predictions
        from the previous timestep, then learn. Then form a new set of predictions
        using the new active cells and the apicalInput.

        @param activeColumns (numpy array)
        List of active columns

        @param apicalInput (numpy array)
        List of active input bits for the apical dendrite segments

        @param apicalGrowthCandidates (numpy array or None)
        List of bits that the active cells may grow new apical synapses to
        If None, the apicalInput is assumed to be growth candidates.

        @param learn (bool)
        Whether to grow / reinforce / punish synapses
        """
        activeColumns = np.asarray(activeColumns)
        apicalInput = np.asarray(apicalInput)

        apicalInputSDR = SDR(self.apicalInputSize)

        basalInputSDR = SDR(self.basalInputSize)
        basalInputSDR.sparse = self.activeCells

        if apicalGrowthCandidates is None:
            apicalGrowthCandidates = apicalInput
        apicalGrowthCandidates = np.asarray(apicalGrowthCandidates)

        self.prevPredictedCells = self.predictedCells

        apicalInputSDR.sparse = self.prevApicalInput
        self.activateCells(activeColumns, basalInputSDR, apicalInputSDR, self.winnerCells, self.prevApicalGrowthCandidates, learn)
        
        apicalInputSDR.sparse = apicalInput
        basalInputSDR.sparse = self.activeCells
        self.depolarizeCells(basalInputSDR, apicalInputSDR, learn)

        self.prevApicalInput = apicalInput.copy()
        self.prevApicalGrowthCandidates = apicalGrowthCandidates.copy()


    def getPredictedCells(self):
        """
        @return (numpy array)
        The prediction from the previous timestep
        """
        return self.prevPredictedCells


    def getNextPredictedCells(self):
        """
        @return (numpy array)
        The prediction for the next timestep
        """
        return self.predictedCells


    def getNextBasalPredictedCells(self):
        """
        @return (numpy array)
        Cells with active basal segments
        """
        return np.unique(self.basalConnections.mapSegmentsToCells(self.activeBasalSegments))


    def getNextApicalPredictedCells(self):
        """
        @return (numpy array)
        Cells with active apical segments
        """
        return np.unique(self.apicalConnections.mapSegmentsToCells(self.activeApicalSegments))
