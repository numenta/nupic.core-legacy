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

import numpy as np

from htm.bindings.math import Random
from .connections import Connections
from htm.bindings.sdr import SDR



class ColumnPooler(object):
    """
    This class constitutes a temporary implementation for a cross-column pooler.
    The implementation goal of this class is to prove basic properties before
    creating a cleaner implementation.
    """

    def __init__(self,
         inputWidth,
         lateralInputWidths=(),
         cellCount=4096,
         sdrSize=40,
         onlineLearning = False,
         maxSdrSize = None,
         minSdrSize = None,

         # Proximal
         synPermProximalInc=0.1,
         synPermProximalDec=0.001,
         initialProximalPermanence=0.6,
         sampleSizeProximal=20,
         minThresholdProximal=10,
         connectedPermanenceProximal=0.50,
         predictedInhibitionThreshold=20,

         # Distal
         synPermDistalInc=0.1,
         synPermDistalDec=0.001,
         initialDistalPermanence=0.6,
         sampleSizeDistal=20,
         activationThresholdDistal=13,
         connectedPermanenceDistal=0.50,
         inertiaFactor=1.,

         seed=42):
        """
        Parameters:
        ----------------------------
        @param  inputWidth (int)
                The number of bits in the feedforward input
    
        @param  lateralInputWidths (list of ints)
                The number of bits in each lateral input
    
        @param  sdrSize (int)
                The number of active cells in an object SDR
    
        @param  onlineLearning (Bool)
                Whether or not the column pooler should learn in online mode.
    
        @param  maxSdrSize (int)
                The maximum SDR size for learning.  If the column pooler has more
                than this many cells active, it will refuse to learn.  This serves
                to stop the pooler from learning when it is uncertain of what object
                it is sensing.
    
        @param  minSdrSize (int)
                The minimum SDR size for learning.  If the column pooler has fewer
                than this many active cells, it will create a new representation
                and learn that instead.  This serves to create separate
                representations for different objects and sequences.
    
                If online learning is enabled, this parameter should be at least
                inertiaFactor*sdrSize.  Otherwise, two different objects may be
                incorrectly inferred to be the same, as SDRs may still be active
                enough to learn even after inertial decay.
    
        @param  synPermProximalInc (float)
                Permanence increment for proximal synapses
    
        @param  synPermProximalDec (float)
                Permanence decrement for proximal synapses
    
        @param  initialProximalPermanence (float)
                Initial permanence value for proximal synapses
    
        @param  sampleSizeProximal (int)
                Number of proximal synapses a cell should grow to each feedforward
                pattern, or -1 to connect to every active bit
    
        @param  minThresholdProximal (int)
                Number of active synapses required for a cell to have feedforward
                support
    
        @param  connectedPermanenceProximal (float)
                Permanence required for a proximal synapse to be connected
    
        @param  predictedInhibitionThreshold (int)
                How much predicted input must be present for inhibitory behavior
                to be triggered.  Only has effects if onlineLearning is true.
    
        @param  synPermDistalInc (float)
                Permanence increment for distal synapses
    
        @param  synPermDistalDec (float)
                Permanence decrement for distal synapses
    
        @param  sampleSizeDistal (int)
                Number of distal synapses a cell should grow to each lateral
                pattern, or -1 to connect to every active bit
    
        @param  initialDistalPermanence (float)
                Initial permanence value for distal synapses
    
        @param  activationThresholdDistal (int)
                Number of active synapses required to activate a distal segment
    
        @param  connectedPermanenceDistal (float)
                Permanence required for a distal synapse to be connected
    
        @param  inertiaFactor (float)
                The proportion of previously active cells that remain
                active in the next timestep due to inertia (in the absence of
                inhibition).  If onlineLearning is enabled, should be at most
                1 - learningTolerance, or representations may incorrectly become
                mixed.
    
        @param  seed (int)
                Random number generator seed
        """

        assert maxSdrSize is None or maxSdrSize >= sdrSize
        assert minSdrSize is None or minSdrSize <= sdrSize

        self.inputWidth = inputWidth
        self.cellCount = cellCount
        self.sdrSize = sdrSize
        self.onlineLearning = onlineLearning
        if maxSdrSize is None:
            self.maxSdrSize = sdrSize
        else:
            self.maxSdrSize = maxSdrSize
        if minSdrSize is None:
            self.minSdrSize = sdrSize
        else:
            self.minSdrSize = minSdrSize
        self.synPermProximalInc = synPermProximalInc
        self.synPermProximalDec = synPermProximalDec
        self.initialProximalPermanence = initialProximalPermanence
        self.connectedPermanenceProximal = connectedPermanenceProximal
        self.sampleSizeProximal = sampleSizeProximal
        self.minThresholdProximal = minThresholdProximal
        self.predictedInhibitionThreshold = predictedInhibitionThreshold
        self.synPermDistalInc = synPermDistalInc
        self.synPermDistalDec = synPermDistalDec
        self.initialDistalPermanence = initialDistalPermanence
        self.connectedPermanenceDistal = connectedPermanenceDistal
        self.sampleSizeDistal = sampleSizeDistal
        self.activationThresholdDistal = activationThresholdDistal
        self.inertiaFactor = inertiaFactor
        self.lateralInputWidths = lateralInputWidths

        self.activeCells = np.empty(0, dtype="uint32")
        self._random = Random(seed)

        # Each cell potentially has
        # 1 proximal segment and 1+len(lateralInputWidths) distal segments.
        self.proximalPermanences = Connections(cellCount, connectedPermanenceProximal, False) #inputWidth max synapses
        self.internalDistalPermanences = Connections(cellCount, connectedPermanenceDistal, False) #cellCount max synapses
        self.distalPermanences = [Connections(cellCount, connectedPermanenceDistal, False) for _ in lateralInputWidths] #lateralInputWidths max synapses

        self.useInertia=True

    def compute(self, 
                feedforwardInput=(), 
                lateralInputs=(),
                feedforwardGrowthCandidates=None, 
                learn=True,
                predictedInput = None):
        """
        Runs one time step of the column pooler algorithm.
    
        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits
    
        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits
    
        @param  feedforwardGrowthCandidates (sequence or None)
                Sorted indices of feedforward input bits that active cells may grow
                new synapses to. If None, the entire feedforwardInput is used.
    
        @param  learn (bool)
                If True, we are learning a new object
    
        @param predictedInput (sequence)
               Sorted indices of predicted cells in the TM layer.
        """

        if feedforwardGrowthCandidates is None:
            feedforwardGrowthCandidates = feedforwardInput

        # inference step
        if not learn:
            self._computeInferenceMode(feedforwardInput, lateralInputs)

        # learning step
        elif not self.onlineLearning:
            self._computeLearningMode(feedforwardInput, lateralInputs, feedforwardGrowthCandidates)
        # online learning step
        else:
            if (predictedInput is not None and len(predictedInput) > self.predictedInhibitionThreshold):
                predictedActiveInput = np.intersect1d(feedforwardInput, predictedInput)
#                 predictedGrowthCandidates = np.intersect1d(feedforwardGrowthCandidates, predictedInput)
                self._computeInferenceMode(predictedActiveInput, lateralInputs)
                self._computeLearningMode(predictedActiveInput, lateralInputs, feedforwardGrowthCandidates)
            elif not self.minSdrSize <= len(self.activeCells) <= self.maxSdrSize:
                # If the pooler doesn't have a single representation, try to infer one,
                # before actually attempting to learn.
                self._computeInferenceMode(feedforwardInput, lateralInputs)
                self._computeLearningMode(feedforwardInput, lateralInputs, feedforwardGrowthCandidates)
            else:
                # If there isn't predicted input and we have a single SDR,
                # we are extending that representation and should just learn.
                self._computeLearningMode(feedforwardInput, lateralInputs, feedforwardGrowthCandidates)


    def _computeLearningMode(self, feedforwardInput, lateralInputs, feedforwardGrowthCandidates):
        """
        Learning mode: we are learning a new object in an online fashion. If there
        is no prior activity, we randomly activate 'sdrSize' cells and create
        connections to incoming input. If there was prior activity, we maintain it.
        If we have a union, we simply do not learn at all.
    
        These cells will represent the object and learn distal connections to each
        other and to lateral cortical columns.
    
        Parameters:
        ----------------------------
        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits
    
        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits
    
        @param  feedforwardGrowthCandidates (sequence or None)
                Sorted indices of feedforward input bits that the active cells may
                grow new synapses to.  This is assumed to be the predicted active
                cells of the input layer.
        """
        prevActiveCells = self.activeCells

        # If there are not enough previously active cells, then we are no longer on
        # a familiar object.    Either our representation decayed due to the passage
        # of time (i.e. we moved somewhere else) or we were mistaken.    Either way,
        # create a new SDR and learn on it.
        # This case is the only way different object representations are created.
        # enforce the active cells in the output layer
        if len(self.activeCells) < self.minSdrSize:
            self.activeCells = self._sampleRange(0, self.numberOfCells(), step=1, k=self.sdrSize)
            self.activeCells.sort()

        # If we have a union of cells active, don't learn.    This primarily affects
        # online learning.
        if len(self.activeCells) > self.maxSdrSize:
            return

        # Finally, now that we have decided which cells we should be learning on, do
        # the actual learning.
        if len(feedforwardInput) > 0:
            feedforwardInputSDR = SDR(self.inputWidth)
            feedforwardInputSDR.sparse = feedforwardInput
            self._learn(self.proximalPermanences,
                        feedforwardInputSDR,
                        feedforwardGrowthCandidates, self.sampleSizeProximal,
                        self.initialProximalPermanence, self.synPermProximalInc,
                        self.synPermProximalDec, self.connectedPermanenceProximal)

            # External distal learning
            for i, lateralInput in enumerate(lateralInputs):
                if len(lateralInput) > 0:
                    lateralInputSDR = SDR(self.lateralInputWidths[i])
                    lateralInputSDR.sparse = lateralInput
                    self._learn(self.distalPermanences[i],
                                lateralInputSDR, lateralInput,
                                self.sampleSizeDistal, self.initialDistalPermanence,
                                self.synPermDistalInc, self.synPermDistalDec,
                                self.connectedPermanenceDistal)

            # Internal distal learning
            if len(prevActiveCells):
                prevActiveCellsSDR = SDR(self.cellCount)
                prevActiveCellsSDR.sparse = prevActiveCells
                self._learn(self.internalDistalPermanences,
                            prevActiveCellsSDR, prevActiveCells,
                            self.sampleSizeDistal, self.initialDistalPermanence,
                            self.synPermDistalInc, self.synPermDistalDec,
                            self.connectedPermanenceDistal)

    def _computeInferenceMode(self, feedforwardInput, lateralInputs):
        """
        Inference mode: if there is some feedforward activity, perform
        spatial pooling on it to recognize previously known objects, then use
        lateral activity to activate a subset of the cells with feedforward
        support. If there is no feedforward activity, use lateral activity to
        activate a subset of the previous active cells.
    
        Parameters:
        ----------------------------
        @param  feedforwardInput (sequence)
                Sorted indices of active feedforward input bits
    
        @param  lateralInputs (list of sequences)
                For each lateral layer, a list of sorted indices of active lateral
                input bits
        """
        prevActiveCells = self.activeCells
        
        # Calculate the feedforward supported cells
        feedforwardInputSDR = SDR(self.inputWidth)
        feedforwardInputSDR.sparse = feedforwardInput
        activeSegments = self.proximalPermanences.computeActiveSegments(feedforwardInputSDR, self.minThresholdProximal)
        feedforwardSupportedCells = self.proximalPermanences.mapSegmentsToCells(activeSegments)
 
        # Calculate the number of active segments on each cell
        numActiveSegmentsByCell = np.zeros(self.cellCount, dtype="int")
        
        prevActiveCellsSDR = SDR(self.cellCount)
        prevActiveCellsSDR.sparse = prevActiveCells
        activeSegments = self.internalDistalPermanences.computeActiveSegments(prevActiveCellsSDR, self.activationThresholdDistal)
        for cell in self.proximalPermanences.mapSegmentsToCells(activeSegments):
            numActiveSegmentsByCell[cell] += 1
        
        for i, lateralInput in enumerate(lateralInputs):
            lateralInputSDR = SDR(self.lateralInputWidths[i])
            lateralInputSDR.sparse = lateralInput
            activeSegments = self.distalPermanences[i].computeActiveSegments(lateralInputSDR, self.activationThresholdDistal)
            for cell in self.proximalPermanences.mapSegmentsToCells(activeSegments):
                numActiveSegmentsByCell[cell] += 1

        chosenCells = []

        # First, activate the FF-supported cells that have the highest number of
        # lateral active segments (as long as it's not 0)
        if len(feedforwardSupportedCells) == 0:
            pass
        else:
            numActiveSegsForFFSuppCells = numActiveSegmentsByCell[feedforwardSupportedCells]

            # This loop will select the FF-supported AND laterally-active cells, in
            # order of descending lateral activation, until we exceed the sdrSize
            # quorum - but will exclude cells with 0 lateral active segments.
            ttop = np.max(numActiveSegsForFFSuppCells)
            while ttop > 0 and len(chosenCells) < self.sdrSize:
                supportedCells = [feedforwardSupportedCells[i] for i in range(len(feedforwardSupportedCells)) if numActiveSegsForFFSuppCells[i] >= ttop]
                chosenCells = np.union1d(chosenCells, supportedCells)
                ttop -= 1

        # If we haven't filled the sdrSize quorum, add in inertial cells.
        if len(chosenCells) < self.sdrSize:
            if self.useInertia:
                prevCells = np.setdiff1d(prevActiveCells, chosenCells)
                inertialCap = int(len(prevCells) * self.inertiaFactor)
                if inertialCap > 0:
                    numActiveSegsForPrevCells = numActiveSegmentsByCell[prevCells]
                    # We sort the previously-active cells by number of active lateral
                    # segments (this really helps).    We then activate them in order of
                    # descending lateral activation.
                    sortIndices = np.argsort(numActiveSegsForPrevCells)[::-1]
                    prevCells = prevCells[sortIndices]
                    numActiveSegsForPrevCells = numActiveSegsForPrevCells[sortIndices]

                    # We use inertiaFactor to limit the number of previously-active cells
                    # which can become active, forcing decay even if we are below quota.
                    prevCells = prevCells[:inertialCap]
                    numActiveSegsForPrevCells = numActiveSegsForPrevCells[:inertialCap]

                    # Activate groups of previously active cells by order of their lateral
                    # support until we either meet quota or run out of cells.
                    ttop = np.max(numActiveSegsForPrevCells)
                    while ttop >= 0 and len(chosenCells) < self.sdrSize:
                        chosenCells = np.union1d(chosenCells, prevCells[numActiveSegsForPrevCells >= ttop])
                        ttop -= 1

        # If we haven't filled the sdrSize quorum, add cells that have feedforward
        # support and no lateral support.
        discrepancy = self.sdrSize - len(chosenCells)
        if discrepancy > 0:
            remainingFFcells = np.setdiff1d(feedforwardSupportedCells, chosenCells)

            # Inhibit cells proportionally to the number of cells that have already
            # been chosen. If ~0 have been chosen activate ~all of the feedforward
            # supported cells. If ~sdrSize have been chosen, activate very few of
            # the feedforward supported cells.

            # Use the discrepancy:sdrSize ratio to determine the number of cells to
            # activate.
            n = (len(remainingFFcells) * discrepancy) // self.sdrSize
            # Activate at least 'discrepancy' cells.
            n = max(n, discrepancy)
            # If there aren't 'n' available, activate all of the available cells.
            n = min(n, len(remainingFFcells))

            if len(remainingFFcells) > n:
                selected = self._random.sample(remainingFFcells, n)
                chosenCells = np.append(chosenCells, selected)
            else:
                chosenCells = np.append(chosenCells, remainingFFcells)

        chosenCells.sort()
        self.activeCells = np.asarray(chosenCells, dtype="uint32")


    def numberOfInputs(self):
        """
        Returns the number of inputs into this layer
        """
        return self.inputWidth


    def numberOfCells(self):
        """
        Returns the number of cells in this layer.
        @return (int) Number of cells
        """
        return self.cellCount


    def getActiveCells(self):
        """
        Returns the indices of the active cells.
        @return (list) Indices of active cells.
        """
        return self.activeCells


    def numberOfConnectedProximalSynapses(self, cells=None):
        """
        Returns the number of proximal connected synapses on these cells.
    
        Parameters:
        ----------------------------
        @param  cells (iterable)
                Indices of the cells. If None return count for all cells.
        """
        if cells is None:
            cells = list(range(self.numberOfCells()))

        return self.proximalPermanences.numConnectedSynapsesForCells(cells)


    def numberOfProximalSynapses(self, cells=None):
        """
        Returns the number of proximal synapses with permanence>0 on these cells.
    
        Parameters:
        ----------------------------
        @param  cells (iterable)
                Indices of the cells. If None return count for all cells.
        """
        if cells is None:
            return self.proximalPermanences.numSynapses()

        return self.proximalPermanences.numSynapsesForCells(cells)


    def numberOfDistalSegments(self, cells=None):
        """
        Returns the total number of distal segments for these cells.
    
        A segment "exists" if its row in the matrix has any permanence values > 0.
    
        Parameters:
        ----------------------------
        @param  cells (iterable)
                Indices of the cells
        """
        if cells is None:
            cells = list(range(self.numberOfCells()))

        n = self.internalDistalPermanences.numSegmentsWithSynapses(cells)

        for permanences in self.distalPermanences:
            n += permanences. numSegmentsWithSynapses(cells)

        return n


    def numberOfConnectedDistalSynapses(self, cells=None):
        """
        Returns the number of connected distal synapses on these cells.
    
        Parameters:
        ----------------------------
        @param  cells (iterable)
                Indices of the cells. If None return count for all cells.
        """
        if cells is None:
            cells = list(range(self.numberOfCells()))

        n = self.internalDistalPermanences.numConnectedSynapsesForCells(cells)

        for permanences in self.distalPermanences:
            n += permanences.numConnectedSynapsesForCells(cells)

        return n


    def numberOfDistalSynapses(self, cells=None):
        """
        Returns the total number of distal synapses for these cells.
    
        Parameters:
        ----------------------------
        @param  cells (iterable)
                Indices of the cells
        """
        if cells is None:
            cells = list(range(self.numberOfCells()))
            
        n = self.internalDistalPermanences.numSynapsesForCells(cells)

        for permanences in self.distalPermanences:
            n += permanences.numSynapsesForCells(cells)

        return n


    def reset(self):
        """
        Reset internal states. When learning this signifies we are to learn a
        unique new object.
        """
        self.activeCells = np.empty(0, dtype="uint32")

    def getUseInertia(self):
        """
        Get whether we actually use inertia    (i.e. a fraction of the
        previously active cells remain active at the next time step unless
        inhibited by cells with both feedforward and lateral support).
        @return (Bool) Whether inertia is used.
        """
        return self.useInertia

    def setUseInertia(self, useInertia):
        """
        Sets whether we actually use inertia (i.e. a fraction of the
        previously active cells remain active at the next time step unless
        inhibited by cells with both feedforward and lateral support).
        @param useInertia (Bool) Whether inertia is used.
        """
        self.useInertia = useInertia
        
    def _learn(self,
             permanences,

             # activity
             activeInput, growthCandidateInput,

             # configuration
             sampleSize, initialPermanence, permanenceIncrement, permanenceDecrement, connectedPermanence):
        """
        For each active cell, reinforce active synapses, punish inactive synapses,
        and grow new synapses to a subset of the active input bits that the cell
        isn't already connected to.
    
        Parameters:
        ----------------------------
        @param  permanences (Connections)
                Matrix of permanences, with cells as rows and inputs as columns
    
        @param  activeInput (SDR)
                Active bits in the input
    
        @param  growthCandidateInput (sorted sequence)
                Sorted list of active bits in the input that the activeCells may
                grow new synapses to
    
        For remaining parameters, see the __init__ docstring.
        """

        active_input_array = activeInput.sparse
        growthCandidateInput = np.uint32(growthCandidateInput)
        
        for cell in self.activeCells:
            segments = permanences.segmentsForCell(cell)
            if not segments:
                segment = permanences.createSegment(cell, 1)
            else:
                segment = segments[0] # Should only have one segment per cell
                
            permanences.adaptSegment(segment, activeInput, permanenceIncrement, permanenceDecrement, False)
            presynamptic_cells = np.array([permanences.presynapticCellForSynapse(synapse) for synapse in permanences.synapsesForSegment(segment)])
            
            if sampleSize == -1:
                active_cells_without_synapses = np.setdiff1d(growthCandidateInput, presynamptic_cells, assume_unique=True)

            else:      
                active_cells_without_synapses = []
                
                existingSynapseCounts = len(np.intersect1d(presynamptic_cells, active_input_array, assume_unique=True))
                effective_sample_size = sampleSize - existingSynapseCounts
                
                if effective_sample_size > 0:
                    active_cells_without_synapses = np.setdiff1d(growthCandidateInput, presynamptic_cells, assume_unique=True)
                    if effective_sample_size < len(active_cells_without_synapses):
                        active_cells_without_synapses = self._random.sample(active_cells_without_synapses, effective_sample_size)
                    
            for c in active_cells_without_synapses:
                permanences.createSynapse(segment, c, initialPermanence)
 
    #
    # Functionality that could be added to the C code or bindings
    #    
    def _sampleRange(self, start, end, step, k):
        """
        Equivalent to:
    
        random.sample(xrange(start, end, step), k)
    
        except it uses our random number generator.
        """
        return np.array(self._random.sample(np.arange(start, end, step, dtype="uint32"), k), dtype="uint32")
        
