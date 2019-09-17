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

"""Emulates a grid cell module"""

from abc import ABC, abstractmethod

import math
import copy

import numpy as np

from htm.advanced.support import numpy_helpers as np2
from htm.bindings.math import Random
from .connections import Connections
from htm.bindings.sdr import SDR

class AbstractLocationModule(ABC):
    """
    A model of a location module. This is an Abstract superclass for location modules.

    Usage:
    - When the sensor moves, call movementCompute.
    - When the sensor senses something, call sensoryCompute.

    """

    def __init__(self,
                 cellsPerAxis,
                 scale,
                 orientation,
                 anchorInputSize,
                 activationThreshold=10,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 learningThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 maxSegmentsPerCell=255,
                 seed=42):

        self.cellsPerAxis = cellsPerAxis
        self.anchorInputSize = anchorInputSize
        self.scale = scale
        self.orientation = orientation

        self.activeCells = np.empty(0, dtype="int")

        # The cells that were activated by sensory input in an inference timestep,
        # or cells that were associated with sensory input in a learning timestep.
        self.sensoryAssociatedCells = np.empty(0, dtype="int")

        self.activeSegments = np.empty(0, dtype="uint32")

        self.connections = None

        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.learningThreshold = learningThreshold
        self.sampleSize = sampleSize
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.activationThreshold = activationThreshold
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell

        self.rng = Random(seed)

    def reset(self):
        """
        Clear the active cells.
        """
        self.cellsForActivePhases = np.empty(0, dtype="int")
        self.activeCells = np.empty(0, dtype="int")
        self.sensoryAssociatedCells = np.empty(0, dtype="int")

    @abstractmethod
    def _computeActiveCells(self):
        pass

    @abstractmethod
    def activateRandomLocation(self):
        """
        Set the location to a random point.
        """
        pass
        
    @abstractmethod
    def _movementComputeDelta(self, displacement):
        """
         Calculate delta in the module's coordinates and
         shift the active coordinates.
         Return phaseDisplacement
        """
        return None

    def movementCompute(self, displacement, noiseFactor = 0):
        """
        Shift the current active cells by a vector.
        This is called when the sensor moves.

        @param displacement (pair of floats)
        A translation vector [di, dj].
        """

        if noiseFactor != 0:
            displacement = copy.deepcopy(displacement)
            xnoise = np.random.normal(0, noiseFactor)
            ynoise = np.random.normal(0, noiseFactor)
            displacement[0] += xnoise
            displacement[1] += ynoise

        phaseDisplacement = self._movementComputeDelta(displacement)

        # In Python, (x % 1.0) can return 1.0 because of floating point goofiness.
        # Generally this doesn't cause problems, it's just confusing when you're
        # debugging.
        np.round(self.bumpPhases, decimals=9, out=self.bumpPhases)
        np.mod(self.bumpPhases, 1.0, out=self.bumpPhases)

        self._computeActiveCells()
        self.phaseDisplacement = phaseDisplacement
        
    @abstractmethod
    def _sensoryComputeInferenceMode(self, anchorInput):
        """
        Infer the location from sensory input. Activate any cells with enough active
        synapses to this sensory input. Deactivate all other cells.

        @param anchorInput (numpy array)
        A sensory input. This will often come from a feature-location pair layer.
        """
        if len(anchorInput.sparse) == 0:
            return

    def _sensoryComputeLearningMode(self, anchorInput):
        """
        Associate this location with a sensory input. Subsequently, anchorInput will
        activate the current location during anchor().

        @param anchorInput SDR
        A sensory input. This will often come from a feature-location pair layer.
        """
        overlaps,potentialOverlaps = self.connections.computeActivityFull(anchorInput, False)
        activeSegments = np.flatnonzero(overlaps >= self.activationThreshold)
        matchingSegments = np.flatnonzero(potentialOverlaps >= self.learningThreshold)

        # Cells with a active segment: reinforce the segment
        cellsForActiveSegments = self.connections.mapSegmentsToCells(activeSegments)
        learningActiveSegments = activeSegments[np.in1d(cellsForActiveSegments, self.getLearnableCells())]
        remainingCells = np.setdiff1d(self.getLearnableCells(), cellsForActiveSegments)

        # Remaining cells with a matching segment: reinforce the best
        # matching segment.
        candidateSegments = self.connections.filterSegmentsByCell(matchingSegments, remainingCells)
        cellsForCandidateSegments = (self.connections.mapSegmentsToCells(candidateSegments))
        candidateSegments = candidateSegments[np.in1d(cellsForCandidateSegments, remainingCells)]
        onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments], cellsForCandidateSegments)
        learningMatchingSegments = candidateSegments[onePerCellFilter]

        newSegmentCells = np.setdiff1d(remainingCells, cellsForCandidateSegments)

        for learningSegments in (learningActiveSegments, learningMatchingSegments):
            self._learn(learningSegments, anchorInput, potentialOverlaps)

        # Remaining cells without a matching segment: grow one.
        self._learnOnNewSegments(newSegmentCells, anchorInput)

        self.activeSegments = activeSegments
        self.sensoryAssociatedCells = self.getLearnableCells()

    def sensoryCompute(self, anchorInput, anchorGrowthCandidates, learn):
        """
        This is called when the sensor senses something
        """
        anchorInputSDR = SDR(self.anchorInputSize)
        
        if learn:
            anchorInputSDR.sparse = anchorGrowthCandidates
            self._sensoryComputeLearningMode(anchorInputSDR)
            
        else:
            anchorInputSDR.sparse = anchorInput
            self._sensoryComputeInferenceMode(anchorInputSDR)

    def _learn(self, learningSegments, activeInput, potentialOverlaps):
        """
        Adjust synapse permanences, grow new synapses, and grow new segments.

        @param learningActiveSegments (numpy array)
        @param learningMatchingSegments (numpy array)
        @param segmentsToPunish (numpy array)
        @param activeInput SDR
        @param potentialOverlaps (numpy array)
        """
        for segment in learningSegments:
            # Learn on existing segments
            self.connections.adaptSegment(segment, activeInput, self.permanenceIncrement, self.permanenceDecrement, False)
    
            # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
            # grow per segment. "maxNew" might be a number or it might be a list of
            # numbers.
            if self.sampleSize == -1:
                maxNew = len(activeInput.sparse)
            else:
                maxNew = self.sampleSize - potentialOverlaps[segment]
    
            if self.maxSynapsesPerSegment != -1:
                synapseCounts = self.connections.numSynapses(segment)
                numSynapsesToReachMax = self.maxSynapsesPerSegment - synapseCounts
                maxNew = np.where(maxNew <= numSynapsesToReachMax, maxNew, numSynapsesToReachMax)
                
            if maxNew > 0:
                self.connections.growSynapsesToSample(segment, activeInput.sparse, maxNew, self.initialPermanence, self.rng)

    def _learnOnNewSegments(self, newSegmentCells, growthCandidates):

        numNewSynapses = len(growthCandidates.sparse)

        if self.sampleSize != -1:
            numNewSynapses = min(numNewSynapses, self.sampleSize)

        if self.maxSynapsesPerSegment != -1:
            numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)
            
        for cell in  newSegmentCells:
            newSegment = self.connections.createSegment(cell, self.maxSegmentsPerCell)
            self.connections.growSynapsesToSample(newSegment, growthCandidates.sparse, numNewSynapses, self.initialPermanence, self.rng)

    def getActiveCells(self):
        return self.activeCells

    @abstractmethod
    def getLearnableCells(self):
        return None

    def getSensoryAssociatedCells(self):
        return self.sensoryAssociatedCells

    @abstractmethod
    def numberOfCells(self):
        return 0


class ThresholdedGaussian2DLocationModule(AbstractLocationModule):
    """
    A model of a grid cell module. The module has one or more Gaussian activity
    bumps that move as the population receives motor input. When two bumps are
    near each other, the intermediate cells have higher firing rates than they
    would with a single bump. The cells with firing rates above a certain
    threshold are considered "active".

    We don't model the neural dynamics of path integration. When the network
    receives a motor command, it shifts its bumps. We do this by tracking each
    bump as floating point coordinates, and we shift the bumps with movement. This
    model isn't attempting to explain how path integration works. It's attempting
    to show how a population of cells that can path integrate are useful in a
    larger network.

    The cells are distributed uniformly through the rhombus, packed in the optimal
    hexagonal arrangement. During learning, the cell nearest to the current phase
    is associated with the sensed feature.

    This class doesn't choose working parameters for you. You need to give it a
    coherent mix of cellsPerAxis, activeFiringRate, and bumpSigma that:
     1. Ensure at least one cell fires at each location
     2. Use a large enough set of active cells that inference accounts for
            uncertainty in the learned locations.
    Use chooseReliableActiveFiringRate() to get good parameters.

    Usage:
    - When the sensor moves, call movementCompute.
    - When the sensor senses something, call sensoryCompute.
    """

    def __init__(self,
                 cellsPerAxis,
                 scale,
                 orientation,
                 anchorInputSize,
                 activeFiringRate,
                 bumpSigma,
                 activationThreshold=10,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 learningThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 maxSegmentsPerCell=255,
                 bumpOverlapMethod="probabilistic",
                 seed=42):
        """
        Uses hexagonal firing fields.

        @param cellsPerAxis (int)
        Determines the number of cells. Determines how space is divided between the
        cells.

        @param scale (float)
        Determines the amount of world space covered by all of the cells combined.
        In grid cell terminology, this is equivalent to the "scale" of a module.

        @param orientation (float)
        The rotation of this map, measured in radians.

        @param anchorInputSize (int)
        The number of input bits in the anchor input.

        @param activeFiringRate (float)
        Between 0.0 and 1.0. A cell is considered active if its firing rate is at
        least this value.

        @param bumpSigma (float)
        Specifies the diameter of a gaussian bump, in units of "rhombus edges". A
        single edge of the rhombus has length 1, and this bumpSigma would typically
        be less than 1. We often use 0.18172 as an estimate for the sigma of a rat
        entorhinal bump.

        @param bumpOverlapMethod ("probabilistic" or "sum")
        Specifies the firing rate of a cell when it's part of two bumps.
        """

        super(ThresholdedGaussian2DLocationModule, self). __init__(
                                                                    cellsPerAxis,
                                                                    scale,
                                                                    orientation,
                                                                    anchorInputSize,
                                                                    activationThreshold,
                                                                    initialPermanence,
                                                                    connectedPermanence,
                                                                    learningThreshold,
                                                                    sampleSize,
                                                                    permanenceIncrement,
                                                                    permanenceDecrement,
                                                                    maxSynapsesPerSegment,
                                                                    maxSegmentsPerCell,
                                                                    seed)

        self.cellsPerAxis = cellsPerAxis

        # Matrix that converts a world displacement into a phase displacement.
        self.A = np.linalg.inv(scale * np.array(
            [[math.cos(orientation), math.cos(orientation + np.radians(60.))],
             [math.sin(orientation), math.sin(orientation + np.radians(60.))]]))

        # Phase is measured as a number in the range [0.0, 1.0)
        self.bumpPhases = np.empty((2,0), dtype="float")
        self.cellsForActivePhases = np.empty(0, dtype="int")
        self.phaseDisplacement = np.empty((0,2), dtype="float")

        # Analogous to "winner cells" in other parts of code.
        self.learningCells = np.empty(0, dtype="int")

        self.connections = Connections(self.cellsPerAxis * self.cellsPerAxis, connectedPermanence, False)

        self.bumpSigma = bumpSigma
        self.activeFiringRate = activeFiringRate
        self.bumpOverlapMethod = bumpOverlapMethod

        cellPhasesAxis = np.linspace(0., 1., self.cellsPerAxis, endpoint=False)
        self.cellPhases = np.array([np.repeat(cellPhasesAxis, self.cellsPerAxis), np.tile(cellPhasesAxis, self.cellsPerAxis)])
        # Shift the cells so that they're more intuitively arranged within the
        # rhombus, rather than along the edges of the rhombus. This has no
        # meaningful impact, but it makes visualizations easier to understand.
        self.cellPhases += [[0.5/self.cellsPerAxis], [0.5/self.cellsPerAxis]]

    def reset(self):
        """
        Clear the active cells.
        """
        super(ThresholdedGaussian2DLocationModule, self).reset()
        
        self.bumpPhases = np.empty((2,0), dtype="float")
        self.phaseDisplacement = np.empty((0,2), dtype="float")
        self.learningCells = np.empty(0, dtype="int")
    
    def _computeActiveCells(self):
        cellExcitations = ThresholdedGaussian2DLocationModule.getCellExcitations(self.cellPhases, self.bumpPhases, self.bumpSigma, self.bumpOverlapMethod)

        self.activeCells = np.where(cellExcitations >= self.activeFiringRate)[0]
        self.learningCells = np.where(cellExcitations == cellExcitations.max())[0]

    def activateRandomLocation(self):
        """
        Set the location to a random point.
        """
        self.bumpPhases = np.array([np.random.random(2)]).T
        self._computeActiveCells()

    def _movementComputeDelta(self, displacement):
        """
         Calculate delta in the module's coordinates and
         shift the active coordinates.
         Return phaseDisplacement
        """
        # Calculate delta in the module's coordinates.
        phaseDisplacement = np.matmul(self.A, displacement)

        # Shift the active coordinates.
        np.add(self.bumpPhases, phaseDisplacement[:,np.newaxis], out=self.bumpPhases)

        return phaseDisplacement 
        
    def _sensoryComputeInferenceMode(self, anchorInput):
        """
        Infer the location from sensory input. Activate any cells with enough active
        synapses to this sensory input. Deactivate all other cells.

        @param anchorInput SDR
        A sensory input. This will often come from a feature-location pair layer.
        """
        if len(anchorInput.sparse) == 0:
            return

        activeSegments = self.connections.computeActiveSegments(anchorInput, self.activationThreshold)

        sensorySupportedCells = np.unique(self.connections.mapSegmentsToCells(activeSegments))

        self.bumpPhases = self.cellPhases[:,sensorySupportedCells]
        self._computeActiveCells()
        self.activeSegments = activeSegments
        self.sensoryAssociatedCells = sensorySupportedCells

    def getLearnableCells(self):
        return self.learningCells

    def numberOfCells(self):
        return self.cellsPerAxis * self.cellsPerAxis


    @staticmethod
    def chooseReliableActiveFiringRate(cellsPerAxis, bumpSigma, minimumActiveDiameter=None):
        """
        When a cell is activated by sensory input, this implies that the phase is
        within a particular small patch of the rhombus. This patch is roughly
        equivalent to a circle of diameter (1/cellsPerAxis)(2/sqrt(3)), centered on
        the cell. This 2/sqrt(3) accounts for the fact that when circles are packed
        into hexagons, there are small uncovered spaces between the circles, so the
        circles need to expand by a factor of (2/sqrt(3)) to cover this space.

        This sensory input will activate the phase at the center of this cell. To
        account for uncertainty of the actual phase that was used during learning,
        the bump of active cells needs to be sufficiently large for this cell to
        remain active until the bump has moved by the above diameter. So the
        diameter of the bump (and, equivalently, the cell's firing field) needs to
        be at least 2 of the above diameters.

        @param minimumActiveDiameter (float or None)
        If specified, this makes sure the bump of active cells is always above a
        certain size. This is useful for testing scenarios where grid cell modules
        can only encode location with a limited "readout resolution", matching the
        biology.

        @return
        An "activeFiringRate" for use in the ThresholdedGaussian2DLocationModule.
        """
        firingFieldDiameter = 2 * (1./cellsPerAxis)*(2./math.sqrt(3))

        if minimumActiveDiameter:
            firingFieldDiameter = max(firingFieldDiameter, minimumActiveDiameter)

        return ThresholdedGaussian2DLocationModule.gaussian(bumpSigma, firingFieldDiameter / 2.)


    @staticmethod
    def gaussian(sig, d):
        return np.exp(-np.power(d, 2.) / (2 * np.power(sig, 2.)))


    @staticmethod
    def getCellExcitations(cellPhases, bumpPhases, bumpSigma, bumpOverlapMethod):
        # For each cell, compute the phase displacement from each bump. Create an
        # array of matrices, one per cell. Each column in a matrix corresponds to
        # the phase displacement from the bump to the cell.
        cell_bump_positivePhaseDisplacement = np.mod( cellPhases.T[:, :, np.newaxis] - bumpPhases, 1.0)

        # For each cell/bump pair, consider the phase displacement vectors reaching
        # that cell from that bump by moving up-and-right, down-and-right,
        # down-and-left, and up-and-left. Create a 2D array of matrices, arranged by
        # cell then direction. Each column in a matrix corresponds to a phase
        # displacement from the bump to the cell in a particular direction.
        cell_direction_bump_phaseDisplacement = (
            cell_bump_positivePhaseDisplacement[:, np.newaxis, :, :] -
                np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])[:,:,np.newaxis])

        # Convert the displacement in phase to a displacement in the world, with
        # scale normalized out. Unless the grid is a square grid, it's important to
        # measure distances using world displacements, not the phase displacements,
        # because two vectors with the same phase distance will typically have
        # different world distances unless they are parallel. (Consider the fact
        # that the two diagonals of a rhombus have different world lengths but the
        # same phase lengths.)
        B = np.array([[np.cos(np.radians(0.)), np.cos(np.radians(60.))], [np.sin(np.radians(0.)), np.sin(np.radians(60.))]])
        cell_direction_bump_worldDisplacement = np.matmul( B, cell_direction_bump_phaseDisplacement)

        # Measure the length of each displacement vector. Create a 3D array of
        # distances, organized by cell, direction, then bump.
        cell_direction_bump_distance = np.linalg.norm( cell_direction_bump_worldDisplacement, axis=-2)

        # Choose the shortest distance from each cell to each bump. Create a 2D
        # array of distances, organized by cell then bump.
        cell_bump_distance = np.amin(cell_direction_bump_distance, axis=1)

        # Compute the gaussian of each of these distances.
        cellExcitationsFromBumps = ThresholdedGaussian2DLocationModule.gaussian( bumpSigma, cell_bump_distance)

        # Combine bumps. Create an array of firing rates, organized by cell.
        if bumpOverlapMethod == "probabilistic":
            # Think of a bump as a probability distribution, with each cell's firing
            # rate encoding its relative probability that it's the correct
            # location. For bump A,
            #     P(A = cell x)
            # is encoded by cell x's firing rate.
            #
            # In this interpretation, a union of bumps A, B, ... is not a probability
            # distribution, rather it is a set of independent events. When multiple
            # bumps overlap, the cell's firing rate should encode its relative
            # probability that it's correct in *any* bump. So it encodes:
            #     P((A = cell x) or (B = cell x) or ...)
            # and so this is equivalent to:
            #     1 - P((A != cell x) and (B != cell x) and ...)
            # We treat the events as independent, so this is equal to:
            #     1 - P(A != cell x) * P(B != cell x) * ...
            #
            # With this approach, as more and more bumps overlap, a cell's firing rate
            # increases, but not as quickly as it would with a sum.
            cellExcitations = 1. - np.prod(1. - cellExcitationsFromBumps, axis=1)
        elif bumpOverlapMethod == "sum":
            # Sum the firing rates. The probabilistic interpretation: this treats a
            # union as a set of equally probable events of which only one is true.
            cellExcitations = np.sum(cellExcitationsFromBumps, axis=1)
        else:
            raise ValueError("Unrecognized bump overlap strategy", bumpOverlapMethod)

        return cellExcitations



class Superficial2DLocationModule(AbstractLocationModule):
    """
    A model of a location module. It's similar to a grid cell module, but it uses
    squares rather than triangles.

    The cells are arranged into a m*n rectangle which is tiled onto 2D space.
    Each cell represents a small rectangle in each tile.

  +------+------+------++------+------+------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #1  |  #2  |  #3  ||  #1  |  #2  |  #3  |
  |      |      |      ||      |      |      |
  +--------------------++--------------------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #4  |  #5  |  #6  ||  #4  |  #5  |  #6  |
  |      |      |      ||      |      |      |
  +--------------------++--------------------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #7  |  #8  |  #9  ||  #7  |  #8  |  #9  |
  |      |      |      ||      |      |      |
  +------+------+------++------+------+------+

    We assume that path integration works *somehow*. This model receives a "delta
    location" vector, and it shifts the active cells accordingly. The model stores
    intermediate coordinates of active cells. Whenever sensory cues activate a
    cell, the model adds this cell to the list of coordinates being shifted.
    Whenever sensory cues cause a cell to become inactive, that cell is removed
    from the list of coordinates.

    (This model doesn't attempt to propose how "path integration" works. It
    attempts to show how locations are anchored to sensory cues.)

    When orientation is set to 0 degrees, the displacement is a [di, dj],
    moving di cells "down" and dj cells "right".

    When orientation is set to 90 degrees, the displacement is essentially a
    [dx, dy], applied in typical x,y coordinates with the origin on the bottom
    left.

    Usage:
    - When the sensor moves, call movementCompute.
    - When the sensor senses something, call sensoryCompute.

    The "anchor input" is typically a feature-location pair SDR.

    To specify how points are tracked, pass anchoringMethod = "corners",
    "narrowing" or "discrete".    "discrete" will cause the network to operate in a
    fully discrete space, where uncertainty is impossible as long as movements are
    integers.    "narrowing" is designed to narrow down uncertainty of initial
    locations of sensory stimuli.    "corners" is designed for noise-tolerance, and
    will activate all cells that are possible outcomes of path integration.
    """

    def __init__(self,
                 cellsPerAxis,
                 scale,
                 orientation,
                 anchorInputSize,
                 activationThreshold=10,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 learningThreshold=10,
                 sampleSize=20,
                 permanenceIncrement=0.1,
                 permanenceDecrement=0.0,
                 maxSynapsesPerSegment=-1,
                 maxSegmentsPerCell=255,
                 cellCoordinateOffsets=(0.5,),
                 anchoringMethod="narrowing",
                 rotationMatrix = None,
                 seed=42):
        """
        @param cellsPerAxis (int)
        Determines the number of cells. Determines how space is divided between the
        cells.

        @param scale (float)
        Determines the amount of world space covered by all of the cells combined.
        In grid cell terminology, this is equivalent to the "scale" of a module.

        @param orientation (float)
        The rotation of this map, measured in radians.

        @param anchorInputSize (int)
        The number of input bits in the anchor input.

        @param cellCoordinateOffsets (list of floats)
        These must each be between 0.0 and 1.0. Every time a cell is activated by
        anchor input, this class adds a "phase" which is shifted in subsequent
        motions. By default, this phase is placed at the center of the cell. This
        parameter allows you to control where the point is placed and whether multiple
        are placed. For example, with value [0.2, 0.8], when cell [2, 3] is activated
        it will place 4 phases, corresponding to the following points in cell
        coordinates: [2.2, 3.2], [2.2, 3.8], [2.8, 3.2], [2.8, 3.8]
        """
        
        super(Superficial2DLocationModule, self). __init__(
                                                            cellsPerAxis,
                                                            scale,
                                                            orientation,
                                                            anchorInputSize,
                                                            activationThreshold,
                                                            initialPermanence,
                                                            connectedPermanence,
                                                            learningThreshold,
                                                            sampleSize,
                                                            permanenceIncrement,
                                                            permanenceDecrement,
                                                            maxSynapsesPerSegment,
                                                            maxSegmentsPerCell,
                                                            seed)

        self.cellDimensions = np.asarray([cellsPerAxis, cellsPerAxis], dtype="int")
        self.moduleMapDimensions = np.asarray([scale, scale], dtype="float")
        self.phasesPerUnitDistance = 1.0 / self.moduleMapDimensions

        if rotationMatrix is None:
            self.orientation = orientation
            self.rotationMatrix = np.array([[math.cos(orientation), -math.sin(orientation)], [math.sin(orientation), math.cos(orientation)]])
            if anchoringMethod == "discrete":
                # Need to convert matrix to have integer values
                nonzeros = self.rotationMatrix[np.where(np.abs(self.rotationMatrix)>0)]
                smallestValue = np.amin(nonzeros)
                self.rotationMatrix /= smallestValue
                self.rotationMatrix = np.ceil(self.rotationMatrix)
        else:
            self.rotationMatrix = rotationMatrix

        self.cellCoordinateOffsets = cellCoordinateOffsets

        # Phase is measured as a number in the range [0.0, 1.0)
        self.activePhases = np.empty((0,2), dtype="float")
        self.cellsForActivePhases = np.empty(0, dtype="int")
        self.phaseDisplacement = np.empty((0,2), dtype="float")

        self.connections = Connections(np.prod(self.cellDimensions), connectedPermanence, False)

        self.anchoringMethod = anchoringMethod

    def reset(self):
        """
        Clear the active cells.
        """
        super(Superficial2DLocationModule, self).reset()
        self.activePhases = np.empty((0,2), dtype="float")
        self.phaseDisplacement = np.empty((0,2), dtype="float")
    
    def _computeActiveCells(self):
        # Round each coordinate to the nearest cell.
        activeCellCoordinates = np.floor(self.activePhases * self.cellDimensions).astype("int")

        # Convert coordinates to cell numbers.
        self.cellsForActivePhases = (np.ravel_multi_index(activeCellCoordinates.T, self.cellDimensions))
        self.activeCells = np.unique(self.cellsForActivePhases)

    def activateRandomLocation(self):
        """
        Set the location to a random point.
        """
        self.activePhases = np.array([np.random.random(2)])
        if self.anchoringMethod == "discrete":
            # Need to place the phase in the middle of a cell
            self.activePhases = np.floor(
                self.activePhases * self.cellDimensions)/self.cellDimensions
        self._computeActiveCells()

    def _movementComputeDelta(self, displacement):
        """
         Calculate delta in the module's coordinates and
         shift the active coordinates.
         Return phaseDisplacement
        """
        # Calculate delta in the module's coordinates.
        phaseDisplacement = (np.matmul(self.rotationMatrix, displacement) * self.phasesPerUnitDistance)

        # Shift the active coordinates.
        np.add(self.activePhases, phaseDisplacement, out=self.activePhases)
        
        return phaseDisplacement

    def _sensoryComputeInferenceMode(self, anchorInput):
        """
        Infer the location from sensory input. Activate any cells with enough active
        synapses to this sensory input. Deactivate all other cells.

        @param anchorInput SDR
        A sensory input. This will often come from a feature-location pair layer.
        """
        if len(anchorInput.sparse) == 0:
            return

        activeSegments = self.connections.computeActiveSegments(anchorInput, self.activationThreshold)

        sensorySupportedCells = np.unique(self.connections.mapSegmentsToCells(activeSegments))

        inactivated = np.setdiff1d(self.activeCells, sensorySupportedCells)
        inactivatedIndices = np.in1d(self.cellsForActivePhases, inactivated).nonzero()[0]
        if inactivatedIndices.size > 0:
            self.activePhases = np.delete(self.activePhases, inactivatedIndices, axis=0)

        activated = np.setdiff1d(sensorySupportedCells, self.activeCells)

        # Find centers of point clouds
        if "corners" in self.anchoringMethod:
            activatedCoordsBase = np.transpose(np.unravel_index(sensorySupportedCells, self.cellDimensions)).astype('float')
        else:
            activatedCoordsBase = np.transpose(np.unravel_index(activated, self.cellDimensions)).astype('float')

        # Generate points to add
        activatedCoords = np.concatenate(
            [activatedCoordsBase + [iOffset, jOffset]
             for iOffset in self.cellCoordinateOffsets
             for jOffset in self.cellCoordinateOffsets]
        )
        if "corners" in self.anchoringMethod:
            self.activePhases = activatedCoords // self.cellDimensions

        else:
            if activatedCoords.size > 0:
                self.activePhases = np.append(self.activePhases,activatedCoords // self.cellDimensions, axis=0)

        self._computeActiveCells()
        self.activeSegments = activeSegments
        self.sensoryAssociatedCells = sensorySupportedCells

    def getLearnableCells(self):
        return self.activeCells

    def numberOfCells(self):
        return np.prod(self.cellDimensions)



