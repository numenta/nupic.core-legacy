# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.    Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
from htm.bindings.regions.PyRegion import PyRegion

from htm.advanced.algorithms.location_modules import ThresholdedGaussian2DLocationModule



class GridCellLocationRegion(PyRegion):
    """
    The GridCellLocationRegion computes the location of the sensor in the space
    of the object given sensory and motor inputs using gaussian grid cell modules
    to update the location.

    Sensory input drives the activity of the region while Motor input causes the
    region to perform path integration, updating its activity.

    The grid cell module algorithm used by this region is based on the
    :class:`ThresholdedGaussian2DLocationModule` where each module has one or more
    gaussian activity bumps that move as the population receives motor input. When
    two bumps are near each other, the intermediate cells have higher firing rates
    than they would with a single bump. The cells with firing rates above a
    certain threshold are considered "active". When the network receives a motor
    command, it shifts its bumps.

    The cells are distributed uniformly through the rhombus, packed in the optimal
    hexagonal arrangement. During learning, the cell nearest to the current phase
    is associated with the sensed feature.

    See :class:`ThresholdedGaussian2DLocationModule` for more details
    """

    @classmethod
    def getSpec(cls):
        """
        Return the spec for the GridCellLocationRegion
        """
        spec = dict(
            description=GridCellLocationRegion.__doc__,
            singleNodeOnly=True,
            inputs=dict(
                anchorInput=dict(
                    description="An array of 0's and 1's representing the sensory input "
                                "during inference. This will often come from a "
                                "feature-location pair layer (L4 active cells).",
                    dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False,
                ),
                anchorGrowthCandidates=dict(
                    description="An array of 0's and 1's representing the sensory input "
                                "during learning. This will often come from a "
                                "feature-location pair layer (L4 winner cells).",
                dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False,
                ),
                displacement=dict(
                    description="An array of floats representing the displacement as a "
                                "multi dimensional translation vector [d1, d2, ..., dk].",
                    dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False,
                ),
                resetIn=dict(
                    description="Clear all cell activity.",
                    dataType="Real32",
                    count=1,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False,
                )
            ),
            outputs=dict(
                activeCells=dict(
                    description="A binary output containing a 1 for every"
                                " cell that is currently active.",
                    dataType="Real32",
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False
                ),
                learnableCells=dict(
                    description="A binary output containing a 1 for every"
                                " cell that is currently learnable",
                    dataType="Real32",
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False
                ),
                sensoryAssociatedCells=dict(
                    description="A binary output containing a 1 for every"
                                " cell that is currently associated with a sensory input",
                    dataType="Real32",
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False
                )
            ),
            parameters=dict(
                moduleCount=dict(
                    description="Number of grid cell modules",
                    dataType="UInt32",
                    accessMode="Create",
                    count=1
                ),
                cellsPerAxis=dict(
                    description="Determines the number of cells. Determines how space is "
                                "divided between the cells",
                    dataType="UInt32",
                    accessMode="Create",
                    count=1
                ),
                scale=dict(
                    description="Determines the amount of world space covered by all of "
                                "the cells combined. In grid cell terminology, this is "
                                "equivalent to the 'scale' of a module. One scale value "
                                "for each grid cell module. Array size must match "
                                "'moduleCount' parameter",
                    dataType="Real32",
                    accessMode="Create",
                    count=0,
                ),
                orientation=dict(
                    description="The rotation of this map, measured in radians. One "
                                "orientation value for each grid cell module. Array size "
                                "must match 'moduleCount' parameter",
                    dataType="Real32",
                    accessMode="Create",
                    count=0,
                ),
                anchorInputSize=dict(
                    description="The number of input bits in the anchor input",
                    dataType="UInt32",
                    accessMode="Create",
                    count=1,
                ),
                activeFiringRate=dict(
                    description="Between 0.0 and 1.0. A cell is considered active if its "
                                "firing rate is at least this value",
                    dataType="Real32",
                    accessMode="Create",
                    count=1,
                ),
                bumpSigma=dict(
                    description="Specifies the diameter of a gaussian bump, in units of "
                                "'rhombus edges'. A single edge of the rhombus has length "
                                "1, and this bumpSigma would typically be less than 1. We "
                                "often use 0.18172 as an estimate for the sigma of a rat "
                                "entorhinal bump",
                    dataType="Real32",
                    accessMode="Create",
                    count=1,
                    defaultValue="0.18172",
                ),
                activationThreshold=dict(
                    description="If the number of active connected synapses on a "
                                "segment is at least this threshold, the segment "
                                "is said to be active",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    constraints="",
                    defaultValue="10"
                ),
                initialPermanence=dict(
                    description="Initial permanence of a new synapse",
                    accessMode="Create",
                    dataType="Real32",
                    count=1,
                    constraints="",
                    defaultValue="0.21"
                ),
                connectedPermanence=dict(
                    description="If the permanence value for a synapse is greater "
                                "than this value, it is said to be connected",
                    accessMode="Create",
                    dataType="Real32",
                    count=1,
                    constraints="",
                    defaultValue="0.50"
                ),
                learningThreshold=dict(
                    description="Minimum overlap required for a segment to learned",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    defaultValue="10"
                ),
                sampleSize=dict(
                    description="The desired number of active synapses for an "
                                "active cell",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    defaultValue="20"
                ),
                permanenceIncrement=dict(
                    description="Amount by which permanences of synapses are "
                                "incremented during learning",
                    accessMode="Create",
                    dataType="Real32",
                    count=1,
                    defaultValue="0.1"
                ),
                permanenceDecrement=dict(
                    description="Amount by which permanences of synapses are "
                                "decremented during learning",
                    accessMode="Create",
                    dataType="Real32",
                    count=1,
                    defaultValue="0.0"
                ),
                maxSynapsesPerSegment=dict(
                    description="The maximum number of synapses per segment",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    defaultValue="-1"
                ),
                bumpOverlapMethod=dict(
                    description="Specifies the firing rate of a cell when it's part of "
                                "two bumps. ('probabilistic' or 'sum')",
                    dataType="Byte",
                    accessMode="Create",
                    constraints=("enum: probabilistic, sum"),
                    defaultValue="probabilistic",
                    count=0,
                ),
                learningMode=dict(
                    description="A boolean flag that indicates whether or not we should "
                                "learn by associating the location with the sensory "
                                "input",
                    dataType="Bool",
                    accessMode="ReadWrite",
                    count=1,
                    defaultValue="False"
                ),
                dualPhase=dict(
                    description="A boolean flag that indicates whether or not we should "
                                "process movement and sensation using two phases of the "
                                "same network. When this flag is enabled, the compute "
                                "method will alternate between movement and sensation on "
                                "each phase. When this flag is disabled both movement and "
                                "sensations will be processed on a single phase.",
                    dataType="Bool",
                    accessMode="ReadWrite",
                    count=1,
                    defaultValue="True"
                ),
                dimensions=dict(
                    description="The number of dimensions represented in the displacement",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    defaultValue="2"
                ),
                seed=dict(
                    description="Seed for the random number generator",
                    accessMode="Create",
                    dataType="UInt32",
                    count=1,
                    defaultValue="42"
                )
            ),
            commands=dict(
                reset=dict(description="Clear all cell activity"),
                activateRandomLocation=dict(description="Set the location to a random point"),
            )
        )
        return spec

    def __init__(self,
                 moduleCount,
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
                 learningMode=False,
                 seed=42,
                 dualPhase=True,
                 dimensions=2,
                 **kwargs):
        if moduleCount <= 0 or cellsPerAxis <= 0:
            raise TypeError("Parameters moduleCount and cellsPerAxis must be > 0")
        if moduleCount != len(scale) or moduleCount != len(orientation):
            raise TypeError("scale and orientation arrays len must match moduleCount")
        if dimensions < 2:
            raise TypeError("dimensions must be >= 2")

        self.moduleCount = moduleCount
        self.cellsPerAxis = cellsPerAxis
        self.cellCount = cellsPerAxis * cellsPerAxis
        self.scale = list(scale)
        self.orientation = list(orientation)
        self.anchorInputSize = anchorInputSize
        self.activeFiringRate = activeFiringRate
        self.bumpSigma = bumpSigma
        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.learningThreshold = learningThreshold
        self.sampleSize = sampleSize
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.bumpOverlapMethod = bumpOverlapMethod
        self.learningMode = learningMode
        self.dualPhase = dualPhase
        self.dimensions = dimensions
        self.seed = seed

        # This flag controls whether the region is processing sensation or movement
        # on dual phase configuration
        self._sensing = False

        self._modules = None

        self._projection = None

        PyRegion.__init__(self, **kwargs)

    def initialize(self):
        """ Initialize grid cell modules """

        if self._modules is None:
            self._modules = []
            for i in range(self.moduleCount):
                self._modules.append(ThresholdedGaussian2DLocationModule(
                    cellsPerAxis=self.cellsPerAxis,
                    scale=self.scale[i],
                    orientation=self.orientation[i],
                    anchorInputSize=self.anchorInputSize,
                    activeFiringRate=self.activeFiringRate,
                    bumpSigma=self.bumpSigma,
                    activationThreshold=self.activationThreshold,
                    initialPermanence=self.initialPermanence,
                    connectedPermanence=self.connectedPermanence,
                    learningThreshold=self.learningThreshold,
                    sampleSize=self.sampleSize,
                    permanenceIncrement=self.permanenceIncrement,
                    permanenceDecrement=self.permanenceDecrement,
                    maxSynapsesPerSegment=self.maxSynapsesPerSegment,
                    maxSegmentsPerCell=self.maxSegmentsPerCell,
                    bumpOverlapMethod=self.bumpOverlapMethod,
                    seed=self.seed))

            # Create a projection matrix for each module used to convert higher
            # dimension displacements to 2D
            if self.dimensions > 2:
                self._projection = [
                    self.createProjectionMatrix(dimensions=self.dimensions)
                        for _ in range(self.moduleCount)]


    def compute(self, inputs, outputs):
        """
        Compute the location based on the 'displacement' and 'anchorInput' by first
        applying the    movement, if 'displacement' is present in the 'input' array
        and then applying the sensation if 'anchorInput' is present in the input
        array. The 'anchorGrowthCandidates' input array is used during learning

        See :meth:`ThresholdedGaussian2DLocationModule.movementCompute` and
                :meth:`ThresholdedGaussian2DLocationModule.sensoryCompute`
        """

        if inputs.get("resetIn", False):
            self.reset()
            if self.learningMode:
                # Initialize to random location after reset when learning
                self.activateRandomLocation()

            # send empty output
            outputs["activeCells"][:] = 0
            outputs["learnableCells"][:] = 0
            outputs["sensoryAssociatedCells"][:] = 0
            return

        displacement = inputs.get("displacement", np.array([]))
        anchorInput = inputs.get("anchorInput", np.array([])).nonzero()[0]
        anchorGrowthCandidates = inputs.get("anchorGrowthCandidates", np.array([])).nonzero()[0]

        # Concatenate the output of all modules
        activeCells = np.array([], dtype=np.uint32)
        learnableCells = np.array([], dtype=np.uint32)
        sensoryAssociatedCells = np.array([], dtype=np.uint32)

        # Only process input when data is available
        shouldMove = displacement.any()
        shouldSense = anchorInput.any() or anchorGrowthCandidates.any()

        if shouldMove and len(displacement) != self.dimensions:
            raise TypeError("displacement must have {} dimensions".format(self.dimensions))

        # Handles dual phase movement/sensation processing
        if self.dualPhase:
            if self._sensing:
                shouldMove = False
            else:
                shouldSense = False

            # Toggle between movement and sensation
            self._sensing = not self._sensing

        for i in range(self.moduleCount):
            module = self._modules[i]

            # Compute movement
            if shouldMove:
                movement = displacement
                if self.dimensions > 2:
                    # Project n-dimension displacements to 2D
                    movement = np.matmul(self._projection[i], movement)

                module.movementCompute(movement)

            # Compute sensation
            if shouldSense:
                module.sensoryCompute(anchorInput, anchorGrowthCandidates, self.learningMode)

            # Concatenate outputs
            start = i * self.cellCount
            activeCells = np.append(activeCells, module.getActiveCells() + start)
            learnableCells = np.append(learnableCells, module.getLearnableCells() + start)
            sensoryAssociatedCells = np.append(sensoryAssociatedCells, module.getSensoryAssociatedCells() + start)

        outputs["activeCells"][:] = 0
        outputs["activeCells"][activeCells] = 1
        outputs["learnableCells"][:] = 0
        outputs["learnableCells"][learnableCells] = 1
        outputs["sensoryAssociatedCells"][:] = 0
        outputs["sensoryAssociatedCells"][sensoryAssociatedCells] = 1

    def reset(self):
        """ Clear the active cells """
        for module in self._modules:
            module.reset()

    def activateRandomLocation(self):
        """
        Set the location to a random point.
        """
        for module in self._modules:
            module.activateRandomLocation()

    def setParameter(self, parameterName, index, parameterValue):
        """
        Set the value of a Spec parameter.
        """
        spec = self.getSpec()
        if parameterName not in spec['parameters']:
            raise Exception("Unknown parameter: " + parameterName)

        setattr(self, parameterName, parameterValue)

    def getOutputElementCount(self, name):
        """
        Returns the size of the output array
        """
        if name in ["activeCells", "learnableCells", "sensoryAssociatedCells"]:
            return (self.cellCount * self.moduleCount)
        else:
            raise Exception("Invalid output name specified: " + name)

    def getModules(self):
        """
        Returns underlying list of modules used by this region
        """
        return self._modules

    def createProjectionMatrix(self, dimensions=3):
        """
        Compute    projection matrix used to convert n-dimensional displacement into
        2D displacement compatible with :class:`ThresholdedGaussian2DLocationModule`
        algorithm. To compute the 2D displacement each module must multiply the
        n-dimensional displacement with the projection matrix for that module.

        :param dimensions: Number of dimensions. Must be greater than 2. Default 3.
        :type int dimensions:

        :return: The projection matrix
        :rtype: array(2, n)
        """
        if dimensions < 3:
            raise ValueError("dimensions value must be 3 or higher")

        b1 = np.random.multivariate_normal(mean=np.zeros(dimensions), cov=np.eye(dimensions))
        b1 /= np.linalg.norm(b1)

        # Choose a random vector orthogonal to b1
        while True:
            randomVector = np.random.multivariate_normal(mean=np.zeros(dimensions), cov=np.eye(dimensions))
            randomVector /= np.linalg.norm(randomVector)
            projectedToPlane = randomVector - np.dot(randomVector, b1) * b1

            # make sure random vector is not parallel to b1 plane
            length = np.linalg.norm(projectedToPlane)
            if length == 0:
                continue

            b2 = projectedToPlane / length

            # b1 and b2 are two orthogonal vectors on the plane.
            # To get a 2D displacement, you'll dot the n-dimensional displacement with
            # each of these vectors.
            return np.array([b1, b2])
