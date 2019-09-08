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
The methods here contain factories to create networks of multiple layers for
experimenting with grid cell location layer (L6a)
"""
import copy
import json

import numpy as np
from htm.bindings.engine_internal import Network

from htm.frameworks.location.path_integration_union_narrowing import computeRatModuleParametersFromReadoutResolution
from htm.frameworks.location.path_integration_union_narrowing import computeRatModuleParametersFromCellCount
from htm.support.logging_decorator import LoggingDecorator
from htm.regions.RawSensor import RawSensor as Sensors
from htm.regions.RawValues import RawValues as Motors



def createL4L6aLocationColumn(network, L4Params, L6aParams, inverseReadoutResolution=None, baselineCellsPerAxis=6, suffix=""):
    """
    Create a single column network containing L4 and L6a layers. L4 layer
    processes sensor inputs while L6a processes motor commands using grid cell
    modules. Sensory input is represented by the feature's active columns and
    motor input is represented by the displacement vector [dx, dy].

    The grid cell modules used by this network are based on
    :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
    from on one or more Gaussian activity bumps. The cells are distributed
    uniformly through the rhombus, packed in the optimal hexagonal arrangement.
    ::

    Phase
    -----                    +-------+
                 +---------->|       |<------------+
     [2]         |     +---->|  L4   |--winner---+ |
                 |     |     |       |           | |
                 |     |     +-------+           | |
                 |     |       |   ^             | |
                 |     |       |   |             | |
                 |     |       |   |             | |
                 |     |       v   |             | |
                 |     |     +-------+           | |
                 |     |     |       |           | |
     [1,3]       |     +---->|  L6a  |<----------+ |
                 |     |     |       |--learnable--+
                 |     |     +-------+
                 |     |         ^
            feature  reset       |
                 |     |         |
                 |     |         |
     [0]      [sensorInput] [motorInput]


    .. note::
        Region names are "motorInput", "sensorInput", "L4", and "L6a".
        Each name has an optional string suffix appended to it.

    :param network: network to add the column
    :type network: Network
    :param L4Params:    constructor parameters for :class:`ApicalTMPairRegion`
    :type L4Params: dict
    :param L6aParams:    constructor parameters for :class:`GridCellLocationRegion`
    :type L6aParams: dict
    :param inverseReadoutResolution: Optional readout resolution.
        The readout resolution specifies the diameter of the circle of phases in the
        rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
    :type inverseReadoutResolution: int
    :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
        resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
        that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
        the readout resolution is approximately 1/4
    :type baselineCellsPerAxis: int or float
    :param suffix: optional string suffix appended to region name. Useful when
                                 creating multicolumn networks.
    :type suffix: str

    :return: Reference to the given network
    :rtype: Network
    """
    L6aParams = copy.deepcopy(L6aParams)
    if inverseReadoutResolution is not None:
        # Configure L6a based on 'resolution'
        params = computeRatModuleParametersFromReadoutResolution(inverseReadoutResolution)
        L6aParams.update(params)
    else:
        params = computeRatModuleParametersFromCellCount(L6aParams["cellsPerAxis"], baselineCellsPerAxis)
        L6aParams.update(params)

    numOfcols = L4Params["columnCount"]
    cellsPerCol = L4Params["cellsPerColumn"]
    L6aParams["anchorInputSize"] = numOfcols * cellsPerCol

    # Configure L4 'basalInputSize' to be compatible L6a output
    moduleCount = L6aParams["moduleCount"]
    cellsPerAxis = L6aParams["cellsPerAxis"]

    L4Params = copy.deepcopy(L4Params)
    L4Params["basalInputWidth"] = moduleCount * cellsPerAxis * cellsPerAxis

    # Configure sensor output to be compatible with L4 params
    columnCount = L4Params["columnCount"]

    # Add regions to network
    motorInputName = "motorInput" + suffix
    sensorInputName = "sensorInput" + suffix
    L4Name = "L4" + suffix
    L6aName = "L6a" + suffix
    dimensions = L6aParams.get("dimensions", 2)

    network.addRegion(sensorInputName, "py.RawSensor", json.dumps({"outputWidth": columnCount}))
    network.addRegion(motorInputName, "py.RawValues", json.dumps({"outputWidth": dimensions}))
    network.addRegion(L4Name, "py.ApicalTMPairRegion", json.dumps(L4Params))
    network.addRegion(L6aName, "py.GridCellLocationRegion", json.dumps(L6aParams))

    # Link sensory input to L4
    network.link(sensorInputName, L4Name, "UniformLink", "", srcOutput="dataOut", destInput="activeColumns")

    # Link motor input to L6a
    network.link(motorInputName, L6aName, "UniformLink", "", srcOutput="dataOut", destInput="displacement")

    # Link L6a to L4
    network.link(L6aName, L4Name, "UniformLink", "", srcOutput="activeCells", destInput="basalInput")
    network.link(L6aName, L4Name, "UniformLink", "", srcOutput="learnableCells", destInput="basalGrowthCandidates")

    # Link L4 feedback to L6a
    network.link(L4Name, L6aName, "UniformLink", "", srcOutput="activeCells", destInput="anchorInput")
    network.link(L4Name, L6aName, "UniformLink", "", srcOutput="winnerCells", destInput="anchorGrowthCandidates")

    # Link reset signal to L4 and L6a
    network.link(sensorInputName, L4Name, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
    network.link(sensorInputName, L6aName, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

    # Set phases appropriately
    network.setPhases(motorInputName, set([0]))
    network.setPhases(sensorInputName, set([0]))
    network.setPhases(L4Name, set([2]))
    network.setPhases(L6aName, set([1, 3]))

    return network



def createL246aLocationColumn(network, L2Params, L4Params, L6aParams,
                                                            baselineCellsPerAxis=6,
                                                            inverseReadoutResolution=None, suffix=""):
    """
    Create a single column network composed of L2, L4 and L6a layers.
    L2 layer computes the object representation using :class:`ColumnPoolerRegion`,
    L4 layer processes sensors input while L6a processes motor commands using grid
    cell modules. Sensory input is represented by the feature's active columns and
    motor input is represented by the displacement vector [dx, dy].

    The grid cell modules used by this network are based on
    :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
    from on one or more Gaussian activity bumps. The cells are distributed
    uniformly through the rhombus, packed in the optimal hexagonal arrangement.
    ::

   Phase                       +-------+
   -----                reset  |       |
                        +----->|  L2   |<------------------+
   [3]                  |      |       |                   |
                        |      +-------+                   |
                        |        |   ^                     |
                        |        |   |                     |
                        |     +1 |   |                     |
                        |        v   |                     |
                        |      +-------+                   |
                  +----------->|       |--winnerCells------+
   [2]            |     |      |  L4   |<------------+
                  |     +----->|       |--winner---+ |
                  |     |      +-------+           | |
                  |     |        |   ^             | |
                  |     |        |   |             | |
                  |     |        |   |             | |
                  |     |        v   |             | |
                  |     |      +-------+           | |
                  |     |      |       |           | |
    [1,3]         |     +----->|  L6a  |<----------+ |
                  |     |      |       |--learnable--+
                  |     |      +-------+
             feature  reset        ^
                  |     |          |
                  |     |          |
    [0]        [sensorInput]  [motorInput]


    .. note::
        Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
        Each name has an optional string suffix appended to it.

    :param network: network to add the column
    :type network: Network
    :param L2Params:    constructor parameters for :class:`ColumnPoolerRegion`
    :type L2Params: dict
    :param L4Params:    constructor parameters for :class:`ApicalTMPairRegion`
    :type L4Params: dict
    :param L6aParams:    constructor parameters for :class:`GridCellLocationRegion`
    :type L6aParams: dict
    :param inverseReadoutResolution: Optional readout resolution.
        The readout resolution specifies the diameter of the circle of phases in the
        rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
    :type inverseReadoutResolution: int
    :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
        resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
        that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
        the readout resolution is approximately 1/4
    :type baselineCellsPerAxis: int or float
    :param suffix: optional string suffix appended to region name. Useful when
                                 creating multicolumn networks.
    :type suffix: str
    :return: Reference to the given network
    :rtype: Network
    """

    # Configure L2 'inputWidth' to be compatible with L4
    numOfcols = L4Params["columnCount"]
    cellsPerCol = L4Params["cellsPerColumn"]
    L2Params = copy.deepcopy(L2Params)
    L2Params["inputWidth"] = numOfcols * cellsPerCol

    # Configure L4 'apicalInputWidth' to be compatible L2 output
    L4Params = copy.deepcopy(L4Params)
    L4Params["apicalInputWidth"] = L2Params["cellCount"]

    # Add L4 - L6a location layers
    network = createL4L6aLocationColumn(network=network,
                                        L4Params=L4Params,
                                        L6aParams=L6aParams,
                                        inverseReadoutResolution=inverseReadoutResolution,
                                        baselineCellsPerAxis=baselineCellsPerAxis,
                                        suffix=suffix)
    L4Name = "L4" + suffix
    sensorInputName = "sensorInput" + suffix

    # Add L2 - L4 object layers
    L2Name = "L2" + suffix
    network.addRegion(L2Name, "py.ColumnPoolerRegion", json.dumps(L2Params))

    # Link L4 to L2
    network.link(L4Name, L2Name, "UniformLink", "", srcOutput="activeCells", destInput="feedforwardInput")
    network.link(L4Name, L2Name, "UniformLink", "", srcOutput="winnerCells", destInput="feedforwardGrowthCandidates")

    # Link L2 feedback to L4
    network.link(L2Name, L4Name, "UniformLink", "", srcOutput="feedForwardOutput", destInput="apicalInput", propagationDelay=1)

    # Link reset output to L2
    network.link(sensorInputName, L2Name, "UniformLink", "", srcOutput="resetOut", destInput="resetIn")

    # Set L2 phase to be after L4
    network.setPhases(L2Name, set([3]))

    return network



def createMultipleL246aLocationColumn(network, numberOfColumns, L2Params,
                                        L4Params, L6aParams,
                                        inverseReadoutResolution=None,
                                        baselineCellsPerAxis=6):
    """
    Create a network consisting of multiple columns. Each column contains one L2,
    one L4 and one L6a layers identical in structure to the network created by
    :func:`createL246aLocationColumn`. In addition all the L2 columns are fully
    connected to each other through their lateral inputs.
    ::

                            +----lateralInput--+
                            | +--------------+ |
                            | |       +1     | |
 Phase                      v |              v |
 -----                   +-------+         +-------+
                  reset  |       |         |       | reset
 [3]              +----->|  L2   |         |  L2   |<----+
                  |      |       |         |       |     |
                  |      +-------+         +-------+     |
                  |        |   ^             |   ^       |
                  |     +1 |   |          +1 |   |       |
                  |        |   |             |   |       |
                  |        v   |             v   |       |
                  |      +-------+         +-------+     |
 [2]        +----------->|       |         |       |<----------+
            |     |      |  L4   |         |  L4   |     |     |
            |     +----->|       |         |       |<----+     |
            |     |      +-------+         +-------+     |     |
            |     |        |   ^             |   ^       |     |
            |     |        |   |             |   |       |     |
            |     |        |   |             |   |       |     |
            |     |        v   |             v   |       |     |
            |     |      +-------+         +-------+     |     |
            |     |      |       |         |       |     |     |
 [1,3]      |     +----->|  L6a  |         |  L6a  |<----+     |
            |     |      |       |         |       |     |     |
            |     |      +-------+         +-------+     |     |
       feature  reset        ^                 ^      reset  feature
            |     |          |                 |         |     |
            |     |          |                 |         |     |
 [0]     [sensorInput]  [motorInput]      [motorInput] [sensorInput]

    .. note::
        Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
        Each name has column number appended to it.
        For example: "sensorInput_0", "L2_1", "L6a_0" etc.

    :param network: network to add the column
    :type network: Network
    :param numberOfColumns: Number of columns to create
    :type numberOfColumns: int
    :param L2Params:    constructor parameters for :class:`ColumnPoolerRegion`
    :type L2Params: dict
    :param L4Params:    constructor parameters for :class:`ApicalTMPairRegion`
    :type L4Params: dict
    :param L6aParams:    constructor parameters for :class:`GridCellLocationRegion`
    :type L6aParams: dict
    :param inverseReadoutResolution: Optional readout resolution.
        The readout resolution specifies the diameter of the circle of phases in the
        rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
    :type inverseReadoutResolution: int
    :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
        resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
        that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
        the readout resolution is approximately 1/4
    :type baselineCellsPerAxis: int or float
    :return: Reference to the given network
    :rtype: Network
    """
    L2Params = copy.deepcopy(L2Params)
    L4Params = copy.deepcopy(L4Params)
    L6aParams = copy.deepcopy(L6aParams)

    # Update L2 numOtherCorticalColumns parameter
    L2Params["numOtherCorticalColumns"] = numberOfColumns - 1

    for i in range(numberOfColumns):
        # Make sure random seed is different for each column
        L2Params["seed"] = L2Params.get("seed", 42) + i
        L4Params["seed"] = L4Params.get("seed", 42) + i
        L6aParams["seed"] = L6aParams.get("seed", 42) + i

        # Create column
        network = createL246aLocationColumn(network=network,
                                            L2Params=L2Params,
                                            L4Params=L4Params,
                                            L6aParams=L6aParams,
                                            inverseReadoutResolution=inverseReadoutResolution,
                                            baselineCellsPerAxis=baselineCellsPerAxis,
                                            suffix="_" + str(i))

    # Now connect the L2 columns laterally
    if numberOfColumns > 1:
        for i in range(numberOfColumns):
            src = str(i)
            for j in range(numberOfColumns):
                if i != j:
                    dest = str(j)
                    network.link("L2_" + src, "L2_" + dest, "UniformLink", "", srcOutput="feedForwardOutput", destInput="lateralInput", propagationDelay=1)

    return network



class L246aNetwork(object):
    """
    This class allows to easily create experiments using a L2-L4-L6a network for
    inference over objects. It uses the network created via
    :func:`createMultipleL246aLocationColumn`. See 'l2l4l6aexperiment.py' for
    examples on how to use this class
    """

    @staticmethod
    def rerunExperimentFromLogfile(logFilename):
        """
        Create an experiment class according to the sequence of operations in
        logFile and return resulting experiment instance. The log file is created
        by setting the 'logCalls' constructor parameter to True
        """
        callLog = LoggingDecorator.load(logFilename)

        # Assume first one is call to constructor
        exp = L246aNetwork(*callLog[0][1]["args"], **callLog[0][1]["kwargs"])

        # Call subsequent methods, using stored parameters
        for call in callLog[1:]:
            method = getattr(exp, call[0])
            method(*call[1]["args"], **call[1]["kwargs"])

        return exp

    @LoggingDecorator()
    def __init__(self, numColumns, L2Params, L4Params, L6aParams, repeat, logCalls=False):
        """
        Create a network consisting of multiple columns. Each column contains one L2,
        one L4 and one L6a layers. In addition all the L2 columns are fully
        connected to each other through their lateral inputs.

        :param numColumns: Number of columns to create
        :type numColumns: int
        :param L2Params: constructor parameters for :class:`ColumnPoolerRegion`
        :type L2Params: dict
        :param L4Params:    constructor parameters for :class:`ApicalTMPairRegion`
        :type L4Params: dict
        :param L6aParams: constructor parameters for :class:`GridCellLocationRegion`
        :type L6aParams: dict
        :param repeat: Number of times each pair should be seen to be learned
        :type repeat: int
        :param logCalls: If true, calls to main functions will be logged internally.
                                         The log can then be saved with saveLogs(). This allows us
                                         to recreate the complete network behavior using
                                         rerunExperimentFromLogfile which is very useful for
                                         debugging.
        :type logCalls: bool
        """
        # Handle logging - this has to be done first
        self.logCalls = logCalls

        self.numColumns = numColumns
        self.repeat = repeat

        network = Network()
        self.network = createMultipleL246aLocationColumn(network=network,
                                                         numberOfColumns=self.numColumns,
                                                         L2Params=L2Params,
                                                         L4Params=L4Params,
                                                         L6aParams=L6aParams)
        network.initialize()

        self.sensorInput = []
        self.motorInput = []
        self.L2Regions = []
        self.L4Regions = []
        self.L6aRegions = []
        for i in range(self.numColumns):
            col = str(i)
            self.sensorInput.append(network.getRegion("sensorInput_" + col))
            self.motorInput.append(network.getRegion("motorInput_" + col))
            self.L2Regions.append(network.getRegion("L2_" + col))
            self.L4Regions.append(network.getRegion("L4_" + col))
            self.L6aRegions.append(network.getRegion("L6a_" + col))

        if L6aParams is not None and "dimensions" in L6aParams:
            self.dimensions = L6aParams["dimensions"]
        else:
            self.dimensions = 2

        self.sdrSize = L2Params["sdrSize"]

        # will be populated during training
        self.learnedObjects = {}

    @LoggingDecorator()
    def sendReset(self):            
        for col in range(self.numColumns):
            displacement = [0] * self.dimensions
            Sensors.addDataToQueue(self.sensorInput[col], [], True, 0)
            Motors.addDataToQueue(self.motorInput[col], displacement, reset=True)

        self.network.run(1)

    @LoggingDecorator()
    def setLearning(self, learn):
        for col in range(self.numColumns):
            self.L2Regions[col].setParameterBool("learningMode", learn)
            self.L4Regions[col].setParameterBool("learn", learn)
            self.L6aRegions[col].setParameterBool("learningMode", learn)

    @LoggingDecorator()
    def learn(self, objects):
        """
        Learns all provided objects

        :param objects: dict mapping object name to array of sensations, where each
                                        sensation is composed of location and feature SDR for each
                                        column. For example:
                                        {'obj1' : [[[1,1,1],[101,205,523, ..., 1021]],...], ...}
                                        Note: Each column must have the same number of sensations as
                                        the other columns.
        :type objects: dict[str, array]
        """
        self.setLearning(True)

        for objectName, sensationList in objects.items():
            self.sendReset()
            print("Learning :", objectName)

            prevLoc = [None] * self.numColumns
            numFeatures = len(sensationList[0])
            displacement = [0] * self.dimensions

            for sensation in range(numFeatures):
                for col in range(self.numColumns):
                    location = np.array(sensationList[col][sensation][0])
                    feature = sensationList[col][sensation][1]

                    # Compute displacement from previous location
                    if prevLoc[col] is not None:
                        displacement = location - prevLoc[col]
                    prevLoc[col] = location

                    # learn each pattern multiple times
                    for _ in range(self.repeat):
                        # Sense feature at location
                        Motors.addDataToQueue(self.motorInput[col], displacement)
                        Sensors.addDataToQueue(self.sensorInput[col], feature, False, 0)
                        # Only move to the location on the first sensation.
                        displacement = [0] * self.dimensions

            self.network.run(self.repeat * numFeatures)

            # update L2 representations for the object
            self.learnedObjects[objectName] = self.getL2Representations()

    def infer(self, sensations, stats=None, objname=None):
        """
        Attempt to recognize the object given a list of sensations.
        You may use :meth:`getCurrentClassification` to extract the current object
        classification from the network

        :param sensations: Array of sensations, where each sensation is composed of
                                             displacement vector and feature SDR for each column.
                                             For example: [[[1,1,1],[101,205,523, ..., 1021]],...]
                                             Note: Each column must have the same number of sensations
                                             as the other columns.

        :type sensations: list[tuple[list[int], list[int]]]
        :param stats: Dictionary holding statistics information.
                                    See '_updateInferenceStats' for information on the statistics
                                    collected
        :type stats: defaultdict[str, list]
        :param objname: Name of the inferred object, if known
        :type objname: str or None
        """
        self.setLearning(False)

        prevLoc = [None] * self.numColumns
        numFeatures = len(sensations[0])
        for sensation in range(numFeatures):
            for col in range(self.numColumns):
                assert numFeatures == len(sensations[col])

                location, feature = sensations[col][sensation]
                # Compute displacement from previous location
                location = np.array(location)
                displacement = [0] * self.dimensions
                if prevLoc[col] is not None:
                    displacement = location - prevLoc[col]
                prevLoc[col] = location

                Motors.addDataToQueue(self.motorInput[col], displacement)
                Sensors.addDataToQueue(self.sensorInput[col], feature, False, 0)

            self.network.run(1)
            if stats is not None:
                self.updateInferenceStats(stats=stats, objectName=objname)

    def updateInferenceStats(self, stats, objectName=None):
        """
        Updates the inference statistics.

        :param objectName: Name of the inferred object, if known
        :type objectName: str or None

        """
        L6aLearnableCells = self.getL6aLearnableCells()
        L6aSensoryAssociatedCells = self.getL6aSensoryAssociatedCells()
        L6aRepresentations = self.getL6aRepresentations()
        L4Representations = self.getL4Representations()
        L4PredictedCells = self.getL4PredictedCells()
        L2Representation = self.getL2Representations()

        for i in range(self.numColumns):
            stats["L6a SensoryAssociatedCells C" + str(i)].append(len(L6aSensoryAssociatedCells[i]))
            stats["L6a LearnableCells C" + str(i)].append(len(L6aLearnableCells[i]))
            stats["L6a Representation C" + str(i)].append(len(L6aRepresentations[i]))
            stats["L4 Representation C" + str(i)].append(len(L4Representations[i]))
            stats["L4 Predicted C" + str(i)].append(len(L4PredictedCells[i]))
            stats["L2 Representation C" + str(i)].append(len(L2Representation[i]))
            stats["Full L2 SDR C" + str(i)].append(sorted([int(c) for c in L2Representation[i]]))

            # add true overlap and classification result if objectName was learned
            if objectName in self.learnedObjects:
                objectRepresentation = self.learnedObjects[objectName]
                stats["Overlap L2 with object C" + str(i)].append(len(objectRepresentation[i] & L2Representation[i]))

        if objectName in self.learnedObjects:
            if self.isObjectClassified(objectName, minOverlap=30):
                stats["Correct classification"].append(1.0)
            else:
                stats["Correct classification"].append(0.0)

    def getL2Representations(self):
        """
        Returns the active representation in L2.
        """
        return [set(np.array(L2.getOutputArray("activeCells")).nonzero()[0]) for L2 in self.L2Regions]

    def getL4Representations(self):
        """
        Returns the active representation in L4.
        """
        return [set(np.array(L4.getOutputArray("activeCells")).nonzero()[0]) for L4 in self.L4Regions]

    def getL4PredictedCells(self):
        """
        Returns the cells in L4 that were predicted by the location input.
        """
        return [set(np.array(L4.getOutputArray("predictedCells")).nonzero()[0]) for L4 in self.L4Regions]

    def getL4PredictedActiveCells(self):
        """
        Returns the cells in L4 that were predicted by the location signal
        and are currently active.    Does not consider apical input.
        """
        return [set(np.array(L4.getOutputArray("predictedActiveCells")).nonzero()[0]) for L4 in self.L4Regions]

    def getL6aRepresentations(self):
        """
        Returns the active representation in L6a.
        """
        return [set(np.array(L6a.getOutputArray("activeCells")).nonzero()[0]) for L6a in self.L6aRegions]

    def getL6aLearnableCells(self):
        """
        Returns the sensoryAssociatedCells in L6a.
        """
        return [set(np.array(L6a.getOutputArray("learnableCells")).nonzero()[0]) for L6a in self.L6aRegions]

    def getL6aSensoryAssociatedCells(self):
        """
        Returns the sensoryAssociatedCells in L6a.
        """
        return [set(np.array(L6a.getOutputArray("sensoryAssociatedCells")).nonzero()[0]) for L6a in self.L6aRegions]

    def isObjectClassified(self, objectName, minOverlap=None, maxL2Size=None):
        """
        Return True if objectName is currently unambiguously classified by every L2
        column. Classification is correct and unambiguous if the current L2 overlap
        with the true object is greater than minOverlap and if the size of the L2
        representation is no more than maxL2Size

        :param minOverlap: min overlap to consider the object as recognized.
                                             Defaults to half of the SDR size

        :param maxL2Size: max size for the L2 representation
                                            Defaults to 1.5 * SDR size

        :return: True/False
        """
        l2sdr = self.getL2Representations()
        try:
            objectRepresentation = self.learnedObjects[objectName]
        except:
            return False
        
        if minOverlap is None:
            minOverlap = self.sdrSize // 2
        if maxL2Size is None:
            maxL2Size = 1.5 * self.sdrSize

        numCorrectClassifications = 0
        for col in range(self.numColumns):
            # Ignore inactive column
            if len(l2sdr[col]) == 0:
                continue

            overlapWithObject = len(objectRepresentation[col] & l2sdr[col])
            if (overlapWithObject >= minOverlap and len(l2sdr[col]) <= maxL2Size):
                numCorrectClassifications += 1

        return numCorrectClassifications == self.numColumns

    def getCurrentClassification(self, minOverlap=None, includeZeros=True):
        """
        Return the current classification for every object.    Returns a dict with a
        score for each object. Score goes from 0 to 1. A 1 means every col (that has
        received input since the last reset) currently has overlap >= minOverlap
        with the representation for that object.

        :param minOverlap: min overlap to consider the object as recognized.
                                             Defaults to half of the SDR size
        :type minOverlap: int
        :param includeZeros: if True, include scores for all objects, even if 0
        :type includeZeros: bool

        :return: dict of object names and their score
        :rtype: dict[str, float]
        """
        results = {}
        l2sdr = self.getL2Representations()
        if minOverlap is None:
            minOverlap = self.sdrSize // 2

        for objectName, objectSdr in self.learnedObjects.items():
            count = 0
            score = 0.0
            for col in range(self.numColumns):
                # Ignore inactive column
                if len(l2sdr[col]) == 0:
                    continue

                count += 1
                overlap = len(l2sdr[col] & objectSdr[col])
                if overlap >= minOverlap:
                    score += 1

            if count == 0:
                if includeZeros:
                    results[objectName] = 0
            else:
                if includeZeros or score > 0.0:
                    results[objectName] = score / count

        return results
