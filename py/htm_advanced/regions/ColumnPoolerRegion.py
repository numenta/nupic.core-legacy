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

import numpy
import inspect

from htm.bindings.regions.PyRegion import PyRegion
from htm_advanced.algorithms.column_pooler import ColumnPooler


def getConstructorArguments():
    """
    Return constructor argument associated with ColumnPooler.
    @return defaults (list)     a list of args and default values for each argument
    """
    argspec = inspect.getargspec(ColumnPooler.__init__)
    return argspec.args[1:], argspec.defaults


class ColumnPoolerRegion(PyRegion):
    """
    The ColumnPoolerRegion implements an L2 layer within a single cortical column / cortical
    module.

    The layer supports feed forward (proximal) and lateral inputs.
    """

    @classmethod
    def getSpec(cls):
        """
        Return the Spec for ColumnPoolerRegion.

        The parameters collection is constructed based on the parameters specified
        by the various components (tmSpec and otherSpec)
        """
        spec = dict(
            description=ColumnPoolerRegion.__doc__,
            singleNodeOnly=True,
            inputs=dict(
                feedforwardInput=dict(
                    description="The primary feed-forward input to the layer, this is a"
                                " binary array containing 0's and 1's",
                    dataType="Real32",
                    count=0,
                    required=True,
                    regionLevel=True,
                    isDefaultInput=True,
                    requireSplitterMap=False),

                feedforwardGrowthCandidates=dict(
                    description=("An array of 0's and 1's representing feedforward input " +
                                 "that can be learned on new proximal synapses. If this " +
                                 "input isn't provided, the whole feedforwardInput is "
                                 "used."),
                    dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False),

                predictedInput=dict(
                    description=("An array of 0s and 1s representing input cells that " +
                                 "are predicted to become active in the next time step. " +
                                 "If this input is not provided, some features related " +
                                 "to online learning may not function properly."),
                    dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False),

                lateralInput=dict(
                    description="Lateral binary input into this column, presumably from"
                                " other neighboring columns.",
                    dataType="Real32",
                    count=0,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False),

                resetIn=dict(
                    description="A boolean flag that indicates whether"
                                " or not the input vector received in this compute cycle"
                                " represents the first presentation in a"
                                " new temporal sequence.",
                    dataType='Real32',
                    count=1,
                    required=False,
                    regionLevel=True,
                    isDefaultInput=False,
                    requireSplitterMap=False),

            ),
            outputs=dict(
                feedForwardOutput=dict(
                    description="The default output of ColumnPoolerRegion. By default this"
                                " outputs the active cells. You can change this "
                                " dynamically using the defaultOutputType parameter.",
                    dataType="Real32",
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=True),

                activeCells=dict(
                    description="A binary output containing a 1 for every"
                                " cell that is currently active.",
                    dataType="Real32",
                    count=0,
                    regionLevel=True,
                    isDefaultOutput=False),

            ),
            parameters=dict(
                learningMode=dict(
                    description="Whether the node is learning (default True).",
                    accessMode="ReadWrite",
                    dataType="Bool",
                    count=1,
                    defaultValue="true"),
                onlineLearning=dict(
                    description="Whether to use onlineLearning or not (default False).",
                    accessMode="ReadWrite",
                    dataType="Bool",
                    count=1,
                    defaultValue="false"),
                learningTolerance=dict(
                    description="How much variation in SDR size to accept when learning. "
                                "Only has an effect if online learning is enabled. "
                                "Should be at most 1 - inertiaFactor.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    defaultValue="false"),
                cellCount=dict(
                    description="Number of cells in this layer",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                inputWidth=dict(
                    description='Number of inputs to the layer.',
                    accessMode='ReadWrite',
                    dataType='UInt32',
                    count=1,
                    constraints=''),
                numOtherCorticalColumns=dict(
                    description="The number of lateral inputs that this L2 will receive. "
                                "This region assumes that every lateral input is of size "
                                "'cellCount'.",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                sdrSize=dict(
                    description="The number of active cells invoked per object",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                maxSdrSize=dict(
                    description="The largest number of active cells in an SDR tolerated "
                                "during learning. Stops learning when unions are active.",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                minSdrSize=dict(
                    description="The smallest number of active cells in an SDR tolerated "
                                "during learning.    Stops learning when possibly on a "
                                "different object or sequence",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),

                #
                # Proximal
                #
                synPermProximalInc=dict(
                    description="Amount by which permanences of proximal synapses are "
                                "incremented during learning.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1),
                synPermProximalDec=dict(
                    description="Amount by which permanences of proximal synapses are "
                                "decremented during learning.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1),
                initialProximalPermanence=dict(
                    description="Initial permanence of a new proximal synapse.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),
                sampleSizeProximal=dict(
                    description="The desired number of active synapses for an active cell",
                    accessMode="ReadWrite",
                    dataType="Int32",
                    count=1),
                minThresholdProximal=dict(
                    description="If the number of synapses active on a proximal segment "
                                "is at least this threshold, it is considered as a "
                                "candidate active cell",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                connectedPermanenceProximal=dict(
                    description="If the permanence value for a synapse is greater "
                                "than this value, it is said to be connected.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),
                predictedInhibitionThreshold=dict(
                    description="How many predicted cells are required to cause "
                                "inhibition in the pooler.    Only has an effect if online "
                                "learning is enabled.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),

                #
                # Distal
                #
                synPermDistalInc=dict(
                    description="Amount by which permanences of synapses are "
                                "incremented during learning.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1),
                synPermDistalDec=dict(
                    description="Amount by which permanences of synapses are "
                                "decremented during learning.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1),
                initialDistalPermanence=dict(
                    description="Initial permanence of a new synapse.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),
                sampleSizeDistal=dict(
                    description="The desired number of active synapses for an active "
                                "segment.",
                    accessMode="ReadWrite",
                    dataType="Int32",
                    count=1),
                activationThresholdDistal=dict(
                    description="If the number of synapses active on a distal segment is "
                                "at least this threshold, the segment is considered "
                                "active",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1,
                    constraints=""),
                connectedPermanenceDistal=dict(
                    description="If the permanence value for a synapse is greater "
                                "than this value, it is said to be connected.",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),
                inertiaFactor=dict(
                    description="Controls the proportion of previously active cells that "
                                "remain active through inertia in the next timestep (in    "
                                "the absence of inhibition).",
                    accessMode="ReadWrite",
                    dataType="Real32",
                    count=1,
                    constraints=""),



                seed=dict(
                    description="Seed for the random number generator.",
                    accessMode="ReadWrite",
                    dataType="UInt32",
                    count=1),
                defaultOutputType=dict(
                    description="Controls what type of cell output is placed into"
                                " the default output 'feedForwardOutput'",
                    accessMode="ReadWrite",
                    dataType="Byte",
                    count=0,
                    constraints="enum: active,predicted,predictedActiveCells",
                    defaultValue="active"),
            ),
            commands=dict(
                reset=dict(description="Explicitly reset TM states now."),
            )
        )

        return spec


    def __init__(self,
         cellCount=4096,
         inputWidth=16384,
         numOtherCorticalColumns=0,
         sdrSize=40,
         onlineLearning = False,
         maxSdrSize = None,
         minSdrSize = None,

         # Proximal
         synPermProximalInc=0.1,
         synPermProximalDec=0.001,
         initialProximalPermanence=0.6,
         sampleSizeProximal=20,
         minThresholdProximal=1,
         connectedPermanenceProximal=0.50,
         predictedInhibitionThreshold=20,

         # Distal
         synPermDistalInc=0.10,
         synPermDistalDec=0.10,
         initialDistalPermanence=0.21,
         sampleSizeDistal=20,
         activationThresholdDistal=13,
         connectedPermanenceDistal=0.50,
         inertiaFactor=1.,

         seed=42,
         defaultOutputType = "active",
         **kwargs):

        # Used to derive Column Pooler params
        self.numOtherCorticalColumns = numOtherCorticalColumns

        # Column Pooler params
        self.inputWidth = inputWidth
        self.cellCount = cellCount
        self.sdrSize = sdrSize
        self.onlineLearning = onlineLearning
        self.maxSdrSize = maxSdrSize
        self.minSdrSize = minSdrSize
        self.synPermProximalInc = synPermProximalInc
        self.synPermProximalDec = synPermProximalDec
        self.initialProximalPermanence = initialProximalPermanence
        self.sampleSizeProximal = sampleSizeProximal
        self.minThresholdProximal = minThresholdProximal
        self.connectedPermanenceProximal = connectedPermanenceProximal
        self.predictedInhibitionThreshold = predictedInhibitionThreshold
        self.synPermDistalInc = synPermDistalInc
        self.synPermDistalDec = synPermDistalDec
        self.initialDistalPermanence = initialDistalPermanence
        self.sampleSizeDistal = sampleSizeDistal
        self.activationThresholdDistal = activationThresholdDistal
        self.connectedPermanenceDistal = connectedPermanenceDistal
        self.inertiaFactor = inertiaFactor
        self.seed = seed

        # Region params
        self.learningMode = True
        self.defaultOutputType = defaultOutputType

        self._pooler = None

        PyRegion.__init__(self, **kwargs)


    def initialize(self):
        """
        Initialize the internal objects.
        """
        if self._pooler is None:
            params = {
                "inputWidth": self.inputWidth,
                "lateralInputWidths": [self.cellCount] * self.numOtherCorticalColumns,
                "cellCount": self.cellCount,
                "sdrSize": self.sdrSize,
                "onlineLearning": self.onlineLearning,
                "maxSdrSize": self.maxSdrSize,
                "minSdrSize": self.minSdrSize,
                "synPermProximalInc": self.synPermProximalInc,
                "synPermProximalDec": self.synPermProximalDec,
                "initialProximalPermanence": self.initialProximalPermanence,
                "minThresholdProximal": self.minThresholdProximal,
                "sampleSizeProximal": self.sampleSizeProximal,
                "connectedPermanenceProximal": self.connectedPermanenceProximal,
                "predictedInhibitionThreshold": self.predictedInhibitionThreshold,
                "synPermDistalInc": self.synPermDistalInc,
                "synPermDistalDec": self.synPermDistalDec,
                "initialDistalPermanence": self.initialDistalPermanence,
                "activationThresholdDistal": self.activationThresholdDistal,
                "sampleSizeDistal": self.sampleSizeDistal,
                "connectedPermanenceDistal": self.connectedPermanenceDistal,
                "inertiaFactor": self.inertiaFactor,
                "seed": self.seed,
            }
            self._pooler = ColumnPooler(**params)
            self._outputs = {}
            
    def getOutputArray(self, name):
        print(name)
        return self._outputs[name]

    def compute(self, inputs, outputs):
        """
        Run one iteration of compute.

        Note that if the reset signal is True (1) we assume this iteration
        represents the *end* of a sequence. The output will contain the
        representation to this point and any history will then be reset. The output
        at the next compute will start fresh, presumably with bursting columns.
        """
        # Handle reset first (should be sent with an empty signal)
        if "resetIn" in inputs:
            assert len(inputs["resetIn"]) == 1
            if inputs["resetIn"][0] != 0:
                # send empty output
                self.reset()
                outputs["feedForwardOutput"][:] = 0
                outputs["activeCells"][:] = 0
                return

        feedforwardInput = numpy.asarray(inputs["feedforwardInput"].nonzero()[0], dtype="uint32")

        if "feedforwardGrowthCandidates" in inputs:
            feedforwardGrowthCandidates = numpy.asarray(inputs["feedforwardGrowthCandidates"].nonzero()[0], dtype="uint32")
        else:
            feedforwardGrowthCandidates = feedforwardInput

        if "lateralInput" in inputs:
            lateralInputs = tuple(numpy.asarray(singleInput.nonzero()[0], dtype="uint32")
                                                        for singleInput
                                                        in numpy.split(inputs["lateralInput"], self.numOtherCorticalColumns))
        else:
            lateralInputs = ()

        if "predictedInput" in inputs:
            predictedInput = numpy.asarray(
                inputs["predictedInput"].nonzero()[0], dtype="uint32")
        else:
            predictedInput = None

        # Send the inputs into the Column Pooler.
        self._pooler.compute(feedforwardInput, 
                             lateralInputs, 
                             feedforwardGrowthCandidates, 
                             learn=self.learningMode,
                             predictedInput = predictedInput)

        # Extract the active / predicted cells and put them into binary arrays.
        outputs["activeCells"][:] = 0
        outputs["activeCells"][self._pooler.getActiveCells()] = 1
        self._outputs["activeCells"] = outputs["activeCells"]

        # Send appropriate output to feedForwardOutput.
        if self.defaultOutputType == "active":
            outputs["feedForwardOutput"][:] = outputs["activeCells"]
        else:
            raise Exception("Unknown outputType: " + self.defaultOutputType)


    def reset(self):
        """ Reset the state of the layer"""
        if self._pooler is not None:
            self._pooler.reset()


    def getParameter(self, parameterName, index=-1):
        """
        Get the value of a NodeSpec parameter. Most parameters are handled
        automatically by PyRegion's parameter get mechanism. The ones that need
        special treatment are explicitly handled here.
        """
        return PyRegion.getParameter(self, parameterName, index)


    def setParameter(self, parameterName, index, parameterValue):
        """
        Set the value of a Spec parameter.
        """
        if hasattr(self, parameterName):
            setattr(self, parameterName, parameterValue)
        else:
            raise Exception("Unknown parameter: " + parameterName)


    def getOutputElementCount(self, name):
        """
        Return the number of elements for the given output.
        """
        if name in ["feedForwardOutput", "activeCells"]:
            return self.cellCount
        else:
            raise Exception("Invalid output name specified: " + name)


    def getAlgorithmInstance(self):
        """
        Returns an instance of the underlying column pooler instance
        """
        return self._pooler


