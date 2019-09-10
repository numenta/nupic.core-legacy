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
Region for Temporal Memory with various apical implementations.
"""

import numpy as np

from htm.bindings.regions.PyRegion import PyRegion
from htm_advanced.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory


class ApicalTMPairRegion(PyRegion):
    """
    Implements pair memory with the TM for the HTM network API. The temporal
    memory uses basal and apical dendrites.
    """

    @classmethod
    def getSpec(cls):
        """
        Return the Spec for ApicalTMPairRegion
        """

        spec = {
            "description": ApicalTMPairRegion.__doc__,
            "singleNodeOnly": True,
            "inputs": {
                "activeColumns": {
                    "description": ("An array of 0's and 1's representing the active "
                                    "minicolumns, i.e. the input to the TemporalMemory"),
                    "dataType": "Real32",
                    "count": 0,
                    "required": True,
                    "regionLevel": True,
                    "isDefaultInput": True,
                    "requireSplitterMap": False
                },
                "resetIn": {
                    "description": ("A boolean flag that indicates whether"
                                    " or not the input vector received in this compute cycle"
                                    " represents the first presentation in a"
                                    " new temporal sequence."),
                    "dataType": "Real32",
                    "count": 1,
                    "required": False,
                    "regionLevel": True,
                    "isDefaultInput": False,
                    "requireSplitterMap": False
                },
                "basalInput": {
                    "description": "An array of 0's and 1's representing basal input",
                    "dataType": "Real32",
                    "count": 0,
                    "required": False,
                    "regionLevel": True,
                    "isDefaultInput": False,
                    "requireSplitterMap": False
                },
                "basalGrowthCandidates": {
                    "description": ("An array of 0's and 1's representing basal input " +
                                    "that can be learned on new synapses on basal " +
                                    "segments. If this input is a length-0 array, the " +
                                    "whole basalInput is used."),
                    "dataType": "Real32",
                    "count": 0,
                    "required": False,
                    "regionLevel": True,
                    "isDefaultInput": False,
                    "requireSplitterMap": False
                },
                "apicalInput": {
                    "description": "An array of 0's and 1's representing top down input."
                    " The input will be provided to apical dendrites.",
                    "dataType": "Real32",
                    "count": 0,
                    "required": False,
                    "regionLevel": True,
                    "isDefaultInput": False,
                    "requireSplitterMap": False
                },
                "apicalGrowthCandidates": {
                    "description": ("An array of 0's and 1's representing apical input " +
                                    "that can be learned on new synapses on apical " +
                                    "segments. If this input is a length-0 array, the " +
                                    "whole apicalInput is used."),
                    "dataType": "Real32",
                    "count": 0,
                    "required": False,
                    "regionLevel": True,
                    "isDefaultInput": False,
                    "requireSplitterMap": False},
            },
            "outputs": {
                "predictedCells": {
                    "description": ("A binary output containing a 1 for every "
                                    "cell that was predicted for this timestep."),
                    "dataType": "Real32",
                    "count": 0,
                    "regionLevel": True,
                    "isDefaultOutput": False
                },

                "predictedActiveCells": {
                    "description": ("A binary output containing a 1 for every "
                                                    "cell that transitioned from predicted to active."),
                    "dataType": "Real32",
                    "count": 0,
                    "regionLevel": True,
                    "isDefaultOutput": False
                },

                "activeCells": {
                    "description": ("A binary output containing a 1 for every "
                                    "cell that is currently active."),
                    "dataType": "Real32",
                    "count": 0,
                    "regionLevel": True,
                    "isDefaultOutput": True
                },

                "winnerCells": {
                    "description": ("A binary output containing a 1 for every "
                                    "'winner' cell in the TM."),
                    "dataType": "Real32",
                    "count": 0,
                    "regionLevel": True,
                    "isDefaultOutput": False
                },
            },

            "parameters": {

                # Input sizes (the network API doesn't provide these during initialize)
                "columnCount": {
                    "description": ("The size of the 'activeColumns' input "
                                    "(i.e. the number of columns)"),
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },

                "basalInputWidth": {
                    "description": "The size of the 'basalInput' input",
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },

                "apicalInputWidth": {
                    "description": "The size of the 'apicalInput' input",
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },

                "learn": {
                    "description": "True if the TM should learn.",
                    "accessMode": "ReadWrite",
                    "dataType": "Bool",
                    "count": 1,
                    "defaultValue": "true"
                },
                "cellsPerColumn": {
                    "description": "Number of cells per column",
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },
                "activationThreshold": {
                    "description": ("If the number of active connected synapses on a "
                                    "segment is at least this threshold, the segment "
                                    "is said to be active."),
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },
                "reducedBasalThreshold": {
                    "description": ("Activation threshold of basal segments for cells "
                                    "with active apical segments (with apicalTiebreak "
                                    "implementation). "),
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite",
                },
                "initialPermanence": {
                    "description": "Initial permanence of a new synapse.",
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },
                "connectedPermanence": {
                    "description": ("If the permanence value for a synapse is greater "
                                    "than this value, it is said to be connected."),
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },
                "minThreshold": {
                    "description": ("If the number of synapses active on a segment is at "
                                    "least this threshold, it is selected as the best "
                                    "matching cell in a bursting column."),
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "constraints": "",
                    "accessMode":"ReadWrite"
                },
                "sampleSize": {
                    "description": ("The desired number of active synapses for an "
                                    "active cell"),
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "learnOnOneCell": {
                    "description": ("If True, the winner cell for each column will be"
                                    " fixed between resets."),
                    "accessMode": "Read",
                    "dataType": "Bool",
                    "count": 1,
                    "defaultValue": "false",
                    "accessMode":"ReadWrite"
                },
                "maxSynapsesPerSegment": {
                    "description": "The maximum number of synapses per segment. Use -1 "
                                    "for unlimited.",
                    "accessMode": "Read",
                    "dataType": "Int32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "maxSegmentsPerCell": {
                    "description": "The maximum number of segments per cell",
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "permanenceIncrement": {
                    "description": ("Amount by which permanences of synapses are "
                                                    "incremented during learning."),
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "permanenceDecrement": {
                    "description": ("Amount by which permanences of synapses are "
                                    "decremented during learning."),
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "basalPredictedSegmentDecrement": {
                    "description": ("Amount by which active permanences of synapses of "
                                    "previously predicted but inactive segments are "
                                    "decremented."),
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "apicalPredictedSegmentDecrement": {
                    "description": ("Amount by which active permanences of synapses of "
                                    "previously predicted but inactive segments are "
                                    "decremented."),
                    "accessMode": "Read",
                    "dataType": "Real32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "seed": {
                    "description": "Seed for the random number generator.",
                    "accessMode": "Read",
                    "dataType": "UInt32",
                    "count": 1,
                    "accessMode":"ReadWrite"
                },
                "implementation": {
                    "description": "Apical implementation",
                    "accessMode": "Read",
                    "dataType": "Byte",
                    "count": 0,
                    "constraints": ("enum: ApicalTiebreak, ApicalTiebreakCPP, ApicalDependent"),
                    "defaultValue": "ApicalTiebreakCPP",
                    "accessMode":"ReadWrite"
                },
            },
        }

        return spec


    def __init__(self,

             # Input sizes
             columnCount,
             basalInputWidth,
             apicalInputWidth=0,

             # TM params
             cellsPerColumn=32,
             activationThreshold=13,
             initialPermanence=0.21,
             connectedPermanence=0.50,
             minThreshold=10,
             reducedBasalThreshold=13, # ApicalTiebreak and ApicalDependent only
             sampleSize=20,
             permanenceIncrement=0.10,
             permanenceDecrement=0.10,
             basalPredictedSegmentDecrement=0.0,
             apicalPredictedSegmentDecrement=0.0,
             learnOnOneCell=False, # ApicalTiebreakCPP only
             maxSegmentsPerCell=255,
             maxSynapsesPerSegment=255, # ApicalTiebreakCPP only
             seed=42,

             # Region params
             implementation="ApicalTiebreak",
             learn=True,
             **kwargs):

        # Input sizes (the network API doesn't provide these during initialize)
        self.columnCount = columnCount
        self.basalInputWidth = basalInputWidth
        self.apicalInputWidth = apicalInputWidth

        # TM params
        self.cellsPerColumn = cellsPerColumn
        self.activationThreshold = activationThreshold
        self.reducedBasalThreshold = reducedBasalThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.sampleSize = sampleSize
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
        self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.learnOnOneCell = learnOnOneCell
        self.seed = seed

        # Region params
        self.implementation = implementation
        self.learn = learn

        PyRegion.__init__(self, **kwargs)

        # TM instance
        self._tm = None


    def initialize(self):
        """
        Initialize the self._tm if not already initialized.
        """

        if self._tm is None:
            params = {
                "columnCount": self.columnCount,
                "basalInputSize": self.basalInputWidth,
                "apicalInputSize": self.apicalInputWidth,
                "cellsPerColumn": self.cellsPerColumn,
                "activationThreshold": self.activationThreshold,
                "initialPermanence": self.initialPermanence,
                "connectedPermanence": self.connectedPermanence,
                "minThreshold": self.minThreshold,
                "sampleSize": self.sampleSize,
                "permanenceIncrement": self.permanenceIncrement,
                "permanenceDecrement": self.permanenceDecrement,
                "basalPredictedSegmentDecrement": self.basalPredictedSegmentDecrement,
                "apicalPredictedSegmentDecrement": self.apicalPredictedSegmentDecrement,
                "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
                "seed": self.seed,
            }

            if self.implementation == "ApicalTiebreakCPP": # TODO
                params["learnOnOneCell"] = self.learnOnOneCell
                params["maxSegmentsPerCell"] = self.maxSegmentsPerCell

                cls = ApicalTiebreakPairMemory

            elif self.implementation == "ApicalTiebreak":
                params["reducedBasalThreshold"] = self.reducedBasalThreshold

                cls = ApicalTiebreakPairMemory

            else:
                raise ValueError("Unrecognized implementation %s" % self.implementation)

            self._tm = cls(**params)

    def compute(self, inputs, outputs):
        """
        Run one iteration of TM's compute.
        """

        # If there's a reset, don't call compute. In some implementations, an empty
        # input might cause unwanted effects.
        if "resetIn" in inputs:
            assert len(inputs["resetIn"]) == 1
            if inputs["resetIn"][0] != 0:
                # send empty output
                self._tm.reset()
                outputs["activeCells"][:] = 0
                outputs["predictedActiveCells"][:] = 0
                outputs["winnerCells"][:] = 0
                return

        activeColumns = inputs["activeColumns"].nonzero()[0]

        if "basalInput" in inputs:
            basalInput = inputs["basalInput"].nonzero()[0]
        else:
            basalInput = np.empty(0, dtype="uint32")

        if "apicalInput" in inputs:
            apicalInput = inputs["apicalInput"].nonzero()[0]
        else:
            apicalInput = np.empty(0, dtype="uint32")

        if "basalGrowthCandidates" in inputs:
            basalGrowthCandidates = inputs["basalGrowthCandidates"].nonzero()[0]
        else:
            basalGrowthCandidates = basalInput

        if "apicalGrowthCandidates" in inputs:
            apicalGrowthCandidates = inputs["apicalGrowthCandidates"].nonzero()[0]
        else:
            apicalGrowthCandidates = apicalInput

        self._tm.compute(activeColumns, basalInput, apicalInput, basalGrowthCandidates, apicalGrowthCandidates, self.learn)

        # Extract the active / predicted cells and put them into binary arrays.
        outputs["activeCells"][:] = 0
        outputs["activeCells"][self._tm.getActiveCells()] = 1
        outputs["predictedCells"][:] = 0
        outputs["predictedCells"][self._tm.getPredictedCells()] = 1
        outputs["predictedActiveCells"][:] = (outputs["activeCells"] * outputs["predictedCells"])
        outputs["winnerCells"][:] = 0
        outputs["winnerCells"][self._tm.getWinnerCells()] = 1


    def getParameter(self, parameterName, index=-1):
        """
            Get the value of a NodeSpec parameter. Most parameters are handled
            automatically by PyRegion's parameter get mechanism. The ones that need
            special treatment are explicitly handled here.
        """
        return PyRegion.getParameter(self, parameterName, index)


    def setParameter(self, parameterName, _, parameterValue):
        """
        Set the value of a Spec parameter. Most parameters are handled
        automatically by PyRegion's parameter set mechanism. The ones that need
        special treatment are explicitly handled here.
        """
        if hasattr(self, parameterName):
            setattr(self, parameterName, parameterValue)
        else:
            raise Exception("Unknown parameter: " + parameterName)


    def reset(self):
        """
        Explicitly reset the TM.
        """
        self._tm.reset()


    def getOutputElementCount(self, name):
        """
        Return the number of elements for the given output.
        """
        if name in ["activeCells", "predictedCells", "predictedActiveCells",
                                "winnerCells"]:
            return self.cellsPerColumn * self.columnCount
        else:
            raise Exception("Invalid output name specified: %s" % name)


    def getAlgorithmInstance(self):
        """
        Returns an instance of the underlying temporal memory instance
        """
        return self._tm
