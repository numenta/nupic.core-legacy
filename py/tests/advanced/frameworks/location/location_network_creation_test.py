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
    Test L4-L6a location network factory
"""
import math
import random
import unittest
from collections import defaultdict

import numpy as np
from htm.bindings.engine_internal import Network

from htm.advanced.frameworks.location.location_network_creation import createL4L6aLocationColumn
from htm.advanced.support.register_regions import registerAllAdvancedRegions

NUM_OF_COLUMNS = 150
CELLS_PER_COLUMN = 16
NUM_OF_CELLS = NUM_OF_COLUMNS * CELLS_PER_COLUMN

# Feature encoded into active columns. It could be also interpreted as the
# output of the spatial pooler
FEATURE_ACTIVE_COLUMNS = {
    "A": np.random.choice(NUM_OF_COLUMNS, NUM_OF_COLUMNS // 10).tolist(),
    "B": np.random.choice(NUM_OF_COLUMNS, NUM_OF_COLUMNS // 10).tolist(),
}

OBJECTS = [
    {"name": "Object 1",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                                {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"},
                                {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"}
                                ]},
    {"name": "Object 2",
     "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "B"},
                                {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"}
                                ]},
    {"name": "Object 3",
     "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"},
                                {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"},
                                {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}
                                ]},
    {"name": "Object 4",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                                {"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                                {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                                {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "B"},
                                {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}
                                ]}]



class LocationNetworkFactoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registerAllAdvancedRegions()

    def _setLearning(self, network, learn):
        for _, region in network.getRegions():
            region_type = region.getType()
            if region_type == "py.ColumnPoolerRegion":
                region.setParameterBool("learningMode", learn)
            elif region_type == "py.ApicalTMPairRegion":
                region.setParameterBool("learn", learn)
            elif region_type == "py.GridCellLocationRegion":
                region.setParameterBool("learningMode", learn)



    def testCreateL4L6aLocationColumn(self):
        """
        Test 'createL4L6aLocationColumn' by inferring a set of hand crafted objects
        """
        scale = []
        orientation = []
        # Initialize L6a location region with 5 modules varying scale by sqrt(2) and
        # 4 different random orientations for each scale
        for i in range(5):
            for _ in range(4):
                angle = np.radians(random.gauss(7.5, 7.5))
                orientation.append(random.choice([angle, -angle]))
                scale.append(10.0 * (math.sqrt(2) ** i))

        net = Network()
        createL4L6aLocationColumn(net, L4Params={
                "columnCount": NUM_OF_COLUMNS,
                "cellsPerColumn": CELLS_PER_COLUMN,
                "activationThreshold": 15,
                "minThreshold": 15,
                "initialPermanence": 1.0,
                "implementation": "ApicalTiebreak",
                "maxSynapsesPerSegment": -1
            },
            L6aParams={
                "moduleCount": len(scale),
                "scale": scale,
                "orientation": orientation,
                "anchorInputSize": NUM_OF_CELLS,
                "activationThreshold": 8,
                "initialPermanence": 1.0,
                "connectedPermanence": 0.5,
                "learningThreshold": 8,
                "sampleSize": 10,
                "permanenceIncrement": 0.1,
                "permanenceDecrement": 0.0,
                "bumpOverlapMethod": "probabilistic"
            },
            inverseReadoutResolution=8
        )
        net.initialize()

        L6a = net.getRegion('L6a')
        sensor = net.getRegion('sensorInput')
        motor = net.getRegion('motorInput')

        # Keeps a list of learned objects
        learnedRepresentations = defaultdict(list)

        # Learn Objects
        self._setLearning(net, True)

        for objectDescription in OBJECTS:
            reset = True
            previousLocation = None
            L6a.executeCommand("activateRandomLocation")

            for iFeature, feature in enumerate(objectDescription["features"]):
                # Move the sensor to the center of the object
                locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])

                # Calculate displacement from previous location
                if previousLocation is not None:
                    motor.executeCommand('addDataToQueue', list(locationOnObject - previousLocation))
                else:
                    motor.executeCommand('addDataToQueue', [0, 0])
                previousLocation = locationOnObject

                # Sense feature at location
                sensor.executeCommand('addDataToQueue', FEATURE_ACTIVE_COLUMNS[feature["name"]], reset, 0)
                net.run(1)
                reset = False

                # Save learned representations
                representation = L6a.getOutputArray("sensoryAssociatedCells")
                representation = np.array(representation).nonzero()[0]
                learnedRepresentations[(objectDescription["name"], iFeature)] = representation

        # Infer objects
        self._setLearning(net, False)

        for objectDescription in OBJECTS:
            reset = True
            previousLocation = None
            inferred = False

            features = objectDescription["features"]
            touchSequence = list(range(len(features)))
            random.shuffle(touchSequence)

            for iFeature in touchSequence:
                feature = features[iFeature]

                # Move the sensor to the center of the object
                locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])

                # Calculate displacement from previous location
                if previousLocation is not None:
                    motor.executeCommand('addDataToQueue', locationOnObject - previousLocation)
                else:
                    motor.executeCommand('addDataToQueue', [0, 0])
                previousLocation = locationOnObject

                # Sense feature at location
                sensor.executeCommand('addDataToQueue', FEATURE_ACTIVE_COLUMNS[feature["name"]], reset, 0)
                net.run(1)
                reset = False

                representation = L6a.getOutputArray("sensoryAssociatedCells")
                representation = np.array(representation).nonzero()[0]
                target_representations = set(
                    learnedRepresentations[(objectDescription["name"], iFeature)])

                inferred = (set(representation) <= target_representations)
                if inferred:
                    break

            self.assertTrue(inferred)
