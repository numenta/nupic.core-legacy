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
import json
import math
import random
import unittest
from collections import defaultdict

import numpy as np
from htm.bindings.engine_internal import Network

from htm.advanced.frameworks.location.path_integration_union_narrowing import computeRatModuleParametersFromReadoutResolution
from htm.advanced.support.register_regions import registerAllAdvancedRegions

NUM_OF_COLUMNS = 150
CELLS_PER_COLUMN = 16
NUM_OF_CELLS = NUM_OF_COLUMNS * CELLS_PER_COLUMN

FEATURE_ACTIVE_COLUMNS = {
    "A": np.random.choice(NUM_OF_COLUMNS, NUM_OF_COLUMNS // 10).tolist(),
    "B": np.random.choice(NUM_OF_COLUMNS, NUM_OF_COLUMNS // 10).tolist(),
    }

FEATURE_CANDIDATE_SDR = {
    "A": np.random.choice(NUM_OF_CELLS, NUM_OF_CELLS // 10).tolist(),
    "B": np.random.choice(NUM_OF_CELLS, NUM_OF_CELLS // 10).tolist(),
    }

OBJECTS = [
    {"name": "Object 1",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                    {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"},
                    {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"}]},
    {"name": "Object 2",
     "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 0, "width": 10, "height": 10, "name": "B"},
                    {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                    {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                    {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"}]},
    {"name": "Object 3",
     "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                    {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"},
                    {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                    {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"},
                    {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}]},
    {"name": "Object 4",
     "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                    {"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                    {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                    {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                    {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                    {"top": 20, "left": 10, "width": 10, "height": 10, "name": "B"},
                    {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}]},
    ]


def _createNetwork(inverseReadoutResolution, anchorInputSize, dualPhase=False):
    """
    Create a simple network connecting sensor and motor inputs to the location
    region. Use :meth:`RawSensor.addDataToQueue` to add sensor input and growth
    candidates. Use :meth:`RawValues.addDataToQueue` to add motor input.
    ::
                        +----------+
    [   sensor*   ] --> |          | --> [     activeCells        ]
    [ candidates* ] --> | location | --> [    learnableCells      ]
    [    motor    ] --> |          | --> [ sensoryAssociatedCells ]
                        +----------+

    :param inverseReadoutResolution:
        Specifies the diameter of the circle of phases in the rhombus encoded by a
        bump.
    :type inverseReadoutResolution: int

    :type anchorInputSize: int
    :param anchorInputSize:
        The number of input bits in the anchor input.

    .. note::
        (*) This function will only add the 'sensor' and 'candidates' regions when
        'anchorInputSize' is greater than zero. This is useful if you would like to
        compute locations ignoring sensor input

    .. seealso::
         - :py:func:`htmresearch.frameworks.location.path_integration_union_narrowing.createRatModuleFromReadoutResolution`

    """
    net = Network()

    # Create simple region to pass motor commands as displacement vectors (dx, dy)
    net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))

    if anchorInputSize > 0:
        # Create simple region to pass growth candidates
        net.addRegion("candidates", "py.RawSensor", json.dumps({"outputWidth": anchorInputSize}))

        # Create simple region to pass sensor input
        net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": anchorInputSize}))

    # Initialize region with 5 modules varying scale by sqrt(2) and 4 different
    # random orientations for each scale
    scale = []
    orientation = []
    for i in range(5):
        for _ in range(4):
            angle = np.radians(random.gauss(7.5, 7.5))
            orientation.append(random.choice([angle, -angle]))
            scale.append(10.0 * (math.sqrt(2) ** i))

    # Create location region
    params = computeRatModuleParametersFromReadoutResolution(inverseReadoutResolution)
    params.update({
        "moduleCount": len(scale),
        "scale": scale,
        "orientation": orientation,
        "anchorInputSize": anchorInputSize,
        "activationThreshold": 8,
        "initialPermanence": 1.0,
        "connectedPermanence": 0.5,
        "learningThreshold": 8,
        "sampleSize": 10,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.0,
        "dualPhase": dualPhase,
        "bumpOverlapMethod": "probabilistic"
    })
    net.addRegion("location", "py.GridCellLocationRegion", json.dumps(params))

    if anchorInputSize > 0:
        # Link sensor
        net.link("sensor", "location", "UniformLink", "", srcOutput="dataOut", destInput="anchorInput")
        net.link("sensor", "location", "UniformLink", "", srcOutput="resetOut", destInput="resetIn")
        net.link("candidates", "location", "UniformLink", "", srcOutput="dataOut", destInput="anchorGrowthCandidates")

    # Link motor input
    net.link("motor", "location", "UniformLink", "", srcOutput="dataOut", destInput="displacement")

    # Initialize network objects
    net.initialize()

    return net



class GridCellLocationRegionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registerAllAdvancedRegions()

    def testPathIntegration(self):
        """
        Test the region path integration properties by moving the sensor around
        verifying the location representations are different when the sensor moves
        and matches the initial representation when returning to the start location
        """
        # Create location only network
        net = _createNetwork(anchorInputSize=0, inverseReadoutResolution=8)
        location = net.getRegion('location')
        motor = net.getRegion('motor')

        # Start from a random location
        location.executeCommand("activateRandomLocation")
        motor.executeCommand('addDataToQueue', [0, 0])
        net.run(1)
        start = np.array(location.getOutputArray("activeCells")).nonzero()[0]

        # Move up 10
        motor.executeCommand('addDataToQueue', [0, 10])
        net.run(1)

        # mid location should not match the start location
        mid = np.array(location.getOutputArray("activeCells")).nonzero()[0]
        try:
            # If path integration is working they should not be equal
            np.testing.assert_array_equal(start, mid)
        except:
            # Not equal because we moved to a different position
            pass

        # Move down 10 in two steps ending at the initial location
        motor.executeCommand('addDataToQueue', [0, -5])
        motor.executeCommand('addDataToQueue', [0, -5])
        net.run(2)

        # end location should match the start location
        end = np.array(location.getOutputArray("activeCells")).nonzero()[0]
        np.testing.assert_equal(start, end)

    def testLearning(self):
        """
        Test ability to learn objects based on their features and locations.
        """
        # Create network
        net = _createNetwork(anchorInputSize=NUM_OF_CELLS, inverseReadoutResolution=8)
        location = net.getRegion('location')
        motor = net.getRegion('motor')
        sensor = net.getRegion('sensor')
        candidates = net.getRegion('candidates')

        # Keeps a list of learned objects
        learnedRepresentations = defaultdict(list)

        # Learn Objects
        location.setParameterBool("learningMode", True)

        for objectDescription in OBJECTS:
            reset = True
            previousLocation = None
            location.executeCommand("activateRandomLocation")

            for iFeature, feature in enumerate(objectDescription["features"]):
                featureName = feature["name"]

                # Move the sensor to the center of the object
                locationOnObject = np.array([feature["top"] + feature["height"] / 2., feature["left"] + feature["width"] / 2.])
                # Calculate displacement from previous location
                if previousLocation is not None:
                    motor.executeCommand('addDataToQueue', list(locationOnObject - previousLocation))
                else:
                    motor.executeCommand('addDataToQueue', [0, 0])
                previousLocation = locationOnObject

                # Sense feature at location
                sensor.executeCommand('addDataToQueue', FEATURE_ACTIVE_COLUMNS[featureName], reset, 0)
                candidates.executeCommand('addDataToQueue', FEATURE_CANDIDATE_SDR[featureName], reset, 0)
                net.run(1)
                reset = False

                # Save learned representation
                representation = np.array(location.getOutputArray("sensoryAssociatedCells"))
                representation = representation.nonzero()[0]
                learnedRepresentations[(objectDescription["name"], iFeature)] = representation

        # Infer learned objects
        location.setParameterBool("learningMode", False)

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
                sensor.executeCommand('addDataToQueue', FEATURE_ACTIVE_COLUMNS[featureName], reset, 0)
                candidates.executeCommand('addDataToQueue', FEATURE_CANDIDATE_SDR[featureName], reset, 0)
                net.run(1)

                representation = np.array(location.getOutputArray("sensoryAssociatedCells"))
                representation = set(representation.nonzero()[0])
                target_representation = set(learnedRepresentations[(objectDescription["name"], iFeature)])

                inferred = (representation <= target_representation)
                if inferred:
                    break

            self.assertTrue(inferred)
