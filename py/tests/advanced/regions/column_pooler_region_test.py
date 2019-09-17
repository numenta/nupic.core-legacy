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
import unittest

from htm.bindings.engine_internal import Network
from htm.advanced.support.register_regions import registerAllAdvancedRegions


class ColumnPoolerRegionTest(unittest.TestCase):
    """ Super simple test of the ColumnPooler region."""

    @classmethod
    def setUpClass(cls):
        registerAllAdvancedRegions()


    def testNetworkCreate(self):
        """Create a simple network to test the region."""

        rawParams = {"outputWidth": 8*2048}
        net = Network()
        rawSensor = net.addRegion("raw","py.RawSensor", json.dumps(rawParams))
        l2c = net.addRegion("L2", "py.ColumnPoolerRegion", "")
        net.link("raw", "L2", "UniformLink", "")

        self.assertEqual(rawSensor.getParameterUInt32("outputWidth"), l2c.getParameterUInt32("inputWidth"), "Incorrect outputWidth parameter")

        rawSensor.executeCommand('addDataToQueue', [2, 4, 6], 0, 42)
        rawSensor.executeCommand('addDataToQueue', [2, 42, 1023], 1, 43)
        rawSensor.executeCommand('addDataToQueue', [18, 19, 20], 0, 44)

        # Run the network and check outputs are as expected
        net.run(3)


    def testOverlap(self):
        """Create a simple network to test the region."""

        rawParams = {"outputWidth": 8 * 2048}
        net = Network()
        rawSensor = net.addRegion("raw", "py.RawSensor", json.dumps(rawParams))
        l2c = net.addRegion("L2", "py.ColumnPoolerRegion", "")
        net.link("raw", "L2", "UniformLink", "")

        self.assertEqual(rawSensor.getParameterUInt32("outputWidth"), l2c.getParameterUInt32("inputWidth"), "Incorrect outputWidth parameter")

        rawSensor.executeCommand('addDataToQueue', [2, 4, 6], 0, 42)
        rawSensor.executeCommand('addDataToQueue', [2, 42, 1023], 1, 43)
        rawSensor.executeCommand('addDataToQueue', [18, 19, 20], 0, 44)

        # Run the network and check outputs are as expected
        net.run(3)


if __name__ == "__main__":
    unittest.main()

