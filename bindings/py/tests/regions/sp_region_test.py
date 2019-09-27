# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Frederick C. Rotbart 
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
import unittest
import json

from htm.bindings.engine_internal import Network

spParams =  {
            "columnCount": 1024,
            "potentialRadius": 100,
            "potentialPct": 0.9,
            "globalInhibition": True,               
            "localAreaDensity": 0.06,
            "stimulusThreshold": 1,
            "synPermInactiveDec": 0.0,
            "synPermActiveInc": 0.0,
            "synPermConnected": 0.2,
            "boostStrength": 10.0,
            "seed": 1956,
            "spVerbosity": 0}

class SPRegionTests(unittest.TestCase):

    def testSPRegionIsCreatable(self):
        """
        Test that the SPRegion can be created in Python.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sp = net.addRegion("sp", "SPRegion", json.dumps({}))
        
    def testSPRegionParametersAreWritable(self):
        """
        Test that the SPRegion parameters can be set.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sp = net.addRegion("sp", "SPRegion", json.dumps(spParams))
        
