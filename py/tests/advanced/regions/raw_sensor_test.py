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

from htm.advanced.support.register_regions import registerAllAdvancedRegions
import numpy as np

class SensorValuesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        registerAllAdvancedRegions()

    def testExecuteCommandListWithReset(self):
        """
        Test that execute command executes the correct command with a list and reset set.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', [0, 1], 1, 0)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([True], output)
        
    def testExecuteCommandListWithNoReset(self):
        """
        Test that execute command executes the correct command with a list with reset 0.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', [0, 1], 0, 0)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testExecuteCommandList(self):
        """
        Test that execute command executes the correct command with a list.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', [0, 1], 0, 0)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testSequence(self):
        """
        Test that RawSensor executes correctly.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', [0, 1], 1, 0)
        sensor.executeCommand('addDataToQueue', [1, 2], 0, 1)
        sensor.executeCommand('addDataToQueue', [2, 3], 0, 2)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([True], output)
        output = list(np.array(sensor.getOutputArray("sequenceIdOut")))     
        self.assertEqual([0], output)

        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([0, 1, 1, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        output = list(np.array(sensor.getOutputArray("sequenceIdOut")))     
        self.assertEqual([1], output)
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0, 1, 1, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        output = list(np.array(sensor.getOutputArray("sequenceIdOut")))     
        self.assertEqual([2], output)
        
    def testQueue(self):
        """
        Test that RawSensor executes correctly.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', [0, 1], 1, 0)
        sensor.executeCommand('addDataToQueue', [1, 2], 0, 0)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([True], output)

        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([0, 1, 1, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testEmptyQueue(self):
        """
        Test that RawSensor detects empty queue.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        
        try:
            net.run(1)        
            output = list(np.array(sensor.getOutputArray("dataOut")))     
            self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
            self.fail("Empty queue should throw exception")
        except:
            pass
        
    def testExecuteCommandArray(self):
        """
        Test that execute command executes the correct command with a list.
        """
        net = Network()
    
        # Create simple region to pass sensor commands as displacement vectors (dx, dy)
        sensor = net.addRegion("sensor", "py.RawSensor", json.dumps({"outputWidth": 8}))
        sensor.executeCommand('addDataToQueue', np.array([0, 1]), 0, 0)
        
        net.run(1)
        
        output = list(np.array(sensor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1, 0, 0, 0, 0, 0, 0], output)
        output = list(np.array(sensor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
