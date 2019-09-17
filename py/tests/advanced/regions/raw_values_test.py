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

class RawValuesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        registerAllAdvancedRegions()

    def testExecuteCommandListWithReset(self):
        """
        Test that execute command executes the correct command with a list and reset set.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        motor.executeCommand('addDataToQueue', [0, 0], 1)
        
        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([True], output)
        
    def testExecuteCommandListWithNoReset(self):
        """
        Test that execute command executes the correct command with a list with reset 0.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        motor.executeCommand('addDataToQueue', [0, 0], 0)
        
        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testExecuteCommandList(self):
        """
        Test that execute command executes the correct command with a list.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        motor.executeCommand('addDataToQueue', [0, 0])
        
        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testQueue(self):
        """
        Test that RawValues executes correctly.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        motor.executeCommand('addDataToQueue', [0, 0], 1)
        motor.executeCommand('addDataToQueue', [1, 1])
        
        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([True], output)

        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([1, 1], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
    def testEmptyQueue(self):
        """
        Test that RawValues detects empty queue.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        
        try:
            net.run(1)        
            output = list(np.array(motor.getOutputArray("dataOut")))     
            self.assertEqual([0, 0], output)
            self.fail("Empty queue should throw exception")
        except:
            pass
        
    def testExecuteCommandArray(self):
        """
        Test that execute command executes the correct command with a list.
        """
        net = Network()
    
        # Create simple region to pass motor commands as displacement vectors (dx, dy)
        motor = net.addRegion("motor", "py.RawValues", json.dumps({"outputWidth": 2}))
        motor.executeCommand('addDataToQueue', np.array([0, 0]))
        
        net.run(1)
        
        output = list(np.array(motor.getOutputArray("dataOut")))     
        self.assertEqual([0, 0], output)
        output = list(np.array(motor.getOutputArray("resetOut")))     
        self.assertEqual([False], output)
        
