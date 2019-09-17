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
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

import json
import unittest
import pytest
import numpy as np

from htm.bindings.regions.PyRegion import PyRegion
from htm.bindings.sdr import SDR
import htm.bindings.engine_internal as engine
from htm.bindings.tools.serialization_test_py_region import \
     SerializationTestPyRegion

TEST_DATA = [0,1,2,3,4]
EXPECTED_RESULT1 = [  4, 5 ]
EXPECTED_RESULT2 = [  4,  11,  28,  42,  43,  87,  89,  93, 110, 127, 132, 137, 149, 187, 193]
EXPECTED_RESULT3 = [ 134, 371, 924, 1358, 1386, 2791, 2876, 2996, 3526, 4089, 4242, 4406, 4778, 5994, 6199]


class LinkRegion(PyRegion):
  """
  Test region used to test link validation
  """
  def __init__(self): pass
  def initialize(self): pass
  def compute(self, inputs, outputs): 
    # This will pass its inuts on to the outputs.
    for key in inputs:
      outputs[key][:] = inputs[key]
      
    
  def getOutputElementCount(self, name): 
    return 5
    
  @classmethod
  def getSpec(cls):
    return {
      "description": LinkRegion.__doc__,
      "inputs": {
        "UInt32": {
          "description": "UInt32 Data",
          "dataType": "UInt32",
          "isDefaultInput": True,
          "required": False,
          "count": 0
        },
        "Real32": {
          "description": "Real32 Data",
          "dataType": "Real32",
          "isDefaultInput": False,
          "required": False,
          "count": 0
        },
      },
      "outputs": {
        "UInt32": {
          "description": "UInt32 Data",
          "dataType": "UInt32",
          "isDefaultOutput": True,
          "required": False,
          "count": 0
        },
        "Real32": {
          "description": "Real32 Data",
          "dataType": "Real32",
          "isDefaultOutput": False,
          "required": False,
          "count": 0
        },
      },
      "parameters": { }
    }

class NetworkTest(unittest.TestCase):

  def setUp(self):
    """Register test region"""
    engine.Network.cleanup()
    engine.Network.registerPyRegion(LinkRegion.__module__, LinkRegion.__name__)

  @pytest.mark.skip(reason="pickle support needs work...another PR")
  def testSerializationWithPyRegion(self):
    """Test  (de)serialization of network containing a python region"""
    engine.Network.registerPyRegion(__name__,
                                    SerializationTestPyRegion.__name__)
    try:
      srcNet = engine.Network()
      srcNet.addRegion(SerializationTestPyRegion.__name__,
                       "py." + SerializationTestPyRegion.__name__,
                       json.dumps({
                         "dataWidth": 128,
                         "randomSeed": 99,
                       }))

      # Serialize
      srcNet.saveToFile("SerializationTest.stream")


      # Deserialize
      destNet = engine.Network()
      destNet.loadFromFile("SerializationTest.stream")

      destRegion = destNet.getRegion(SerializationTestPyRegion.__name__)

      self.assertEqual(destRegion.getParameterUInt32("dataWidth"), 128)
      self.assertEqual(destRegion.getParameterUInt32("randomSeed"), 99)

    finally:
      engine.Network.unregisterPyRegion(SerializationTestPyRegion.__name__)


  def testSimpleTwoRegionNetworkIntrospection(self):
    # Create Network instance
    network = engine.Network()

    # Add two TestNode regions to network
    network.addRegion("region1", "TestNode", "")
    network.addRegion("region2", "TestNode", "")

    # Set dimensions on first region
    region1 = network.getRegion("region1")
    region1.setDimensions(engine.Dimensions([1, 1]))

    # Link region1 and region2
    network.link("region1", "region2")

    # Initialize network
    network.initialize()

    for link in network.getLinks():
      # Compare Link API to what we know about the network
      self.assertEqual(link.getDestRegionName(), "region2")
      self.assertEqual(link.getSrcRegionName(), "region1")
      self.assertEqual(link.getDestInputName(), "bottomUpIn")
      self.assertEqual(link.getSrcOutputName(), "bottomUpOut")
      break
    else:
      self.fail("Unable to iterate network links.")


  def testNetworkLinkTypeValidation(self):
    """
    This tests whether the links source and destination dtypes match
    """
    network = engine.Network()
    r_from = network.addRegion("from", "py.LinkRegion", "")
    r_to = network.addRegion("to", "py.LinkRegion", "")
    cnt = r_from.getOutputElementCount("UInt32")
    self.assertEqual(5, cnt)


    # Check for valid links
    network.link("from", "to", "", "", "UInt32", "UInt32")
    network.link("from", "to", "", "", "Real32", "Real32")
    network.link("from", "to", "", "", "Real32", "UInt32")
    network.link("from", "to", "", "", "UInt32", "Real32")
	

  @pytest.mark.skip(reason="parameter types don't match.")
  def testParameters(self):

    n = engine.Network()
    l1 = n.addRegion("l1", "TestNode", "")
    scalars = [
      ("int32Param", l1.getParameterInt32, l1.setParameterInt32, 32, int, 35),
      ("uint32Param", l1.getParameterUInt32, l1.setParameterUInt32, 33, int, 36),
      ("int64Param", l1.getParameterInt64, l1.setParameterInt64, 64, int, 74),
      ("uint64Param", l1.getParameterUInt64, l1.setParameterUInt64, 65, int, 75),
      ("real32Param", l1.getParameterReal32, l1.setParameterReal32, 32.1, float, 33.1),
      ("real64Param", l1.getParameterReal64, l1.setParameterReal64, 64.1, float, 65.1),
      ("stringParam", l1.getParameterString, l1.setParameterString, "nodespec value", str, "new value")]

    for paramName, paramGetFunc, paramSetFunc, initval, paramtype, newval in scalars:
      # Check the initial value for each parameter.
      x = paramGetFunc(paramName)
      self.assertEqual(type(x), paramtype, paramName)
      if initval is None:
        continue
      if type(x) == float:
        self.assertTrue(abs(x - initval) < 0.00001, paramName)
      else:
        self.assertEqual(x, initval, paramName)

      # Now set the value, and check to make sure the value is updated
      paramSetFunc(paramName, newval)
      x = paramGetFunc(paramName)
      self.assertEqual(type(x), paramtype)
      if type(x) == float:
        self.assertTrue(abs(x - newval) < 0.00001)
      else:
        self.assertEqual(x, newval)
        
  def testParameterArray(self):
    """
    Tests the setParameterArray( ) and getParameterArray( )
    The TestNode contains 'int64ArrayParam' and 'real32ArrayParam' parameters which are vectors.
    This test will write to each and read from each.
    """
    
    network = engine.Network()
    r1 = network.addRegion("region1", "TestNode", "")
    
    orig = np.array([1,2,3,4, 5,6,7,8], dtype=np.int64)
    r1.setParameterArray("int64ArrayParam", engine.Array(orig, True))
    self.assertEqual(r1.getParameterArrayCount("int64ArrayParam"), 8)
    a = engine.Array()
    r1.getParameterArray("int64ArrayParam", a)
    result = np.array(a)
    self.assertTrue( np.array_equal(orig, result))
    
    orig = np.array([1,2,3,4, 5,6,7,8], dtype=np.float32)
    r1.setParameterArray("real32ArrayParam", engine.Array(orig, True))
    self.assertEqual(r1.getParameterArrayCount("real32ArrayParam"), 8)
    a = engine.Array()
    r1.getParameterArray("real32ArrayParam", a)
    result = np.array(a)
    self.assertTrue( np.array_equal(orig, result))
    
  def testGetInputArray(self):
    """
    This tests whether the input to r_to is accessible and matches the output from the r_from region
    """
    engine.Network.registerPyRegion(LinkRegion.__module__, LinkRegion.__name__)
    
    network = engine.Network()
    r_from = network.addRegion("from", "py.LinkRegion", "")
    r_to = network.addRegion("to", "py.LinkRegion", "")
    network.link("from", "to", "", "", "UInt32", "UInt32")
    network.initialize()
    
    # Populate the input data
    r_from.setInputArray("UInt32", np.array(TEST_DATA))
        
    network.run(1)
    
    to_input = np.array(r_to.getInputArray("UInt32"))
    self.assertTrue(np.array_equal(to_input, TEST_DATA))

  def testGetOutputArray(self):
    """
    This tests whether the final output of the network is accessible
    """
    engine.Network.registerPyRegion(LinkRegion.__module__, LinkRegion.__name__)
    
    network = engine.Network()
    r_from = network.addRegion("from", "py.LinkRegion", "")
    r_to = network.addRegion("to", "py.LinkRegion", "")
    network.link("from", "to", "", "", "UInt32", "UInt32")
    network.initialize()
    r_from.setInputArray("UInt32", np.array(TEST_DATA))
        
    network.run(1)
    
    output = r_to.getOutputArray("UInt32")
    self.assertTrue(np.array_equal(output, TEST_DATA))

    

  def testBuiltInRegions(self):
    """
    This sets up a network with built-in regions.
    """
    
    net = engine.Network()
    net.setLogLevel(engine.Verbose)     # Verbose shows data inputs and outputs while executing.
    
    encoder = net.addRegion("encoder", "ScalarSensor", "{n: 6, w: 2}");
    sp = net.addRegion("sp", "SPRegion", "{columnCount: 200}");
    tm = net.addRegion("tm", "TMRegion", "");
    net.link("encoder", "sp"); 
    net.link("sp", "tm"); 
    net.initialize();

    encoder.setParameterReal64("sensedValue", 0.8);  #Note: default range setting is -1.0 to +1.0
    net.run(1)
    
    sp_input = sp.getInputArray("bottomUpIn")
    sdr = sp_input.getSDR()
    self.assertTrue(np.array_equal(sdr.sparse, EXPECTED_RESULT1))
    
    sp_output = sp.getOutputArray("bottomUpOut")
    sdr = sp_output.getSDR()
    self.assertTrue(np.array_equal(sdr.sparse, EXPECTED_RESULT2))
    
    tm_output = tm.getOutputArray("predictedActiveCells")
    sdr = tm_output.getSDR()
    self.assertTrue(np.array_equal(sdr.sparse, EXPECTED_RESULT3))

  def testExecuteCommand(self):
    """
    Check to confirm that the ExecuteCommand( ) funtion works.
    """
    net = engine.Network()
    r = net.addRegion("test", "TestNode", "")
    
    lst = ["list arg", 86]
    result = r.executeCommand("HelloWorld", 42, lst)
    self.assertTrue(result == "Hello World says: arg1=42 arg2=['list arg', 86]")
    

