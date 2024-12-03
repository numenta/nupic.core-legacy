# Copyright 2017 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import json
import unittest
import pytest

try:
  # NOTE need to import capnp first to activate the magic necessary for
  # PythonDummyRegion_capnp, etc.
  import capnp
except ImportError:
  capnp = None
else:
  from nupic.proto.NetworkProto_capnp import NetworkProto

from nupic.bindings.regions.PyRegion import PyRegion

import nupic.bindings.engine_internal as engine
from nupic.bindings.tools.serialization_test_py_region import \
     SerializationTestPyRegion


class TestLinks(PyRegion):
  """
  Test region used to test link validation
  """
  def __init__(self): pass
  def initialize(self): pass
  def compute(self): pass
  def getOutputElementCount(self): pass
  @classmethod
  def getSpec(cls):
    return {
      "description": TestLinks.__doc__,
      "singleNodeOnly": True,
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
    engine.Network.registerPyRegion(TestLinks.__module__, TestLinks.__name__)


  @unittest.skipUnless(
    capnp, "pycapnp is not installed, skipping serialization test.")
  def testCapnpSerializationWithPyRegion(self):
    """Test capnp (de)serialization of network containing a python region"""
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
      builderProto = NetworkProto.new_message()
      srcNet.write(builderProto)

      # Construct NetworkProto reader from populated builder
      readerProto = NetworkProto.from_bytes(builderProto.to_bytes())

      # Deserialize
      destNet = engine.Network.read(readerProto)

      destRegion = destNet.getRegions().getByName(
        SerializationTestPyRegion.__name__)

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
    region1 = network.getRegions().getByName("region1")
    region1.setDimensions(engine.Dimensions([1, 1]))

    # Link region1 and region2
    network.link("region1", "region2", "UniformLink", "")

    # Initialize network
    network.initialize()

    for linkName, link in network.getLinks():
      # Compare Link API to what we know about the network
      self.assertEqual(link.toString(), linkName)
      self.assertEqual(link.getDestRegionName(), "region2")
      self.assertEqual(link.getSrcRegionName(), "region1")
      self.assertEqual(link.getLinkType(), "UniformLink")
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
    network.addRegion("from", "py.TestLinks", "")
    network.addRegion("to", "py.TestLinks", "")

    # Check for valid links
    network.link("from", "to", "UniformLink", "", "UInt32", "UInt32")
    network.link("from", "to", "UniformLink", "", "Real32", "Real32")

    # Check for invalid links
    with pytest.raises(RuntimeError):
      network.link("from", "to", "UniformLink", "", "Real32", "UInt32")
    with pytest.raises(RuntimeError):
      network.link("from", "to", "UniformLink", "", "UInt32", "Real32")

  def testParameters(self):

    n = engine.Network()
    l1 = n.addRegion("l1", "TestNode", "")
    scalars = [
      ("int32Param", l1.getParameterInt32, l1.setParameterInt32, 32, int, 35),
      ("uint32Param", l1.getParameterUInt32, l1.setParameterUInt32, 33, int, 36),
      ("int64Param", l1.getParameterInt64, l1.setParameterInt64, 64, long, 74),
      ("uint64Param", l1.getParameterUInt64, l1.setParameterUInt64, 65, long, 75),
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
