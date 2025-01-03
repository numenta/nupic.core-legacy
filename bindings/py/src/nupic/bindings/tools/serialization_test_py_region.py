# Copyright 2017 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Simple custom python region containing an extension-based Random instance
for use in capnp network serialization test and cyclical serialization
performance tool.
"""

try:
  # NOTE need to import capnp first to activate the magic necessary for
  # PythonDummyRegion_capnp, etc.
  import capnp
except ImportError:
  capnp = None
else:
  from nupic.bindings.tools.SerializationTestPyRegionProto_capnp import \
       SerializationTestPyRegionProto


from nupic.bindings.math import Random
from nupic.bindings.regions.PyRegion import PyRegion



class SerializationTestPyRegion(PyRegion):
  """Custom python region for testing serialization/deserialization of network
  containing a python region that contains a nupic.bindings-based Random
  instance.
  """


  def __init__(self, dataWidth, randomSeed):
    if dataWidth <= 0:
      raise ValueError("Parameter dataWidth must be > 0")

    # Arbitrary value that's compatible with UInt32 in the proto schema
    # for testing serialization of python-native property
    self._dataWidth = dataWidth

    # For testing serialization of object implemented in the extension
    self._rand = Random(randomSeed)


  @property
  def dataWidth(self):
    return self._dataWidth


  @property
  def randomSeed(self):
    return self._rand.getSeed()


  @staticmethod
  def getSchema():
    """Return the pycapnp proto type that the class uses for serialization.

    This is used to convert the proto into the proper type before passing it
    into the read or write method of the subclass.
    """
    return SerializationTestPyRegionProto


  def writeToProto(self, proto):
    """Write state to proto object.

    The type of proto is determined by getSchema().
    """
    proto.dataWidth = self._dataWidth
    self._rand.write(proto.random)


  @classmethod
  def readFromProto(cls, proto):
    """Read state from proto object.

    The type of proto is determined by getSchema().

    :returns: Instance of SerializationTestPyRegion initialized from proto
    """
    obj = object.__new__(cls)
    obj._dataWidth = proto.dataWidth
    obj._rand = Random()
    obj._rand.read(proto.random)

    return obj


  def initialize(self, dims=None, splitterMaps=None):
    pass


  def compute(self, inputs, outputs):
    """
    Run one iteration of SerializationTestPyRegion's compute
    """
    outputs["out"][:] = inputs["in"]


  @classmethod
  def getSpec(cls):
    """Return the Spec for SerializationTestPyRegion.
    """
    spec = {
        "description":SerializationTestPyRegion.__doc__,
        "singleNodeOnly":True,
        "inputs":{
          "in":{
            "description":"The input vector.",
            "dataType":"Real32",
            "count":0,
            "required":True,
            "regionLevel":False,
            "isDefaultInput":True,
            "requireSplitterMap":False},
        },
        "outputs":{
          "out":{
            "description":"A copy of the input vector.",
            "dataType":"Real32",
            "count":0,
            "regionLevel":True,
            "isDefaultOutput":True},
        },

        "parameters":{
          "dataWidth":{
            "description":"Size of inputs",
            "accessMode":"Read",
            "dataType":"UInt32",
            "count":1,
            "constraints":""},
          "randomSeed":{
            "description":"Seed for constructing the Random instance",
            "accessMode":"Read",
            "dataType":"UInt32",
            "count":1,
            "constraints":""},
        },

    }

    return spec


  def getOutputElementCount(self, name):
    if name == "out":
      return self._dataWidth
    else:
      raise Exception("Unrecognized output: " + name)
