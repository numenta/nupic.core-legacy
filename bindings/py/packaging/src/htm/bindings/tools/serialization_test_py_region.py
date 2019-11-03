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

"""Simple custom python region containing an extension-based Random instance
for use in network serialization test and cyclical serialization
performance tool.
"""

from htm.bindings.math import Random
from htm.bindings.regions.PyRegion import PyRegion



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
            "accessMode":"Create",
            "dataType":"UInt32",
            "count":1,
            "constraints":""},
          "randomSeed":{
            "description":"Seed for constructing the Random instance",
            "accessMode":"Create",
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
