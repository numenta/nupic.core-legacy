# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import unittest

import numpy as np
import numpy.testing
from nupic.bindings.regions.PyRegion import PyRegion
from nupic.engine import Network

TEST_DATA_SPARSE = np.array([4, 7])
MAX_ACTIVE = TEST_DATA_SPARSE.size
OUTPUT_WIDTH = 10
TEST_DATA_DENSE = np.zeros(OUTPUT_WIDTH, dtype=np.bool)
TEST_DATA_DENSE[TEST_DATA_SPARSE] = True


class SparseRegion(PyRegion):
  """
  This region receives sparse input and returns the same sparse output
  
  :param maxActive: Max active bits in the sparse data
  :param outputWidth: Size of output vector
  """

  def __init__(self, maxActive, outputWidth, **kwargs):
    PyRegion.__init__(self, **kwargs)

    self.maxActive = maxActive
    self.outputWidth = outputWidth

  @classmethod
  def getSpec(cls):
    return {
      "description": "Sparse Region",
      "singleNodeOnly": True,
      "inputs": {
        "dataIn": {
          "description": "Sparse Data In",
          "dataType": "UInt32",
          "isDefaultInput": True,
          "required": False,
          "sparse": True,
          "count": 0
        },
      },
      "outputs": {
        "dataOut": {
          "description": "Sparse Data Out",
          "dataType": "UInt32",
          "isDefaultOutput": True,
          "sparse": True,
          "count": 0
        },
      },
      "parameters": {
        "maxActive": {
          "description": "Max active bits in the sparse data",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "constraints": "",
        },
        "outputWidth": {
          "description": "Size of output vector",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "constraints": "",
        }
      }
    }

  def compute(self, inputs, outputs):
    if "dataIn" in inputs:
      PyRegion.setSparseOutput(outputs, "dataOut", inputs["dataIn"])
    else:
      PyRegion.setSparseOutput(outputs, "dataOut", self.data)

  def initialize(self):
    self.data = TEST_DATA_SPARSE

  def getOutputElementCount(self, name):
    return self.outputWidth


class DenseRegion(PyRegion):
  """
  This region receives dense input and returns the same dense output

  :param maxActive: Max active bits in the sparse data
  :param outputWidth: Size of output vector
  """

  def __init__(self, maxActive, outputWidth, **kwargs):
    PyRegion.__init__(self, **kwargs)

    self.maxActive = maxActive
    self.outputWidth = outputWidth

  @classmethod
  def getSpec(cls):
    return {
      "description": "Dense Region",
      "singleNodeOnly": True,
      "inputs": {
        "dataIn": {
          "description": "Dense Data In",
          "dataType": "Bool",
          "isDefaultInput": True,
          "required": False,
          "count": 0
        },
      },
      "outputs": {
        "dataOut": {
          "description": "Dense Data Out",
          "dataType": "Bool",
          "isDefaultOutput": True,
          "count": 0
        },
      },
      "parameters": {
        "maxActive": {
          "description": "Max active bits in the sparse data",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "constraints": "",
        },
        "outputWidth": {
          "description": "Size of output vector",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "constraints": "",
        }
      }
    }

  def compute(self, inputs, outputs):
    if "dataIn" in inputs:
      outputs["dataOut"][:] = inputs["dataIn"]
    else:
      outputs["dataOut"][:] = self.data

  def initialize(self):
    self.data = TEST_DATA_DENSE

  def getOutputElementCount(self, name):
    return self.outputWidth


def createNetwork(fromRegion, toRegion):
  """Create test network"""
  network = Network()
  config = str({"maxActive": MAX_ACTIVE, "outputWidth": OUTPUT_WIDTH})
  network.addRegion("from", fromRegion, config)
  network.addRegion("to", toRegion, config)

  network.link("from", "to", "UniformLink", "")
  return network


class SparseLinkTest(unittest.TestCase):
  """Test sparse link"""
  __name__ = "SparseLinkTest"

  def setUp(self):
    """Register test regions"""
    Network.registerPyRegion(SparseRegion.__module__, SparseRegion.__name__)
    Network.registerPyRegion(DenseRegion.__module__, DenseRegion.__name__)

  def testSparseToSparse(self):
    """Test links between sparse to sparse"""
    net = createNetwork("py.SparseRegion", "py.SparseRegion")
    net.initialize()
    net.run(1)

    actual = net.regions["to"].getOutputData("dataOut")
    np.testing.assert_array_equal(actual, TEST_DATA_SPARSE)

  def testSparseToDense(self):
    """Test links between sparse to dense"""
    net = createNetwork("py.SparseRegion", "py.DenseRegion")
    net.initialize()
    net.run(1)

    actual = net.regions["to"].getOutputData("dataOut")
    np.testing.assert_array_equal(actual, TEST_DATA_DENSE)

  def testDenseToSparse(self):
    """Test links between dense to sparse"""
    net = createNetwork("py.DenseRegion", "py.SparseRegion")
    net.initialize()
    net.run(1)

    actual = net.regions["to"].getOutputData("dataOut")
    np.testing.assert_array_equal(actual, TEST_DATA_SPARSE)

  def testDenseToDense(self):
    """Test links between dense to dense"""
    net = createNetwork("py.DenseRegion", "py.DenseRegion")
    net.initialize()
    net.run(1)

    actual = net.regions["to"].getOutputData("dataOut")
    np.testing.assert_array_equal(actual, TEST_DATA_DENSE)


if __name__ == '__main__':
  unittest.main()
