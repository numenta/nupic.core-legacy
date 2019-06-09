# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, David McDougall
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

import unittest

from nupic.bindings.sdr import SDR
from nupic.algorithms import SpatialPooler as SP
import numpy as np

class SpatialPoolerTest(unittest.TestCase):

  def testCompute(self):
    """ Check that there are no errors in call to compute. """
    inputs = SDR( 100 ).randomize( .05 )
    active = SDR( 100 )
    sp = SP( inputs.dimensions, active.dimensions, stimulusThreshold = 1 )
    sp.compute( inputs, True, active )
    assert( active.getSum() > 0 )


  def _runGetPermanenceTrial(self, float_type):
    """ 
    Check that getPermanence() returns values for a given float_type. 
    These tests are sensitive to the data type. This because if you pass a 
    numpy array of the type matching the C++ argument then PyBind11 does not
    convert the data, and the C++ code can modify the data in-place
    If you pass the wrong data type, then PyBind11 does a conversion so 
    your C++ function only gets a converted copy of a numpy array, and any changes 
    are lost after returning
    """
    inputs = SDR( 100 ).randomize( .05 )
    active = SDR( 100 )
    sp = SP( inputs.dimensions, active.dimensions, stimulusThreshold = 1 )

    # Make sure that the perms start off zero.
    perms_in = np.zeros(sp.getNumInputs(), dtype=float_type)
    sp.setPermanence(0, perms_in)
    perms = np.zeros(sp.getNumInputs(), dtype=float_type)
    sp.getPermanence(0, perms)
    assert( perms.sum() == 0.0 )
    
    for i in range(10):
      sp.compute( inputs, True, active )
    
    # There should be at least one perm none zero
    total = np.zeros(sp.getNumInputs(), dtype=float_type)
    for i in range(100):
      perms = np.zeros(sp.getNumInputs(), dtype=float_type)
      sp.getPermanence(i, perms)
      total = total + perms
    assert( total.sum() > 0.0 )
    
  def testGetPermanenceFloat64(self):
    """ Check that getPermanence() returns values. """
    try:
      # This is when NTA_DOUBLE_PRECISION is true
      self._runGetPermanenceTrial(np.float64)
      
    except ValueError:
      # This has caught wrong precision error
      print("Successfully caught incorrect float numpy data length")
      pass     

  def testGetPermanenceFloat32(self):
    """ Check that getPermanence() returns values. """
    try:
      self._runGetPermanenceTrial(np.float32)
      
    except ValueError:
      print("Successfully caught incorrect float numpy data length")
      # This has correctly caught wrong precision error
      pass     

  def _runGetConnectedSynapses(self, uint_type):
    """ Check that getConnectedSynapses() returns values. """
    inputs = SDR( 100 ).randomize( .05 )
    active = SDR( 100 )
    sp = SP( inputs.dimensions, active.dimensions, stimulusThreshold = 1 )

    for i in range(10):
      sp.compute( inputs, True, active )
    
    # There should be at least one connected none zero
    total = np.zeros(sp.getNumInputs(), dtype=uint_type)
    for i in range(100):
      connected = np.zeros(sp.getNumInputs(), dtype=uint_type)
      sp.getConnectedSynapses(i, connected)
      total = total + connected
    assert( total.sum() > 0 )

  def testGetConnectedSynapsesUint64(self):
    """ Check that getConnectedSynapses() returns values. """
    try:
      # This is when NTA_DOUBLE_PRECISION is true
      self._runGetConnectedSynapses(np.uint64)
      
    except ValueError:
      # This has correctly caught wrong precision error
      print("Successfully caught incorrect uint numpy data length")
      pass     

  def testGetConnectedSynapsesUint32(self):
    """ Check that getConnectedSynapses() returns values. """
    try:
      # This is when NTA_DOUBLE_PRECISION is true
      self._runGetConnectedSynapses(np.uint32)
      
    except ValueError:
      # This has correctly caught wrong precision error
      print("Successfully caught incorrect uint numpy data length")
      pass     

  def _runGetConnectedCounts(self, uint_type):
    """ Check that getConnectedCounts() returns values. """
    inputs = SDR( 100 ).randomize( .05 )
    active = SDR( 100 )
    sp = SP( inputs.dimensions, active.dimensions, stimulusThreshold = 1 )

    for _ in range(10):
      sp.compute( inputs, True, active )
    
    # There should be at least one connected none zero
    connected = np.zeros(sp.getNumColumns(), dtype=uint_type)
    sp.getConnectedCounts(connected)
    assert( connected.sum() > 0 )

  def testGetConnectedCountsUint64(self):
    """ Check that getConnectedCounts() returns values. """
    try:
      # This is when NTA_DOUBLE_PRECISION is true
      self._runGetConnectedCounts(np.uint64)
      
    except ValueError:
      # This has correctly caught wrong precision error
      print("Successfully caught incorrect uint numpy data length")
      pass     

  def testGetConnectedCountsUint32(self):
    """ Check that getConnectedCounts() returns values. """
    try:
      # This is when NTA_DOUBLE_PRECISION is true
      self._runGetConnectedCounts(np.uint32)
      
    except ValueError:
      # This has correctly caught wrong precision error
      print("Successfully caught incorrect uint numpy data length")
      pass     


if __name__ == "__main__":
  unittest.main()
