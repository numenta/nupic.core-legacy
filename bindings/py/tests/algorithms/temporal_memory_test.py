# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2014-2015, Numenta, Inc.
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
import pytest
import pickle
import sys
import os

from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
import numpy as np


class TemporalMemoryBindingsTest(unittest.TestCase):

  def testCompute(self):
    """ Check that there are no errors in call to compute. """
    inputs = SDR( 100 ).randomize( .05 )
    
    tm = TM( inputs.dimensions)
    tm.compute( inputs, True )

    active = tm.getActiveCells()
    self.assertTrue( active.getSum() > 0 )

  @pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails for python2 with segmentation fault")
  def testNupicTemporalMemoryPickling(self):
    """Test pickling / unpickling of NuPIC TemporalMemory."""

    # Simple test: make sure that dumping / loading works...
    inputs = SDR( 100 ).randomize( .05 ) 
    tm = TM( inputs.dimensions)
    for _ in range(10):
      tm.compute( inputs, True)

    pickledTm = pickle.dumps(tm, 2)
    tm2 = pickle.loads(pickledTm)

    self.assertEqual(tm.numberOfCells(), tm2.numberOfCells(),
                     "Simple NuPIC TemporalMemory pickle/unpickle failed.")


  @pytest.mark.skip(reason="Fails with rapidjson internal assertion -- indicates a bad serialization")
  def testNupicTemporalMemorySavingToString(self):
    """Test writing to and reading from TemporalMemory."""
    inputs = SDR( 100 ).randomize( .05 ) 
    tm = TM( inputs.dimensions)
    for _ in range(10):
      tm.compute( inputs, True)

    # Simple test: make sure that writing/reading works...
    s = tm.writeToString()

    tm2 = TM()
    tm2.loadFromString(s)

    self.assertEqual(str(tm), str(tm),
                     "TemporalMemory write to/read from string failed.")

  def testNupicTemporalMemorySerialization(self):
     # Test serializing with each type of interface.
    inputs = SDR( 100 ).randomize( .05 ) 
    tm = TM( inputs.dimensions)
    for _ in range(10):
      tm.compute( inputs, True)
      
    #print(str(tm))
     
    # The TM now has some data in it, try serialization.  
    file = "temporalMemory_test_save2.bin"
    tm.saveToFile(file)
    tm3 = TM()
    tm3.loadFromFile(file)
    self.assertEqual(str(tm), str(tm3), "TemporalMemory serialization (using saveToFile/loadFromFile) failed.")
    os.remove(file)
