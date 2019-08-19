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
    tm = TM(columnDimensions=(16,))

    pickledTm = pickle.dumps(tm)
    tm2 = pickle.loads(pickledTm)

    self.assertEqual(tm.numberOfCells(), tm2.numberOfCells(),
                     "Simple NuPIC TemporalMemory pickle/unpickle failed.")
