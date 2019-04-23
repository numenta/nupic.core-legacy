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
from nupic.bindings.algorithms import SpatialPooler as SP

class SpatialPoolerTest(unittest.TestCase):

  def testCompute(self):
    """ Check that there are no errors in call to compute. """
    inputs = SDR( 100 ).randomize( .05 )
    active = SDR( 100 )
    sp = SP( inputs.dimensions, active.dimensions, stimulusThreshold = 1 )
    sp.compute( inputs, True, active )
    assert( active.getSum() > 0 )


if __name__ == "__main__":
  unittest.main()
