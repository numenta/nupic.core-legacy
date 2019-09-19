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

from htm.advanced.algorithms.thalamus import Thalamus
from htm.bindings.sdr import SDR
import numpy as np

class ThalamusTests(unittest.TestCase):

    def _inferThalamus(self, t, l6Input, ffInput):
        """
        Compute the effect of this feed forward input given the specific L6 input.
    
        :param t: instance of Thalamus
        :param l6Input:
        :param ffInput: a numpy array of 0's and 1's
        :return:
        """
        t.reset()
        t.deInactivateCells(l6Input)
        ffOutput = t.computeFeedForwardActivity(ffInput)
        return ffOutput
    
    
    # Simple tests for debugging
    def _trainThalamus(self, t):
        # Learn
        L6Pattern = SDR(t.l6CellCount)
        L6Pattern.sparse = [0, 1, 2, 3, 4, 5]
        t.learnL6Pattern(L6Pattern, [(0, 0), (2, 3)])
        L6Pattern.sparse = [6, 7, 8, 9, 10]
        t.learnL6Pattern(L6Pattern, [(1, 1), (3, 4)])
    
    
    def testThalamusNoBursting(self, verbose=False):
        """
        Test that thalamus relays around the trained locations,
        but does not busrt.
        """
        t = Thalamus()
    
        self._trainThalamus(t)
        ff = np.zeros((32,32))
        ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
        
        L6Pattern = SDR(t.l6CellCount)
        L6Pattern.sparse = [0, 1, 2, 3, 4, 5]
        
        result = self._inferThalamus(t, L6Pattern, ff)
        
        non_bursting = result[result >= 0.4].nonzero()[0].tolist()
        bursting = result[result >= 1.4].nonzero()[0].tolist()
        self.assertEqual([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], non_bursting, "Non-bursting not correct")
        self.assertEqual([], bursting, "Bursting not correct")
        
        if verbose:
            print(non_bursting)
            print(bursting)

    def testThalamusBursting(self, verbose=False):
        """
        Test that thalamus relays around the trained locations,
        and also does busrts.
        """
        t = Thalamus(trnThreshold=6)
    
        self._trainThalamus(t)
        ff = np.zeros((32,32))
        ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
        
        L6Pattern = SDR(t.l6CellCount)
        L6Pattern.sparse = [0, 1, 2, 3, 4, 5]
        
        result = self._inferThalamus(t, L6Pattern, ff)
        
        non_bursting = result[result >= 0.4].nonzero()[0].tolist()
        bursting = result[result >= 1.4].nonzero()[0].tolist()
        self.assertEqual([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], non_bursting, "Non-bursting not correct")
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8], bursting, "Bursting not correct")
        
        if verbose:
            print(non_bursting)
            print(bursting)


