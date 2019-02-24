# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, David McDougall
# The following terms and conditions apply:
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

"""Unit tests for Random Distributed Scalar Encoder."""

import pickle
import numpy as np
import unittest
import pytest
import time

from nupic.bindings.algorithms import SDR, SDR_Metrics, RDSE

class RDSE_Test(unittest.TestCase):
    @pytest.mark.skip("TODO UNIMPLEMENTED!")
    def testExampleUsage(self):
        1/0

    def testConstructor(self):
        A = SDR( 100 )
        R = RDSE( A.size, .10, 10 )
        R.encode( 66, A )
        B = R.encode( 66 )
        assert( A == B )

    def testAverageOverlap(self):
        """ Verify that nearby values have the correct amount of semantic
        similarity. Also measure sparsity & activation frequency. """
        A = SDR( 1000 )
        M = SDR_Metrics( A, 999999 )
        R = RDSE( A.size, .10, 12 )
        for i in range( 10000 ):
            R.encode( i, A )
        assert(M.sparsity.min()  > .10 - .02 )
        assert(M.sparsity.max()  < .10 + .02 )
        assert(M.sparsity.mean() > .10 - .005 )
        assert(M.sparsity.mean() < .10 + .005 )
        assert(M.activationFrequency.min()  > .10 - .05 )
        assert(M.activationFrequency.max()  < .10 + .05 )
        assert(M.activationFrequency.mean() > .10 - .005 )
        assert(M.activationFrequency.mean() < .10 + .005 )
        assert(M.activationFrequency.entropy() > .98 )
        assert(M.overlap.min()  > (1-1./12) - .25 )
        assert(M.overlap.max()  < (1-1./12) + .25 )
        assert(M.overlap.mean() > (1-1./12) - .05 )
        assert(M.overlap.mean() < (1-1./12) + .05 )

    def testRandomOverlap(self):
        """ Verify that distant values have little to no semantic similarity.
        Also measure sparsity & activation frequency. """
        A = SDR( 1000 )
        M = SDR_Metrics( A, 999999 )
        R = RDSE( A.size, .10, 12 )
        x = 0
        y = 2 * 12
        for i in range( 10000 ):
            R.encode( x, A )
            x += max( y, i )
        assert(M.sparsity.min()  > .10 - .02 )
        assert(M.sparsity.max()  < .10 + .02 )
        assert(M.sparsity.mean() > .10 - .005 )
        assert(M.sparsity.mean() < .10 + .005 )
        assert(M.activationFrequency.min()  > .10 - .05 )
        assert(M.activationFrequency.max()  < .10 + .05 )
        assert(M.activationFrequency.mean() > .10 - .005 )
        assert(M.activationFrequency.mean() < .10 + .005 )
        assert(M.activationFrequency.entropy() > .98 )
        assert(M.overlap.max()  < .40 )
        assert(M.overlap.mean() < .20 )

    def testDeterminism(self):
        """ Verify that the same seed always gets the same results. """
        GOLD = SDR( 1000 )
        GOLD.flatSparse = [
            11, 21, 49, 100, 136, 140, 150, 151, 177, 207, 242, 284, 287, 292,
            295, 323, 341, 377, 455, 475, 501, 520, 547, 560, 574, 595, 681,
            693, 702, 710, 739, 742, 748, 776, 794, 798, 805, 896, 898, 915,
            937, 950, 954, 955, 983, 984, 992]

        A = SDR( 1000 )
        seed = 93
        R = RDSE( A.size, .05, 12, seed )
        R.encode( 987654, A )
        print( A )
        assert( A == GOLD )

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
