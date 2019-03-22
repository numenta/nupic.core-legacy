# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, David McDougall
#
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
# ----------------------------------------------------------------------

"""Unit tests for Random Distributed Scalar Encoder."""

import pickle
import numpy as np
import unittest
import pytest
import time

from nupic.bindings.sdr import SDR, Metrics
from nupic.bindings.encoders import RDSE, RDSE_Parameters

class RDSE_Test(unittest.TestCase):
    @pytest.mark.skip("TODO UNIMPLEMENTED!")
    def testExampleUsage(self):
        1/0

    def testConstructor(self):
        params1 = RDSE_Parameters()
        params1.size     = 100
        params1.sparsity = .10
        params1.radius   = 10
        R1 = RDSE( params1 )

        params2 = R1.parameters
        params2.sparsity = 0 # Remove duplicate arguments
        params2.radius   = 0 # Remove duplicate arguments
        R2 = RDSE( params2 )

        A = SDR( R1.parameters.size )
        R1.encode( 66, A )

        B = R2.encode( 66 )
        assert( A == B )

    def testErrorChecks(self):
        params1 = RDSE_Parameters()
        params1.size     = 100
        params1.sparsity = .10
        params1.radius   = 10
        R1 = RDSE( params1 )
        A = SDR([10, 10])
        R1.encode( 33, A )

        # Test wrong input dimensions
        B = SDR( 1 )
        with self.assertRaises(RuntimeError):
            R1.encode( 3, B )

        # Test invalid parameters, size == 0
        params1.size = 0
        with self.assertRaises(RuntimeError):
            RDSE( params1 )
        params1.size = 100

        # Test invalid parameters, activeBits == 0
        params1.activeBits = 0
        params1.sparsity = 0.00001 # Rounds to zero!
        with self.assertRaises(RuntimeError):
            RDSE( params1 )

        # Test missing activeBits
        params2 = RDSE_Parameters()
        params2.size     = 100
        params2.radius   = 10
        with self.assertRaises(RuntimeError):
            RDSE( params2 )
        # Test missing resolution/radius
        params3 = RDSE_Parameters()
        params3.size       = 100
        params3.activeBits = 10
        with self.assertRaises(RuntimeError):
            RDSE( params3 )

        # Test too many parameters: activeBits & sparsity
        params4 = RDSE_Parameters()
        params4.size       = 100
        params4.sparsity   = .6
        params4.activeBits = 10
        params4.radius     = 4
        with self.assertRaises(RuntimeError):
            RDSE( params4 )
        # Test too many parameters: resolution & radius
        params5 = RDSE_Parameters()
        params5.size       = 100
        params5.activeBits = 10
        params5.radius     = 4
        params5.resolution = 4
        with self.assertRaises(RuntimeError):
            RDSE( params5 )

    def testSparsityActiveBits(self):
        """ Check that these arguments are equivalent. """
        # Round sparsity up
        P = RDSE_Parameters()
        P.size     = 100
        P.sparsity = .0251
        P.radius   = 10
        R = RDSE( P )
        assert( R.parameters.activeBits == 3 )
        # Round sparsity down
        P = RDSE_Parameters()
        P.size     = 100
        P.sparsity = .0349
        P.radius   = 10
        R = RDSE( P )
        assert( R.parameters.activeBits == 3 )
        # Check activeBits
        P = RDSE_Parameters()
        P.size       = 100
        P.activeBits = 50 # No floating point issues here.
        P.radius     = 10
        R = RDSE( P )
        assert( R.parameters.sparsity == .5 )

    def testRadiusResolution(self):
        """ Check that these arguments are equivalent. """
        # radius -> resolution
        P = RDSE_Parameters()
        P.size       = 2000
        P.activeBits = 100
        P.radius     = 12
        R = RDSE( P )
        self.assertAlmostEqual( R.parameters.resolution, 12./100, places=5)

        # resolution -> radius
        P = RDSE_Parameters()
        P.size       = 2000
        P.activeBits = 100
        P.resolution = 12
        R = RDSE( P )
        self.assertAlmostEqual( R.parameters.radius, 12*100, places=5)

        # Moving 1 resolution moves 1 bit (usually)
        P = RDSE_Parameters()
        P.size       = 2000
        P.activeBits = 100
        P.resolution = 3.33
        P.seed       = 42
        R = RDSE( P )
        sdrs = []
        for i in range(100):
            X = i * ( R.parameters.resolution )
            sdrs.append( R.encode( X ) )
            print("X", X, sdrs[-1])
        moved_one = 0
        moved_one_samples = 0
        for A, B in zip(sdrs, sdrs[1:]):
            delta = A.getSum() - A.getOverlap( B )
            if A.getSum() == B.getSum():
                assert( delta < 2 )
                moved_one += delta
                moved_one_samples += 1
        assert( moved_one >= .9 * moved_one_samples )

    def testAverageOverlap(self):
        """ Verify that nearby values have the correct amount of semantic
        similarity. Also measure sparsity & activation frequency. """
        P = RDSE_Parameters()
        P.size     = 2000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = RDSE( P )
        A = SDR( R.parameters.size )
        num_samples = 10000
        M = Metrics( A, num_samples + 1 )
        for i in range( num_samples ):
            R.encode( i, A )
        print( M )
        assert(M.overlap.min()  > (1 - 1. / R.parameters.radius) - .04 )
        assert(M.overlap.max()  < (1 - 1. / R.parameters.radius) + .04 )
        assert(M.overlap.mean() > (1 - 1. / R.parameters.radius) - .001 )
        assert(M.overlap.mean() < (1 - 1. / R.parameters.radius) + .001 )
        assert(M.sparsity.min()  > R.parameters.sparsity - .01 )
        assert(M.sparsity.max()  < R.parameters.sparsity + .01 )
        assert(M.sparsity.mean() > R.parameters.sparsity - .005 )
        assert(M.sparsity.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.min()  > R.parameters.sparsity - .05 )
        assert(M.activationFrequency.max()  < R.parameters.sparsity + .05 )
        assert(M.activationFrequency.mean() > R.parameters.sparsity - .005 )
        assert(M.activationFrequency.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.entropy() > .99 )

    def testRandomOverlap(self):
        """ Verify that distant values have little to no semantic similarity.
        Also measure sparsity & activation frequency. """
        P = RDSE_Parameters()
        P.size     = 2000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = RDSE( P )
        num_samples = 1000
        A = SDR( R.parameters.size )
        M = Metrics( A, num_samples + 1 )
        for i in range( num_samples ):
            X = i * R.parameters.radius
            R.encode( X, A )
        print( M )
        assert(M.overlap.max()  < .15 )
        assert(M.overlap.mean() < .10 )
        assert(M.sparsity.min()  > R.parameters.sparsity - .01 )
        assert(M.sparsity.max()  < R.parameters.sparsity + .01 )
        assert(M.sparsity.mean() > R.parameters.sparsity - .005 )
        assert(M.sparsity.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.min()  > R.parameters.sparsity - .05 )
        assert(M.activationFrequency.max()  < R.parameters.sparsity + .05 )
        assert(M.activationFrequency.mean() > R.parameters.sparsity - .005 )
        assert(M.activationFrequency.mean() < R.parameters.sparsity + .005 )
        assert(M.activationFrequency.entropy() > .99 )

    def testDeterminism(self):
        """ Verify that the same seed always gets the same results. """
        GOLD = SDR( 1000 )
        GOLD.sparse = [
            28, 47, 63, 93, 123, 124, 129, 131, 136, 140, 196, 205, 213, 239,
            258, 275, 276, 286, 305, 339, 345, 350, 372, 394, 395, 443, 449,
            462, 468, 471, 484, 514, 525, 557, 565, 570, 576, 585, 600, 609,
            631, 632, 635, 642, 651, 683, 693, 694, 696, 699, 721, 734, 772,
            790, 792, 795, 805, 806, 833, 836, 842, 846, 892, 896, 911, 914,
            927, 936, 947, 953, 955, 962, 965, 989, 990, 996]

        P = RDSE_Parameters()
        P.size     = GOLD.size
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = RDSE( P )
        A = R.encode( 987654 )
        print( A )
        assert( A == GOLD )

    def testSeed(self):
        P = RDSE_Parameters()
        P.size     = 1000
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 98
        R = RDSE( P )
        A = R.encode( 987654 )

        P.seed = 99
        R = RDSE( P )
        B = R.encode( 987654 )
        assert( A != B )

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
