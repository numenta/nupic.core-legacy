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
        num_samples = 5000
        M = Metrics( A, num_samples + 1 )
        for i in range( num_samples ):
            R.encode( i, A )
        print( M )
        assert(M.overlap.min()  > (1 - 1. / R.parameters.radius) - .02 )
        assert(M.overlap.max()  < (1 - 1. / R.parameters.radius) + .02 )
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
            1, 47, 76, 79, 80, 85, 102, 124, 134, 141, 150, 158, 161, 168, 176,
            202, 227, 236, 240, 246, 263, 273, 295, 319, 352, 367, 377, 380,
            392, 400, 410, 439, 468, 472, 493, 500, 506, 508, 515, 539, 542,
            574, 580, 583, 584, 617, 618, 636, 640, 648, 652, 664, 671, 697,
            708, 727, 734, 736, 744, 760, 773, 774, 777, 780, 785, 795, 796,
            801, 809, 810, 840, 863, 900, 902, 941, 950, 998]

        P = RDSE_Parameters()
        P.size     = GOLD.size
        P.sparsity = .08
        P.radius   = 12
        P.seed     = 42
        R = RDSE( P )
        A = R.encode( 987654 )
        print( A )
        assert( A == GOLD )

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
