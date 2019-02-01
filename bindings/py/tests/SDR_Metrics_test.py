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

"""Unit tests for SDR_Metrics python bindings"""

import pickle
import numpy as np
import unittest
import pytest

from nupic.bindings.algorithms import SDR, SDR_Proxy
from nupic.bindings.algorithms import SDR_Sparsity, SDR_ActivationFrequency, SDR_Overlap, SDR_Metrics

class SdrMetricsTest(unittest.TestCase):
    def testSparsityExample(self):
        A = SDR( dimensions = 1000 )
        B = SDR_Sparsity( A, period = 1000 )
        A.randomize( 0.01 )
        A.randomize( 0.15 )
        A.randomize( 0.05 )
        self.assertAlmostEqual(B.sparsity, 0.05, places = 2)
        self.assertAlmostEqual(B.min(),    0.01, places = 2)
        self.assertAlmostEqual(B.max(),    0.15, places = 2)
        self.assertAlmostEqual(B.mean(),   0.07, places = 2)
        self.assertAlmostEqual(B.std(),    0.06, places = 2)
        assert(str(B)     == "Sparsity Min/Mean/Std/Max 0.01 / 0.0700033 / 0.0588751 / 0.15")

    def testSparsityConstructor(self):
        A = SDR(1000)
        S = SDR_Sparsity(A, 100)
        A.randomize( .05 )
        A.randomize( .05 )
        A.randomize( .05 )
        assert( S.period == 100 )
        assert( S.dimensions == [1000,] )
        assert( S.samples == 3 )
        with self.assertRaises(RuntimeError):
            S.addData( A )

        B = SDR_Sparsity( dimensions = (1000,), period = 100)
        A.randomize( 0.10 )
        B.addData(A)
        A.randomize( 0.10 )
        B.addData(A)
        A.randomize( 0.10 )
        B.addData(A)
        assert( B.period == 100 )
        assert( B.dimensions == [1000,] )
        assert( B.samples == 3 )
        self.assertAlmostEqual( B.sparsity,  .10, places = 4)
        self.assertAlmostEqual( B.min(),     .10, places = 4)
        self.assertAlmostEqual( B.mean(),    .10, places = 4)
        self.assertAlmostEqual( B.max(),     .10, places = 4)
        self.assertAlmostEqual( B.std(),     0  , places = 4)

    def testAF_Example(self):
        A = SDR( 2 )
        B = SDR_ActivationFrequency( A, period = 1000 )
        A.dense = [0, 0]
        A.dense = [1, 1]
        A.dense = [0, 1]
        self.assertAlmostEqual( B.activationFrequency[0], 1 / 3., places = 2)
        self.assertAlmostEqual( B.activationFrequency[1], 2 / 3., places = 2)
        self.assertAlmostEqual( B.min(),                  1 / 3., places = 2)
        self.assertAlmostEqual( B.max(),                  2 / 3., places = 2)
        self.assertAlmostEqual( B.mean(),                 1 / 2., places = 2)
        self.assertAlmostEqual( B.std(),                  0.1666, places = 2)
        self.assertAlmostEqual( B.entropy(),              0.92,   places = 2)
        assert(str(B) ==
"""Activation Frequency Min/Mean/Std/Max 0.333333 / 0.5 / 0.166667 / 0.666667
Entropy 0.918296""")

    def testOverlapExample(self):
        A = SDR( dimensions = 2000 )
        B = SDR_Overlap( A, period = 1000 )
        A.randomize( 0.20 )
        A.addNoise( 0.95 )   # ->  5% overlap
        A.addNoise( 0.55 )   # -> 45% overlap
        A.addNoise( 0.72 )   # -> 28% overlap
        self.assertAlmostEqual( B.overlap , 0.28, places = 2)
        self.assertAlmostEqual( B.min()   , 0.05, places = 2)
        self.assertAlmostEqual( B.max()   , 0.45, places = 2)
        self.assertAlmostEqual( B.mean()  , 0.26, places = 2)
        self.assertAlmostEqual( B.std()   , 0.16, places = 2)
        assert(str(B) == "Overlap Min/Mean/Std/Max 0.05 / 0.260016 / 0.16389 / 0.45")

    def testMetricsExample(self):
        A = SDR( dimensions = 2000 )
        M = SDR_Metrics( A, period = 1000 )
        seed = 42 # Use hardcoded seed. Seed 0 will seed from system time, not what we want here.
        A.randomize( 0.10, seed )
        for i in range( 20 ):
            A.addNoise( 0.55, seed + i )

        assert( type(M.sparsity)            == SDR_Sparsity)
        assert( type(M.activationFrequency) == SDR_ActivationFrequency)
        assert( type(M.overlap)             == SDR_Overlap)
        gold = """SDR( 2000 )
    Sparsity Min/Mean/Std/Max 0.1 / 0.0999989 / 5.20038e-06 / 0.1
    Activation Frequency Min/Mean/Std/Max 0 / 0.100001 / 0.0974391 / 0.619048
    Entropy 0.830798
    Overlap Min/Mean/Std/Max 0.45 / 0.449998 / 1.06406e-05 / 0.45"""
        import re
        gold = re.sub(r'\s+', ' ', gold)
        real = re.sub(r'\s+', ' ', str(M))
        assert(gold == real)
