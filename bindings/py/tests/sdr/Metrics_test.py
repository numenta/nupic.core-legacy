# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
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

"""Unit tests for Metrics python bindings"""

import pickle
import numpy as np
import unittest
import pytest

from htm.bindings import sdr

class MetricsTest(unittest.TestCase):
    def testSparsityExample(self):
        A = sdr.SDR( dimensions = 1000 )
        B = sdr.Sparsity( A, period = 1000 )
        A.randomize( 0.01 )
        A.randomize( 0.15 )
        A.randomize( 0.05 )
        self.assertAlmostEqual(B.sparsity, 0.05, places = 2)
        self.assertAlmostEqual(B.min(),    0.01, places = 2)
        self.assertAlmostEqual(B.max(),    0.15, places = 2)
        self.assertAlmostEqual(B.mean(),   0.07, places = 2)
        self.assertAlmostEqual(B.std(),    0.06, places = 2)
        assert(str(B) == "Sparsity Min/Mean/Std/Max 0.01 / 0.0700033 / 0.0588751 / 0.15")

    def testSparsityConstructor(self):
        A = sdr.SDR(1000)
        S = sdr.Sparsity(A, 100)
        A.randomize( .05 )
        A.randomize( .05 )
        A.randomize( .05 )
        assert( S.period == 100 )
        assert( S.dimensions == [1000,] )
        assert( S.samples == 3 )
        with self.assertRaises(RuntimeError):
            S.addData( A )

        B = sdr.Sparsity( dimensions = (1000,), period = 100)
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
        A = sdr.SDR( 2 )
        B = sdr.ActivationFrequency( A, period = 1000 )
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

    def testAF_initializeToValue(self):
        X  = sdr.SDR( 1 )
        AF = sdr.ActivationFrequency( X, period = 100, initialValue = .02 )
        assert( np.all( AF.activationFrequency == .02 ))
        X.sparse = [0]
        assert( AF.samples == 1 )
        alpha = 1. / AF.period
        decay = 1. - alpha
        assert( np.all( AF.activationFrequency == .02 * decay + alpha ))

    def testOverlapExample(self):
        A = sdr.SDR( dimensions = 2000 )
        B = sdr.Overlap( A, period = 1000 )
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

    def testReset(self):
        A = sdr.SDR( dimensions = 2000 )
        M = sdr.Metrics( A, period = 1000 )
        A.randomize( .10 )
        A.addNoise( .10 )
        # Test Metrics Reset
        M.reset()
        A.addNoise( 1 ) # Change every bit!
        A.addNoise( .10 )
        # Test Overlap Reset
        M.overlap.reset()
        A.addNoise( 1 ) # Change every bit!
        A.addNoise( .10 )
        assert(M.overlap.min() >= .89 ) # Allow 1% rounding error.

    def testMetricsExample(self):
        A = sdr.SDR( dimensions = 2000 )
        M = sdr.Metrics( A, period = 1000 )
        seed = 42 # Use hardcoded seed. Seed 0 will seed from system time, not what we want here.
        A.randomize( 0.10, seed )
        for i in range( 20 ):
            A.addNoise( 0.55, seed + i )

        assert( type(M.sparsity)            == sdr.Sparsity)
        assert( type(M.activationFrequency) == sdr.ActivationFrequency)
        assert( type(M.overlap)             == sdr.Overlap)
        gold = """SDR( 2000 )
    Sparsity Min/Mean/Std/Max 0.1 / 0.0999989 / 5.20038e-06 / 0.1
    Activation Frequency Min/Mean/Std/Max 0 / 0.100001 / 0.095393 / 0.571429
    Entropy 0.834509
    Overlap Min/Mean/Std/Max 0.45 / 0.449998 / 1.06406e-05 / 0.45"""
        import re
        gold = re.sub(r'\s+', ' ', gold)
        real = re.sub(r'\s+', ' ', str(M))
        assert(gold == real)
