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
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Unit tests for Scalar Encoder."""

import pickle
import numpy as np
import unittest
import pytest
import time

from nupic.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from nupic.bindings.algorithms import SDR, SDR_Metrics

class ScalarEncoder_Test(unittest.TestCase):
    @pytest.mark.skip("TODO UNIMPLEMENTED!")
    def testExampleUsage(self):
        1/0

    def testConstructor(self):
        p = ScalarEncoderParameters()
        p.size    = 1000
        p.active  = 20
        p.minimum = 0
        p.maximum = 345
        enc = ScalarEncoder( p )
        assert( not enc.parameters.clipInput )
        assert( not enc.parameters.periodic )
        assert( abs(enc.parameters.sparsity   - 20./1000) < .01 )
        assert( abs(enc.parameters.radius     - 7)        < 1 )
        assert( abs(enc.parameters.resolution - .35)      < .1 )

    def testEncode(self):
        p = ScalarEncoderParameters()
        p.size    = 10
        p.active  = 3
        p.minimum = 0
        p.maximum = 1
        enc = ScalarEncoder(p)
        sdr = SDR( 10 )
        enc.encode( 0, sdr )
        assert( list(sdr.sparse) == [0, 1, 2] )
        enc.encode( 1, sdr )
        assert( list(sdr.sparse) == [7, 8, 9] )

    def testBadParameters(self):
        # Start with sane parameters.
        p = ScalarEncoderParameters()
        p.size    = 10
        p.active  = 2
        p.minimum = 0
        p.maximum = 1
        ScalarEncoder(p)

        # Check a lot of bad parameters
        p.active = 12  # Can not activate more bits than are in the SDR.
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)

        p.active = 0 # not enough active
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)

        p.active = 1
        p.size = 0 # not enough bits
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.active = 2

        p.maximum = -1 # Maximum is less than the minimum
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.maximum = 1

        p.size     = 0
        p.active   = 0
        p.sparsity = .1  # Specify sparsity without output size
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.size     = 10
        p.active   = 2
        p.sparsity = 0

        p.sparsity = .2  # Sparsity & num active specified
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.sparsity = 0

        p.clipInput = True # Incompatible features...
        p.periodic  = True
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.clipInput = False
        p.periodic  = False

        p.radius = 1 # Size specified too many times
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.radius = 0

        p.resolution = 1 # Size specified too many times
        with self.assertRaises(RuntimeError):
            ScalarEncoder(p)
        p.resolution = 0

    def testBadEncode(self):
        # Test bad SDR
        p = ScalarEncoderParameters()
        p.size    = 10
        p.active  = 2
        p.minimum = 0
        p.maximum = 1
        enc  = ScalarEncoder(p)
        good = SDR( 10 )
        bad  = SDR( 5 )
        enc.encode( .25, good )
        with self.assertRaises(RuntimeError):
            enc.encode( .25, bad )

        # Test bad inputs, out of valid range & clipping disabled.
        with self.assertRaises(RuntimeError):
            enc.encode( -.0001, good )
        with self.assertRaises(RuntimeError):
            enc.encode( 1.0001, good )

    def testClipInput(self):
        p = ScalarEncoderParameters()
        p.size      = 345
        p.sparsity  = .05
        p.minimum   = 0
        p.maximum   = 1
        p.clipInput = 1
        enc  = ScalarEncoder(p)
        sdr1 = SDR( 345 )
        sdr2 = SDR( 345 )
        enc.encode( 0,  sdr1 )
        enc.encode( -1, sdr2 )
        assert( sdr1 == sdr2 )
        enc.encode( 1,  sdr1 )
        enc.encode( 10, sdr2 )
        assert( sdr1 == sdr2 )

    def testRadius(self):
        p = ScalarEncoderParameters()
        p.active  =  10
        p.minimum =   0
        p.maximum = 100
        p.radius  =  10
        enc = ScalarEncoder(p)
        sdr1 = SDR( enc.parameters.size )
        sdr2 = SDR( enc.parameters.size )

        enc.encode( 77, sdr1 )
        enc.encode( 77, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 10 )

        enc.encode( 0, sdr1 )
        enc.encode( 1, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 9 )

        enc.encode( 60, sdr1 )
        enc.encode( 69, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 1 )

        enc.encode( 45, sdr1 )
        enc.encode( 55, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 0 )

    def testResolution(self):
        p = ScalarEncoderParameters()
        p.active     =  10
        p.minimum    =   0
        p.maximum    = 100
        p.resolution =  .5
        enc = ScalarEncoder(p)
        sdr1 = SDR( enc.parameters.size )
        sdr2 = SDR( enc.parameters.size )

        enc.encode( .0, sdr1 )
        enc.encode( .1, sdr2 )
        assert( sdr1 == sdr2 )

        enc.encode( .0, sdr1 )
        enc.encode( .6, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 9 )

        enc.encode( 70,   sdr1 )
        enc.encode( 72.5, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 5 )

        enc.encode( 70, sdr1 )
        enc.encode( 75, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 0 )

        enc.encode( 60, sdr1 )
        enc.encode( 80, sdr2 )
        assert( sdr1.getOverlap( sdr2 ) == 0 )

    def testNaNs(self):
        p = ScalarEncoderParameters()
        p.size    = 100
        p.active  =  10
        p.minimum =   0
        p.maximum = 100
        enc = ScalarEncoder(p)
        sdr = SDR( 100 )
        enc.encode( float("nan"), sdr )
        assert( sdr.getSum() == 0 )

    def testPeriodic(self):
        p = ScalarEncoderParameters()
        p.size     = 100
        p.active   = 10
        p.minimum  = 0
        p.maximum  = 20
        p.periodic = True
        enc = ScalarEncoder( p )
        out = SDR( enc.parameters.size )
        mtr = SDR_Metrics(out, 9999)

        for i in range(201 * 10 + 1):
            x = (i % 201) / 10
            enc.encode( x, out )
            # print( x, out.sparse )

        print(str(mtr))
        assert( mtr.sparsity.min() >  .95 * .10 )
        assert( mtr.sparsity.max() < 1.05 * .10 )
        assert( mtr.activationFrequency.min() >  .9 * .10 )
        assert( mtr.activationFrequency.max() < 1.1 * .10 )
        assert( mtr.overlap.min() > .85 )

    def testStatistics(self):
        p = ScalarEncoderParameters()
        p.size      = 100
        p.active    = 10
        p.minimum   = 0
        p.maximum   = 20
        p.clipInput = True
        enc = ScalarEncoder( p )
        del p
        out = SDR( enc.parameters.size )
        mtr = SDR_Metrics(out, 9999)

        # The activation frequency of bits near the endpoints of the range is a
        # little weird, because the bits at the very end are not used as often
        # as the ones in the middle of the range, unless clipInputs is enabled.
        # If clipInputs is enabled then the bits 1 radius from the end get used
        # twice as often as the should because they respond to inputs off
        # outside of the valid range as well as inputs inside of the range.
        for i in np.linspace(
                        enc.parameters.minimum - enc.parameters.radius / 2,
                        enc.parameters.maximum + enc.parameters.radius / 2,
                        100 + 10 ):
            enc.encode( i, out )
            # print( i, out.sparse )

        print(str(mtr))
        assert( mtr.sparsity.min() >  .95 * .10 )
        assert( mtr.sparsity.max() < 1.05 * .10 )
        assert( mtr.activationFrequency.min() >  .50 * .10 )
        assert( mtr.activationFrequency.max() < 1.75 * .10 )
        assert( mtr.overlap.min() > .85 )

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
