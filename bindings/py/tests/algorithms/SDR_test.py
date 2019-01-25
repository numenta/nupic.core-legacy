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

"""Unit tests for SDR."""

import pickle
import numpy as np
import unittest

from nupic.bindings.algorithms import SDR

class SdrTest(unittest.TestCase):
    def testConstructor(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))
        assert(tuple(B.dimensions) == (100, 100, 1))
        C = SDR((2, 4, 5, 1,1,1,1, 3))
        assert(C.size == 2*4*5*3)

        # Test copy constructor
        D = SDR( C )
        assert( D.dimensions == C.dimensions )
        C.randomize( .5 )
        assert( D != C )
        D = SDR( C )
        assert( D == C )

    def testZero(self):
        A = SDR((103,))
        A.flatSparse = list(range(20))
        A.zero()
        assert( np.sum( A.dense ) == 0 )

    def testDense(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))

        A.dense
        # Test is the same buffer every time
        A.dense[0] = 1
        A.dense[99] = 1
        assert(A.dense[0] + A.dense[99] == 2)
        # Test modify in-place
        A.setDenseInplace()
        assert(set(A.flatSparse) == set((0, 99)))
        # Test dense dimensions
        assert(B.dense.shape == (100, 100, 1))
        # No crash with dimensions
        B.dense[0, 0, 0] += 1
        B.dense[66, 2, 0] += 1
        B.dense[99, 99, 0] += 1
        B.dense = B.dense.reshape(-1)
        # Test wrong dimensions assigned
        C = SDR((2, 4, 5, 1,1,1,1, 3))
        C.randomize( .5 )
        try:
            A.dense = C.dense.reshape(-1)
        except RuntimeError:
            pass
        else:
            self.fail()

    def testFlatSparse(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))

        A.flatSparse
        B.flatSparse = [1,2,3,4]
        assert(all(B.flatSparse == np.array([1,2,3,4])))

        # Test wrong dimensions assigned
        C = SDR((2, 4, 5, 1,1,1,1, 3))
        C.randomize( .5 )
        try:
            A.flatSparse = C.flatSparse
        except RuntimeError:
            pass
        else:
            self.fail()

    def testSparse(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))
        C = SDR((2, 4, 5, 1,1,1,1, 3))

        A.sparse
        B.sparse = [[0, 55, 99], [0, 11, 99], [0, 0, 0]]
        assert(B.dense[0, 0, 0]   == 1)
        assert(B.dense[55, 11, 0] == 1)
        assert(B.dense[99, 99, 0] == 1)
        C.randomize( .5 )
        assert( len(C.sparse) == len(C.dimensions) )

        # Test wrong dimensions assigned
        C = SDR((2, 4, 5, 1,1,1,1, 3))
        C.randomize( .5 )
        try:
            A.sparse = C.sparse
        except RuntimeError:
            pass
        else:
            self.fail()

    def testSetSDR(self):
        A = SDR((103,))
        B = SDR((103,))
        A.flatSparse = [66]
        B.setSDR( A )
        assert( B.dense[66] == 1 )
        assert( B.getSum() == 1 )
        B.dense[77] = 1
        B.setDenseInplace()
        A.setSDR( B )
        assert( set(A.flatSparse) == set((66, 77)) )

        # Test wrong dimensions assigned
        C = SDR((2, 4, 5, 1,1,1,1, 3))
        C.randomize( .5 )
        try:
            A.setSDR(C)
        except RuntimeError:
            pass
        else:
            self.fail()

    def testGetSum(self):
        A = SDR((103,))
        assert(A.getSum() == 0)
        A.dense.fill(1)
        A.setDenseInplace()
        assert(A.getSum() == 103)

    def testGetSparsity(self):
        A = SDR((103,))
        assert(A.getSparsity() == 0)
        A.dense.fill(1)
        A.setDenseInplace()
        assert(A.getSparsity() == 1)

    def testOverlap(self):
        A = SDR((103,))
        B = SDR((103,))
        assert(A.overlap(B) == 0)

        A.dense[:10] = 1
        B.dense[:20] = 1
        A.setDenseInplace()
        B.setDenseInplace()
        assert(A.overlap(B) == 10)

        A.dense[:20] = 1
        A.setDenseInplace()
        assert(A.overlap(B) == 20)

        A.dense[50:60] = 1
        B.dense[0] = 0
        A.setDenseInplace()
        B.setDenseInplace()
        assert(A.overlap(B) == 19)

        # Test wrong dimensions
        C = SDR((1,1,1,1, 103))
        C.randomize( .5 )
        try:
            A.overlap(C)
        except RuntimeError:
            pass
        else:
            self.fail()

    def testRandomizeEqNe(self):
        A = SDR((103,))
        B = SDR((103,))
        A.randomize( .1 )
        B.randomize( .1 )
        assert( A != B )
        A.randomize( .1, 0 )
        B.randomize( .1, 0 )
        assert( A != B )
        A.randomize( .1, 42 )
        B.randomize( .1, 42 )
        assert( A == B )

    def testAddNoise(self):
        A = SDR((103,))
        B = SDR((103,))
        A.randomize( .1 )
        B.setSDR( A )
        A.addNoise( .5 )
        assert( A.overlap(B) == 5 )

        A.randomize( .3, 42 )
        B.randomize( .3, 42 )
        A.addNoise( .5 )
        B.addNoise( .5 )
        assert( A != B )

        A.randomize( .3, 42 )
        B.randomize( .3, 42 )
        A.addNoise( .5, 42 )
        B.addNoise( .5, 42 )
        assert( A == B )

    def testStr(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))
        A.dense[0] = 1
        A.dense[9] = 1
        A.dense[102] = 1
        A.setDenseInplace()
        assert(str(A) == "SDR( 103 ) 0, 9, 102")
        A.zero()
        assert(str(A) == "SDR( 103 )")
        B.dense[0, 0, 0] = 1
        B.dense[99, 99, 0] = 1
        B.setDenseInplace()
        assert(str(B) == "SDR( 100, 100, 1 ) 0, 9999")

    def testPickle(self):
        for sparsity in (0, .3, 1):
            A = SDR((103,))
            A.randomize( sparsity )
            P = pickle.dumps( A )
            B = pickle.loads( P )
            assert( A == B )
