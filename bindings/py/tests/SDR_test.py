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
import pytest
import time

from nupic.bindings.algorithms import SDR, SDR_Proxy

class SdrTest(unittest.TestCase):
    def testExampleUsage(self):
        # Make an SDR with 9 values, arranged in a (3 x 3) grid.
        X = SDR(dimensions = (3, 3))

        # These three statements are equivalent.
        X.dense = [[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1]]
        assert( X.dense.tolist() == [[ 0, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ]] )
        assert( [list(v) for v in X.sparse] == [[ 0, 1, 2 ], [1, 1, 2 ]] )
        assert( list(X.flatSparse) == [ 1, 4, 8 ] )
        X.sparse = [[0, 1, 2], [1, 1, 2]]
        assert( X.dense.tolist() == [[ 0, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ]] )
        assert( [list(v) for v in X.sparse] == [[ 0, 1, 2 ], [1, 1, 2 ]] )
        assert( list(X.flatSparse) == [ 1, 4, 8 ] )
        X.flatSparse = [ 1, 4, 8 ]

        # Access data in any format, SDR will automatically convert data formats,
        # even if it was not the format used by the most recent assignment to the
        # SDR.
        assert( X.dense.tolist() == [[ 0, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ]] )
        assert( [list(v) for v in X.sparse] == [[ 0, 1, 2 ], [1, 1, 2 ]] )
        assert( list(X.flatSparse) == [ 1, 4, 8 ] )

        # Data format conversions are cached, and when an SDR value changes the
        # cache is cleared.
        X.flatSparse = [1, 2, 3] # Assign new data to the SDR, clearing the cache.
        X.dense     # This line will convert formats.
        X.dense     # This line will resuse the result of the previous line

        X = SDR((1000, 1000))
        data = X.dense
        data[  0,   4] = 1
        data[444, 444] = 1
        X.dense = data
        assert(list(X.flatSparse) == [ 4, 444444 ])

    def testConstructor(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))
        assert(tuple(B.dimensions) == (100, 100, 1))
        # Test crazy dimensions, also test keyword arguments.
        C = SDR(dimensions = (2, 4, 5, 1,1,1,1, 3) )
        assert(C.size == 2*4*5*3)

        # Test copy constructor
        D = SDR( sdr = C ) # also test KW-arg
        assert( D.dimensions == C.dimensions )
        C.randomize( .5 )
        assert( D != C )
        D = SDR( C )
        assert( D == C )
        # Test convenience constructor, integer argument instead of list.
        V = SDR( 999 )
        assert( V.size == 999 )

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
        A.dense = A.dense
        assert(set(A.flatSparse) == set((0, 99)))
        # Test dense dimensions
        assert(B.dense.shape == (100, 100, 1))
        # No crash with dimensions
        B.dense[0, 0, 0] += 1
        B.dense[66, 2, 0] += 1
        B.dense[99, 99, 0] += 1
        B.dense = B.dense
        # Test wrong dimensions assigned
        C = SDR(( A.size + 1 ))
        C.randomize( .5 )
        test_cases = [
            (SDR(1), SDR(2)),
            (SDR(100),      SDR((100, 1))),
            (SDR((1, 100)), SDR((100, 1))),
        ]
        for left, right in test_cases:
            try:
                left.dense = right.dense
            except RuntimeError:
                pass
            else:
                self.fail()
        # Test assign data.
        A.dense = np.zeros( A.size, dtype=np.int16 )
        A.dense = np.ones(  A.size, dtype=np.uint64 )
        A.dense = np.zeros( A.size, dtype=np.int8 )
        A.dense = [1] * A.size
        B.dense = [[[1]] * 100 for _ in range(100)]

    def testDenseInplace(self):
        # Check that assigning dense data to itself (ie: sdr.dense = sdr.dense)
        # is significantly faster than copying the data on assignment.

        # Also, it should not be *too* much faster because this test-case is
        # tuned to very fast in both situations.
        A = SDR( 10*1000 )
        B = np.copy(A.dense)

        copy_time = time.clock()
        for i in range(100):
            A.dense = B
        copy_time = time.clock() - copy_time

        inplace_time = time.clock()
        for i in range(100):
            A.dense = A.dense
        inplace_time = time.clock() - inplace_time

        assert( inplace_time < copy_time / 3 )

    def testFlatSparse(self):
        A = SDR((103,))
        B = SDR((100, 100, 1))

        A.flatSparse
        B.flatSparse = [1,2,3,4]
        assert(all(B.flatSparse == np.array([1,2,3,4])))

        B.flatSparse = []
        assert( not B.dense.any() )

        # Test wrong dimensions assigned
        C = SDR( 1000 )
        C.randomize( .98 )
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
        B.dense = B.dense
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
        A.dense = np.ones( A.size )
        assert(A.getSum() == 103)

    def testGetSparsity(self):
        A = SDR((103,))
        assert(A.getSparsity() == 0)
        A.dense = np.ones( A.size )
        assert(A.getSparsity() == 1)

    def testGetOverlap(self):
        A = SDR((103,))
        B = SDR((103,))
        assert(A.getOverlap(B) == 0)

        A.dense[:10] = 1
        B.dense[:20] = 1
        A.dense = A.dense
        B.dense = B.dense
        assert(A.getOverlap(B) == 10)

        A.dense[:20] = 1
        A.dense = A.dense
        assert(A.getOverlap(B) == 20)

        A.dense[50:60] = 1
        B.dense[0] = 0
        A.dense = A.dense
        B.dense = B.dense
        assert(A.getOverlap(B) == 19)

        # Test wrong dimensions
        C = SDR((1,1,1,1, 103))
        C.randomize( .5 )
        try:
            A.getOverlap(C)
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
        assert( A.getOverlap(B) == 5 )

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
        A.dense = A.dense
        assert(str(A) == "SDR( 103 ) 0, 9, 102")
        A.zero()
        assert(str(A) == "SDR( 103 )")
        B.dense[0, 0, 0] = 1
        B.dense[99, 99, 0] = 1
        B.dense = B.dense
        assert(str(B) == "SDR( 100, 100, 1 ) 0, 9999")

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        for sparsity in (0, .3, 1):
            A = SDR((103,))
            A.randomize( sparsity )
            P = pickle.dumps( A )
            B = pickle.loads( P )
            assert( A == B )


class SdrProxyTest(unittest.TestCase):
    def testExampleUsage(self):
        assert( issubclass(SDR_Proxy, SDR) )
        # Convert SDR dimensions from (4 x 4) to (8 x 2)
        A = SDR([ 4, 4 ])
        B = SDR_Proxy( A, [8, 2])
        A.sparse =  ([1, 1, 2], [0, 1, 2])
        assert( (np.array(B.sparse) == ([2, 2, 5], [0, 1, 0]) ).all() )

    def testLostSDR(self):
        # You need to keep a reference to the SDR, since SDR class does not use smart pointers.
        B = SDR_Proxy(SDR((1000,)))
        with self.assertRaises(RuntimeError):
            B.dense

    def testChaining(self):
        A = SDR([10,10])
        B = SDR_Proxy(A)
        C = SDR_Proxy(B)
        D = SDR_Proxy(B)

        A.dense.fill( 1 )
        A.dense = A.dense
        assert( len(C.flatSparse) == A.size )
        assert( len(D.flatSparse) == A.size )
        del B

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
