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

from nupic.bindings.algorithms import SDR, SDR_Proxy, SDR_Intersection, SDR_Concatenation

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
        A = SDR( 100*1000 )
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


class SdrIntersectionTest(unittest.TestCase):
    def testExampleUsage(self):
        A = SDR( 10 )
        B = SDR( 10 )
        A.flatSparse = [2, 3, 4, 5]
        B.flatSparse = [0, 1, 2, 3]
        X = SDR_Intersection(A, B)
        assert((X.flatSparse == [2, 3]).all())
        B.zero()
        assert(X.getSparsity() == 0)

    def testConstructor(self):
        assert( issubclass(SDR_Intersection, SDR) )
        # Test 2 Arguments
        A = SDR( 2000 )
        B = SDR( A.size )
        X = SDR_Intersection(A, B)
        A.randomize( .20 )
        B.randomize( .20 )
        assert( X.getSum() > 0 )
        assert( X.inputs == [A, B] )
        A.zero()
        assert( X.getSum() == 0 )
        del X
        A.zero()
        B.zero()
        del B
        del A
        # Test 3 Arguments
        A = SDR( 2000 )
        B = SDR( 2000 )
        C = SDR( 2000 )
        X = SDR_Intersection(A, B, C)
        A.randomize( .6 )
        B.randomize( .6 )
        C.randomize( .6 )
        assert( X.inputs == [A, B, C] )
        assert( X.getSparsity() >  .75 * ( .6 ** 3 ))
        assert( X.getSparsity() < 1.25 * ( .6 ** 3 ))
        del B
        del A
        del X
        del C
        # Test 4 Arguments
        A = SDR( 2000 ); A.randomize( .9 )
        B = SDR( 2000 ); B.randomize( .9 )
        C = SDR( 2000 ); C.randomize( .9 )
        D = SDR( 2000 ); D.randomize( .9 )
        X = SDR_Intersection(A, B, C, D)
        assert( X.inputs == [A, B, C, D] )
        # Test list constructor
        X = SDR_Intersection( [A, B, C, D] )
        assert( X.size       == 2000 )
        assert( X.dimensions == [2000] )
        assert( X.getSum()    > 0 )
        A.zero()
        assert( X.getSum()   == 0 )
        assert( X.inputs     == [A, B, C, D] )

    def testSparsity(self):
        test_cases = [
            ( 0.5,  0.5 ),
            ( 0.1,  0.9 ),
            ( 0.25, 0.3 ),
            ( 0.5,  0.5,  0.5 ),
            ( 0.95, 0.95, 0.95 ),
            ( 0.10, 0.10, 0.60 ),
            ( 0.0,  1.0,  1.0 ),
            ( 0.5,  0.5,  0.5, 0.5),
            ( 0.11, 0.25, 0.33, 0.5, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98),
        ]
        seed = 99
        for sparsities in test_cases:
            sdrs = []
            for S in sparsities:
                inp = SDR( 10000 )
                inp.randomize( S, seed)
                seed += 1
                sdrs.append( inp )
            X = SDR_Intersection( sdrs )
            mean_sparsity = np.product( sparsities )
            assert( X.getSparsity() >= (2./3.) * mean_sparsity )
            assert( X.getSparsity() <= (4./3.) * mean_sparsity )

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented


class SdrConcatenationTest(unittest.TestCase):
    def testExampleUsage(self):
        assert( issubclass(SDR_Intersection, SDR) )
        A = SDR( 100 )
        B = SDR( 100 )
        C = SDR_Concatenation( A, B )
        assert(C.dimensions == [200])

        D = SDR(( 640, 480, 3 ))
        E = SDR(( 640, 480, 7 ))
        F = SDR_Concatenation( D, E, 2 )
        assert(F.dimensions == [ 640, 480, 10 ])

    def testConstructor(self):
        # Test all of the constructor overloads
        A = SDR(( 100, 2 ))
        B = SDR(( 100, 2 ))
        C = SDR(( 100, 2 ))
        D = SDR(( 100, 2 ))
        SDR_Concatenation( A, B )
        SDR_Concatenation( A, B, 1 )
        SDR_Concatenation( A, B, C )
        SDR_Concatenation( A, B, C, 1 )
        SDR_Concatenation( A, B, C, D )
        SDR_Concatenation( A, B, C, D, 1 )
        SDR_Concatenation( [A, B, C, D] )
        SDR_Concatenation( [A, B, C, D], 1 )
        SDR_Concatenation( inputs = [A, B, C, D], axis = 1 )

    def testConstructorErrors(self):
        def _assertAnyException(func):
            try:
                func()
            except RuntimeError:
                return
            except TypeError:
                return
            else:
                self.fail()

        A = SDR( 100 )
        B = SDR(( 100, 2 ))
        C = SDR([ 3, 3 ])
        D = SDR([ 3, 4 ])
        # Test bad argument dimensions
        _assertAnyException(lambda: SDR_Concatenation(A))      # Not enough inputs!
        _assertAnyException(lambda: SDR_Concatenation(A, B))
        _assertAnyException(lambda: SDR_Concatenation(B, C))  # All dims except axis must match!
        _assertAnyException(lambda: SDR_Concatenation(C, D))  # All dims except axis must match!
        SDR_Concatenation(C, D, 1) # This should work
        _assertAnyException(lambda: SDR_Concatenation( inputs = (C, D), axis = 2))  # invalid axis
        _assertAnyException(lambda: SDR_Concatenation( inputs = (C, D), axis = -1))  # invalid axis

    def testDelete(self):
        # Make & Delete it a few times to make sure that doesn't crash.
        A = SDR(100)
        B = SDR(100)
        C = SDR(100)
        X = SDR_Concatenation(A, B, C)
        SDR_Concatenation(A, B, C)
        Y = SDR_Concatenation(A, C)
        SDR_Concatenation(B, C)
        del B
        del A
        del Y
        del C
        del X

    def testMirroring(self):
        A = SDR( 200 )
        Ax10 = SDR_Concatenation( [A] * 10 )
        A.randomize( .33 )
        assert( .30 < Ax10.getSparsity() and Ax10.getSparsity() < .36 )

    def testVersusNumpy(self):
        # Each testcase is a pair of lists of SDR dimensions and axis
        # dimensions.
        test_cases = [
            ([(9, 30, 40),  (2, 30, 40)],          0),
            ([(2, 30, 40),  (2, 99, 40)],          1),
            ([(2, 30, 40),  (2, 30, 99)],          2),
            ([(100,), (10), (30)],                 0),
            ([(100,2), (10,2), (30,2)],            0),
            ([(1,77), (1,99), (1,88)],             1),
            ([(1,77,2), (1,99,2), (1,88,2)],       1),
        ]
        for sdr_dims, axis in test_cases:
            sdrs = [SDR(dims) for dims in sdr_dims]
            [sdr.randomize(.50) for sdr in sdrs]
            cat    = SDR_Concatenation( sdrs, axis )
            np_cat = np.concatenate([sdr.dense for sdr in sdrs], axis=axis)
            assert((cat.dense == np_cat).all())

    @pytest.mark.skip(reason="Known issue: https://github.com/htm-community/nupic.cpp/issues/160")
    def testPickle(self):
        assert(False) # TODO: Unimplemented
