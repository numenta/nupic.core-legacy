/* ---------------------------------------------------------------------
 * Copyright (C) 2018, David McDougall.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 * ----------------------------------------------------------------------
 */

#include <gtest/gtest.h>
#include <nupic/ntypes/Sdr.hpp>
#include <vector>

using namespace std;
using namespace nupic;

/* This also tests the size and dimensions are correct */
TEST(SdrTest, TestConstructor) {
    // Test 0 size
    ASSERT_ANY_THROW( SDR( vector<UInt>(0) ));
    ASSERT_ANY_THROW( SDR({ 0 }) );
    ASSERT_ANY_THROW( SDR({ 3, 2, 1, 0 }) );

    // Test 1-D
    vector<UInt> b_dims = {3};
    SDR b(b_dims);
    ASSERT_EQ( b.size, 3 );
    ASSERT_EQ( b.dimensions, b_dims );
    ASSERT_EQ( b.getSparse().size(), 1 );
    // zero initialized
    ASSERT_EQ( b.getDense(),     vector<Byte>({0, 0, 0}) );
    ASSERT_EQ( b.getFlatSparse(), vector<UInt>(0) );
    ASSERT_EQ( b.getSparse(),     vector<vector<UInt>>({{}}) );

    // Test 3-D
    vector<UInt> c_dims = {11, 15, 3};
    SDR c(c_dims);
    ASSERT_EQ( c.size, 11 * 15 * 3 );
    ASSERT_EQ( c.dimensions, c_dims );
    ASSERT_EQ( c.getSparse().size(), 3 );
    ASSERT_EQ( c.getFlatSparse().size(), 0 );
    // Test dimensions are copied not referenced
    c_dims.push_back(7);
    ASSERT_EQ( c.dimensions, vector<UInt>({11, 15, 3}) );
}

TEST(SdrTest, TestConstructorCopy) {
    // Test value/no-value is preserved
    SDR x({23});
    SDR x_copy(x);
    ASSERT_TRUE( x == x_copy );
    x.zero();
    SDR x_copy2 = SDR(x);
    ASSERT_TRUE( x == x_copy2 );

    // Test data is copied.
    SDR a({5});
    a.setDense( SDR_dense_t({0, 1, 0, 0, 0}));
    SDR b(a);
    ASSERT_EQ( b.getFlatSparse(),  vector<UInt>({1}) );
    ASSERT_TRUE(a == b);
}

TEST(SdrTest, TestZero) {
    SDR a({4, 4});
    a.setDense( vector<Byte>(16, 1) );
    a.zero();
    ASSERT_EQ( vector<Byte>(16, 0), a.getDense() );
    ASSERT_EQ( a.getFlatSparse().size(),  0);
    ASSERT_EQ( a.getSparse().size(),  2);
    ASSERT_EQ( a.getSparse().at(0).size(),  0);
    ASSERT_EQ( a.getSparse().at(1).size(),  0);
}

TEST(SdrTest, TestExample) {
    // Make an SDR with 9 values, arranged in a (3 x 3) grid.
    // "SDR" is an alias/typedef for SparseDistributedRepresentation.
    SDR  X( {3, 3} );
    vector<Byte> data({
        0, 1, 0,
        0, 1, 0,
        0, 0, 1 });

    // These three statements are equivalent.
    X.setDense(SDR_dense_t({ 0, 1, 0,
                             0, 1, 0,
                             0, 0, 1 }));
    ASSERT_EQ( data, X.getDense() );
    X.setFlatSparse(SDR_flatSparse_t({ 1, 4, 8 }));
    ASSERT_EQ( data, X.getDense() );
    X.setSparse(SDR_sparse_t({{ 0, 1, 2,}, { 1, 1, 2 }}));
    ASSERT_EQ( data, X.getDense() );

    // Access data in any format, SDR will automatically convert data formats.
    ASSERT_EQ( X.getDense(),      SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }) );
    ASSERT_EQ( X.getSparse(),     SDR_sparse_t({{ 0, 1, 2 }, {1, 1, 2}}) );
    ASSERT_EQ( X.getFlatSparse(), SDR_flatSparse_t({ 1, 4, 8 }) );

    // Data format conversions are cached, and when an SDR value changes the
    // cache is cleared.
    X.setFlatSparse(SDR_flatSparse_t({}));  // Assign new data to the SDR, clearing the cache.
    X.getDense();        // This line will convert formats.
    X.getDense();        // This line will resuse the result of the previous line

    X.zero();
    Byte *before = X.getDense().data();
    SDR_dense_t newData({ 1, 0, 0, 1, 0, 0, 1, 0, 0 });
    Byte *data_ptr = newData.data();
    X.setDense( newData );
    Byte *after = X.getDense().data();
    // X now points to newData, and newData points to X's old data.
    ASSERT_EQ( after, data_ptr );
    ASSERT_EQ( newData.data(), before );
    ASSERT_NE( before, after );

    X.zero();
    before = X.getDense().data();
    // The "&" is really important!  Otherwise vector copies.
    auto & dense = X.getDense();
    dense[2] = true;
    X.setDense( dense );              // Notify the SDR of the changes.
    after = X.getDense().data();
    ASSERT_EQ( X.getFlatSparse(), SDR_flatSparse_t({ 2 }) );
    ASSERT_EQ( before, after );
}

TEST(SdrTest, TestSetDenseVec) {
    SDR a({11, 10, 4});
    Byte *before = a.getDense().data();
    SDR_dense_t vec = vector<Byte>(440, 1);
    Byte *data = vec.data();
    a.setDense( vec );
    Byte *after = a.getDense().data();
    ASSERT_NE( before, after ); // not a copy.
    ASSERT_EQ( after, data ); // correct data buffer.
}

TEST(SdrTest, TestSetDenseByte) {
    SDR a({11, 10, 4});
    auto vec = vector<Byte>(a.size, 1);
    a.zero();
    a.setDense( (Byte*) vec.data());
    ASSERT_EQ( a.getDense(), vec );
    ASSERT_NE( ((vector<Byte>&) a.getDense()).data(), vec.data() ); // true copy not a reference
    ASSERT_EQ( a.getDense().data(), a.getDense().data() ); // But not a copy every time.
}

TEST(SdrTest, TestSetDenseUInt) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    a.setDense( (UInt*) vec.data() );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 1) );
    ASSERT_NE( a.getDense().data(), (const char*) vec.data()); // true copy not a reference
}

TEST(SdrTest, TestSetDenseArray) {
    // Test Byte sized data
    SDR A({ 3, 3 });
    vector<Byte> vec_byte({ 0, 1, 0, 0, 1, 0, 0, 0, 1 });
    auto arr = Array(NTA_BasicType_Byte, vec_byte.data(), vec_byte.size());
    A.setDense( arr );
    ASSERT_EQ( A.getFlatSparse(), SDR_flatSparse_t({ 1, 4, 8 }));

    // Test UInt64 sized data
    A.zero();
    vector<UInt64> vec_uint({ 1, 1, 0, 0, 1, 0, 0, 0, 1 });
    auto arr_uint64 = Array(NTA_BasicType_UInt64, vec_uint.data(), vec_uint.size());
    A.setDense( arr_uint64 );
    ASSERT_EQ( A.getFlatSparse(), SDR_flatSparse_t({ 0, 1, 4, 8 }));

    // Test Real sized data
    A.zero();
    vector<Real> vec_real({ 1., 1., 0., 0., 1., 0., 0., 0., 1. });
    auto arr_real = Array(NTA_BasicType_Real, vec_real.data(), vec_real.size());
    A.setDense( arr_real );
    ASSERT_EQ( A.getFlatSparse(), SDR_flatSparse_t({ 0, 1, 4, 8 }));
}

TEST(SdrTest, TestSetDenseInplace) {
    SDR a({10, 10});
    auto& a_data = a.getDense();
    ASSERT_EQ( a_data, vector<Byte>(100, 0) );
    a_data.assign( a.size, 1 );
    a.setDense( a_data );
    ASSERT_EQ( a.getDense().data(), a.getDense().data() );
    ASSERT_EQ( a.getDense().data(), a_data.data() );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 1) );
    ASSERT_EQ( a.getDense(), a_data );
}

TEST(SdrTest, TestSetFlatSparseVec) {
    SDR a({11, 10, 4});
    UInt *before = a.getFlatSparse().data();
    auto vec = vector<UInt>(a.size, 1);
    UInt *data = vec.data();
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.setFlatSparse( vec );
    UInt *after = a.getFlatSparse().data();
    ASSERT_NE( before, after );
    ASSERT_EQ( after, data );
}

TEST(SdrTest, TestSetFlatSparsePtr) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.zero();
    a.setFlatSparse( (UInt*) vec.data(), a.size );
    ASSERT_EQ( a.getFlatSparse(), vec );
    ASSERT_NE( a.getFlatSparse().data(), vec.data()); // true copy not a reference
}

TEST(SdrTest, TestSetFlatSparseArray) {
    SDR A({ 3, 3 });
    // Test UInt32 sized data
    vector<UInt32> vec_uint32({ 1, 4, 8 });
    auto arr_uint32 = Array(NTA_BasicType_UInt32, vec_uint32.data(), vec_uint32.size());
    A.setFlatSparse( arr_uint32 );
    ASSERT_EQ( A.getDense(), SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));

    // Test UInt64 sized data
    A.zero();
    vector<UInt64> vec_uint64({ 1, 4, 8 });
    auto arr_uint64 = Array(NTA_BasicType_UInt64, vec_uint64.data(), vec_uint64.size());
    A.setFlatSparse( arr_uint64 );
    ASSERT_EQ( A.getDense(), SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));

    // Test Real sized data
    A.zero();
    vector<Real> vec_real({ 1, 4, 8 });
    auto arr_real = Array(NTA_BasicType_Real, vec_real.data(), vec_real.size());
    A.setFlatSparse( arr_real );
    ASSERT_EQ( A.getDense(), SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
}

TEST(SdrTest, TestSetFlatSparseInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getFlatSparse();
    ASSERT_EQ( a_data, vector<UInt>(0) );
    a_data.push_back(0);
    a.setFlatSparse( a_data );
    ASSERT_EQ( a.getFlatSparse().data(), a.getFlatSparse().data() );
    ASSERT_EQ( a.getFlatSparse(),        a.getFlatSparse() );
    ASSERT_EQ( a.getFlatSparse().data(), a_data.data() );
    ASSERT_EQ( a.getFlatSparse(),        a_data );
    ASSERT_EQ( a.getFlatSparse(), vector<UInt>(1) );
    a_data.clear();
    a.setFlatSparse( a_data );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 0) );
}

TEST(SdrTest, TestSetSparse) {
    SDR a({4, 1, 3});
    void *before = a.getSparse().data();
    auto vec = vector<vector<UInt>>({
        { 0, 1, 2, 0 },
        { 0, 0, 0, 0 },
        { 0, 1, 2, 1 } });
    void *data = vec.data();
    a.setSparse( vec );
    void *after = a.getSparse().data();
    ASSERT_EQ( after, data );
    ASSERT_NE( before, after );
}

TEST(SdrTest, TestSetSparseCopy) {
    SDR a({ 3, 3 });
    void *before = a.getSparse().data();
    auto vec = vector<vector<Real>>({
        { 0., 1., 2. },
        { 1., 1., 2. } });
    void *data = vec.data();
    a.setSparse( vec );
    void *after = a.getSparse().data();
    ASSERT_EQ( before, after );  // Data copied from vec into sdr's buffer
    ASSERT_NE( after, data );   // Data copied from vec into sdr's buffer
    ASSERT_EQ( a.getFlatSparse(), SDR_flatSparse_t({ 1, 4, 8 }));
}

TEST(SdrTest, TestSetSparseInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getSparse();
    ASSERT_EQ( a_data, vector<vector<UInt>>({{}, {}}) );
    a_data[0].push_back(0);
    a_data[1].push_back(0);
    a_data[0].push_back(3);
    a_data[1].push_back(7);
    a_data[0].push_back(7);
    a_data[1].push_back(1);
    a.setSparse( a_data );
    ASSERT_EQ( a.getSum(), 3 );
    // I think some of these check the same things but thats ok.
    ASSERT_EQ( (void*) a.getSparse().data(), (void*) a.getSparse().data() );
    ASSERT_EQ( a.getSparse(), a.getSparse() );
    ASSERT_EQ( a.getSparse().data(), a_data.data() );
    ASSERT_EQ( a.getSparse(),        a_data );
    ASSERT_EQ( a.getFlatSparse(), vector<UInt>({0, 37, 71}) ); // Check data ok
    a_data[0].clear();
    a_data[1].clear();
    a.setSparse( a_data );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 0) );
}

TEST(SdrTest, TestSetSDR) {
    SDR a({5});
    SDR b({5});
    // Test dense assignment works
    a.setDense(SDR_dense_t({1, 1, 1, 1, 1}));
    b.setSDR(a);
    ASSERT_EQ( b.getFlatSparse(), vector<UInt>({0, 1, 2, 3, 4}) );
    // Test flat sparse assignment works
    a.setFlatSparse(SDR_flatSparse_t({0, 1, 2, 3, 4}));
    b.setSDR(a);
    ASSERT_EQ( b.getDense(), vector<Byte>({1, 1, 1, 1, 1}) );
    // Test sparse assignment works
    a.setSparse(SDR_sparse_t({{0, 1, 2, 3, 4}}));
    b.setSDR(a);
    ASSERT_EQ( b.getDense(), vector<Byte>({1, 1, 1, 1, 1}) );
}

TEST(SdrTest, TestGetDenseFromFlatSparse) {
    // Test zeros
    SDR z({4, 4});
    z.setFlatSparse(SDR_flatSparse_t({}));
    ASSERT_EQ( z.getDense(), vector<Byte>(16, 0) );

    // Test ones
    SDR nz({4, 4});
    nz.setFlatSparse(SDR_flatSparse_t(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
    ASSERT_EQ( nz.getDense(), vector<Byte>(16, 1) );

    // Test 1-D
    SDR d1({30});
    d1.setFlatSparse(SDR_flatSparse_t({1, 29, 4, 5, 7}));
    vector<Byte> ans(30, 0);
    ans[1] = 1;
    ans[29] = 1;
    ans[4] = 1;
    ans[5] = 1;
    ans[7] = 1;
    ASSERT_EQ( d1.getDense(), ans );

    // Test 3-D
    SDR d3({10, 10, 10});
    d3.setFlatSparse(SDR_flatSparse_t({0, 5, 50, 55, 500, 550, 555, 999}));
    vector<Byte> ans2(1000, 0);
    ans2[0]   = 1;
    ans2[5]   = 1;
    ans2[50]  = 1;
    ans2[55]  = 1;
    ans2[500] = 1;
    ans2[550] = 1;
    ans2[555] = 1;
    ans2[999] = 1;
    ASSERT_EQ( d3.getDense(), ans2 );
}

TEST(SdrTest, TestGetDenseFromSparse) {
    // Test simple 2-D
    SDR a({3, 3});
    a.setSparse(SDR_sparse_t({{1, 0, 2}, {2, 0, 2}}));
    vector<Byte> ans(9, 0);
    ans[0] = 1;
    ans[5] = 1;
    ans[8] = 1;
    ASSERT_EQ( a.getDense(), ans );

    // Test zeros
    SDR z({99, 1});
    z.setSparse(SDR_sparse_t({{}, {}}));
    ASSERT_EQ( z.getDense(), vector<Byte>(99, 0) );
}

TEST(SdrTest, TestGetFlatSparseFromDense) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto dense = a.getDense();
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ(a.getFlatSparse().at(0), 5);
    ASSERT_EQ(a.getFlatSparse().at(1), 8);

    // Test zero'd SDR.
    a.setDense( vector<Byte>(a.size, 0) );
    ASSERT_EQ( a.getFlatSparse().size(), 0 );
}

TEST(SdrTest, TestGetFlatSparseFromSparse) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto& index = a.getSparse();
    ASSERT_EQ( index.size(), 2 );
    ASSERT_EQ( index[0].size(), 0 );
    ASSERT_EQ( index[1].size(), 0 );
    // Insert flat index 4
    index.at(0).push_back(1);
    index.at(1).push_back(1);
    // Insert flat index 8
    index.at(0).push_back(2);
    index.at(1).push_back(2);
    // Insert flat index 5
    index.at(0).push_back(1);
    index.at(1).push_back(2);
    a.setSparse( index );
    ASSERT_EQ(a.getFlatSparse().at(0), 4);
    ASSERT_EQ(a.getFlatSparse().at(1), 8);
    ASSERT_EQ(a.getFlatSparse().at(2), 5);

    // Test zero'd SDR.
    a.setSparse(SDR_sparse_t( {{}, {}} ));
    ASSERT_EQ( a.getFlatSparse().size(), 0 );
}

TEST(SdrTest, TestGetSparseFromFlat) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto& index = a.getSparse();
    ASSERT_EQ( index.size(), 2 );
    ASSERT_EQ( index[0].size(), 0 );
    ASSERT_EQ( index[1].size(), 0 );
    a.setFlatSparse(SDR_flatSparse_t({ 4, 8, 5 }));
    ASSERT_EQ( a.getSparse(), vector<vector<UInt>>({
        { 1, 2, 1 },
        { 1, 2, 2 } }) );

    // Test zero'd SDR.
    a.setFlatSparse(SDR_flatSparse_t( { } ));
    ASSERT_EQ( a.getSparse(), vector<vector<UInt>>({{}, {}}) );
}

TEST(SdrTest, TestGetSparseFromDense) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto dense = a.getDense();
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ( a.getSparse(), vector<vector<UInt>>({
        { 1, 2 },
        { 2, 2 }}) );

    // Test zero'd SDR.
    a.setDense( vector<Byte>(a.size, 0) );
    ASSERT_EQ( a.getSparse()[0].size(), 0 );
    ASSERT_EQ( a.getSparse()[1].size(), 0 );
}

TEST(SdrTest, TestAt) {
    SDR a({3, 3});
    a.setFlatSparse(SDR_flatSparse_t( {4, 5, 8} ));
    ASSERT_TRUE( a.at( {1, 1} ));
    ASSERT_TRUE( a.at( {1, 2} ));
    ASSERT_TRUE( a.at( {2, 2} ));
    ASSERT_FALSE( a.at( {0 , 0} ));
    ASSERT_FALSE( a.at( {0 , 1} ));
    ASSERT_FALSE( a.at( {0 , 2} ));
    ASSERT_FALSE( a.at( {1 , 0} ));
    ASSERT_FALSE( a.at( {2 , 0} ));
    ASSERT_FALSE( a.at( {2 , 1} ));
}

TEST(SdrTest, TestSumSparsity) {
    SDR a({31, 17, 3});
    auto& dense = a.getDense();
    for(UInt i = 0; i < a.size; i++) {
        ASSERT_EQ( i, a.getSum() );
        EXPECT_FLOAT_EQ( (Real) i / a.size, a.getSparsity() );
        dense[i] = 1;
        a.setDense( dense );
    }
    ASSERT_EQ( a.size, a.getSum() );
    ASSERT_FLOAT_EQ( 1, a.getSparsity() );
}

TEST(SdrTest, TestPrint) {
    stringstream str;
    SDR a({100});
    a.print(str);
    // Use find so that trailing whitespace differences on windows/unix don't break it.
    ASSERT_NE( str.str().find( "SDR( 100 )" ), std::string::npos);

    stringstream str2;
    SDR b({ 9, 8 });
    b.print(str2);
    ASSERT_NE( str2.str().find( "SDR( 9, 8 )" ), std::string::npos);

    stringstream str3;
    SDR sdr3({ 3, 3 });
    sdr3.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
    sdr3.print(str3);
    ASSERT_NE( str3.str().find( "SDR( 3, 3 ) 1, 4, 8" ), std::string::npos);

    // Check that default aruments don't crash.
    cout << "PRINTING \"SDR( 3, 3 ) 1, 4, 8\" TO STDOUT: ";
    sdr3.print();
}

TEST(SdrTest, TestOverlap) {
    SDR a({3, 3});
    a.setDense(SDR_dense_t({1, 1, 1, 1, 1, 1, 1, 1, 1}));
    SDR b(a);
    ASSERT_EQ( a.overlap( b ), 9 );
    b.zero();
    ASSERT_EQ( a.overlap( b ), 0 );
    b.setDense(SDR_dense_t({0, 1, 0, 0, 1, 0, 0, 0, 1}));
    ASSERT_EQ( a.overlap( b ), 3 );
    a.zero(); b.zero();
    ASSERT_EQ( a.overlap( b ), 0 );
}

TEST(SdrTest, TestRandomize) {
    // Test sparsity is OK
    SDR a({1000});
    a.randomize( 0. );
    ASSERT_EQ( a.getSum(), 0 );
    a.randomize( .25 );
    ASSERT_EQ( a.getSum(), 250 );
    a.randomize( .5 );
    ASSERT_EQ( a.getSum(), 500 );
    a.randomize( .75 );
    ASSERT_EQ( a.getSum(), 750 );
    a.randomize( 1. );
    ASSERT_EQ( a.getSum(), 1000 );
    // Test RNG is deterministic
    SDR b(a);
    Random rng(77);
    Random rng2(77);
    a.randomize( .02, rng );
    b.randomize( .02, rng2 );
    ASSERT_TRUE( a == b);
    // Test different random number generators have different results.
    Random rng3( 1 );
    Random rng4( 2 );
    a.randomize( .02, rng3 );
    b.randomize( .02, rng4 );
    ASSERT_TRUE( a != b);
    // Test that this modifies RNG state and will generate different
    // distributions with the same RNG.
    Random rng5( 88 );
    a.randomize( .02, rng5 );
    b.randomize( .02, rng5 );
    ASSERT_TRUE( a != b);
    // Test default RNG has a different result every time
    a.randomize( .02 );
    b.randomize( .02 );
    ASSERT_TRUE( a != b);
    // Methodically test by running it many times and checking for an even
    // activation frequency at every bit.
    SDR af_test({ 251 /* prime number */ });
    UInt iterations = 100000;
    Real sparsity   = .10;
    vector<Real> af( af_test.size, 0 );
    for( UInt i = 0; i < iterations; i++ ) {
        af_test.randomize( sparsity );
        for( auto idx : af_test.getFlatSparse() )
            af[ idx ] += 1;
    }
    for( auto f : af ) {
        f = f / iterations / sparsity;
        ASSERT_GT( f,  .95 );
        ASSERT_LT( f, 1.05 );
    }
}

TEST(SdrTest, TestAddNoise) {
    SDR a({1000});
    a.randomize( .10 );
    SDR b(a);
    SDR c(a);
    // Test seed is deteministic
    b.setSDR(a);
    c.setSDR(a);
    Random b_rng( 44 );
    Random c_rng( 44 );
    b.addNoise( .5, b_rng );
    c.addNoise( .5, c_rng );
    ASSERT_TRUE( b == c );
    ASSERT_FALSE( a == b );
    // Test different seed generates different distributions
    b.setSDR(a);
    c.setSDR(a);
    Random rng1( 1 );
    Random rng2( 2 );
    b.addNoise( .5, rng1 );
    c.addNoise( .5, rng2 );
    ASSERT_TRUE( b != c );
    // Test addNoise changes PRNG state so two consequtive calls yeild different
    // results.
    Random prng( 55 );
    b.setSDR(a);
    b.addNoise( .5, prng );
    SDR b_cpy(b);
    b.setSDR(a);
    b.addNoise( .5, prng );
    ASSERT_TRUE( b_cpy != b );
    // Test default seed works ok
    b.setSDR(a);
    c.setSDR(a);
    b.addNoise( .5 );
    c.addNoise( .5 );
    ASSERT_TRUE( b != c );
    // Methodically test for every overlap.
    for( UInt x = 0; x <= 100; x++ ) {
        b.setSDR( a );
        b.addNoise( (Real)x / 100. );
        ASSERT_EQ( a.overlap( b ), 100 - x );
        ASSERT_EQ( b.getSum(), 100 );
    }
}

TEST(SdrTest, TestEquality) {
    vector<SDR*> test_cases;
    // Test different dimensions
    test_cases.push_back( new SDR({ 11 }));
    test_cases.push_back( new SDR({ 1, 1 }));
    test_cases.push_back( new SDR({ 1, 2, 3 }));
    // Test different data
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setDense(SDR_dense_t({0, 0, 1, 0, 1, 0, 1, 0, 0,}));
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setDense(SDR_dense_t({0, 1, 0, 0, 1, 0, 0, 1, 0}));
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setDense(SDR_dense_t({0, 1, 0, 0, 1, 0, 0, 0, 1}));
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setFlatSparse(SDR_flatSparse_t({0,}));
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setFlatSparse(SDR_flatSparse_t({3, 4, 6}));

    // Check that SDRs equal themselves
    for(UInt x = 0; x < test_cases.size(); x++) {
        for(UInt y = 0; y < test_cases.size(); y++) {
            SDR *a = test_cases[x];
            SDR *b = test_cases[y];
            if( x == y ) {
                ASSERT_TRUE(  *a == *b );
                ASSERT_FALSE( *a != *b );
            }
            else {
                ASSERT_TRUE(  *a != *b );
                ASSERT_FALSE( *a == *b );
            }
        }
    }

    for( SDR* z : test_cases )
        delete z;
}

TEST(SdrTest, TestSaveLoad) {
    const char *filename = "SdrSerialization.tmp";
    ofstream outfile;
    outfile.open(filename);

    // Test zero value
    SDR zero({ 3, 3 });
    zero.save( outfile );

    // Test dense data
    SDR dense({ 3, 3 });
    dense.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
    dense.save( outfile );

    // Test flat data
    SDR flat({ 3, 3 });
    flat.setFlatSparse(SDR_flatSparse_t({ 1, 4, 8 }));
    flat.save( outfile );

    // Test index data
    SDR index({ 3, 3 });
    index.setSparse(SDR_sparse_t({
            { 0, 1, 2 },
            { 1, 1, 2 }}));
    index.save( outfile );

    // Now load all of the data back into SDRs.
    outfile.close();
    ifstream infile( filename );

    if( false ) {
        // Print the file's contents
        std::stringstream buffer; buffer << infile.rdbuf();
        cout << buffer.str() << "EOF" << endl;
        infile.seekg( 0 ); // rewind to start of file.
    }

    SDR zero_2;
    zero_2.load( infile );
    SDR dense_2;
    dense_2.load( infile );
    SDR flat_2;
    flat_2.load( infile );
    SDR index_2;
    index_2.load( infile );

    infile.close();
    int ret = ::remove( filename );
    ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

    // Check that all of the data is OK
    ASSERT_TRUE( zero    == zero_2 );
    ASSERT_TRUE( dense   == dense_2 );
    ASSERT_TRUE( flat    == flat_2 );
    ASSERT_TRUE( index   == index_2 );
}
