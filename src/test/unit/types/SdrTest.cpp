/* ---------------------------------------------------------------------
 * Copyright (C) 2018-2019, David McDougall.
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
 * ---------------------------------------------------------------------- */

#include <gtest/gtest.h>
#include <htm/types/Sdr.hpp>
#include <vector>
#include <random>

static bool verbose = false;
#define VERBOSE if(verbose) std::cerr << "[          ]"

namespace testing {
    
using namespace std;
using namespace htm;

/* This also tests the size and dimensions are correct */
TEST(SdrTest, TestConstructor) {
    // Test 0 dimensions
    EXPECT_ANY_THROW( SDR( vector<UInt>(0) ));
    // Test 0 size
    EXPECT_NO_THROW( SDR({ 0 }) );
    EXPECT_ANY_THROW( SDR({ 3, 2, 1, 0 }) );
    EXPECT_NO_THROW(  SDR({ 3, 2, 1}) );

    // Test 1-D
    vector<UInt> b_dims = {3};
    SDR b(b_dims);
    ASSERT_EQ( b.size, 3ul );
    ASSERT_EQ( b.dimensions, b_dims );
    ASSERT_EQ( b.getCoordinates().size(), 1ul );
    // zero initialized
    ASSERT_EQ( b.getDense(),     vector<Byte>({0, 0, 0}) );
    ASSERT_EQ( b.getSparse(), vector<UInt>(0) );
    ASSERT_EQ( b.getCoordinates(),     vector<vector<UInt>>({{}}) );

    // Test 3-D
    vector<UInt> c_dims = {11u, 15u, 3u};
    SDR c(c_dims);
    ASSERT_EQ( c.size, 11u * 15u * 3u );
    ASSERT_EQ( c.dimensions, c_dims );
    ASSERT_EQ( c.getCoordinates().size(), 3ul );
    ASSERT_EQ( c.getSparse().size(), 0ul );
    // Test dimensions are copied not referenced
    c_dims.push_back(7);
    ASSERT_EQ( c.dimensions, vector<UInt>({11u, 15u, 3u}) );
}

TEST(SdrTest, TestEmptyPlaceholder) { //for NetworkAPI
    EXPECT_NO_THROW( SDR({0})  );
    EXPECT_EQ( SDR({0}).size, 0u  );
}

TEST(SdrTest, TestConstructorCopy) {
    // Test data is copied.
    SDR a({5});
    a.setDense( SDR_dense_t({0, 1, 0, 0, 0}));
    SDR b(a);
    ASSERT_EQ( b.getSparse(),  vector<UInt>({1}) );
    ASSERT_TRUE(a == b);
}

TEST(SdrTest, TestZero) {
    SDR a({4, 4});
    a.setDense( vector<Byte>(16, 1) );
    a.zero();
    ASSERT_EQ( vector<Byte>(16, 0), a.getDense() );
    ASSERT_EQ( a.getSparse().size(),  0ul);
    ASSERT_EQ( a.getCoordinates().size(),  2ul);
    ASSERT_EQ( a.getCoordinates().at(0).size(),  0ul);
    ASSERT_EQ( a.getCoordinates().at(1).size(),  0ul);
}

TEST(SdrTest, TestSDR_Examples) {
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
    X.setSparse(SDR_sparse_t({ 1, 4, 8 }));
    ASSERT_EQ( data, X.getDense() );
    X.setCoordinates(SDR_coordinate_t({{ 0, 1, 2,}, { 1, 1, 2 }}));
    ASSERT_EQ( data, X.getDense() );

    // Access data in any format, SDR will automatically convert data formats.
    ASSERT_EQ( X.getDense(),      SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }) );
    ASSERT_EQ( X.getCoordinates(),     SDR_coordinate_t({{ 0, 1, 2 }, {1, 1, 2}}) );
    ASSERT_EQ( X.getSparse(), SDR_sparse_t({ 1, 4, 8 }) );

    // Data format conversions are cached, and when an SDR value changes the
    // cache is cleared.
    X.setSparse(SDR_sparse_t({}));  // Assign new data to the SDR, clearing the cache.
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
    ASSERT_EQ( X.getSparse(), SDR_sparse_t({ 2 }) );
    ASSERT_EQ( before, after );
}

TEST(SdrTest, TestReshape) {
    SDR A({3,4,5});
    A.randomize( 5.0f / A.size );
    const auto data = A.getSparse();
    // Make the SDR have only the coordinate dataformat, because that could get
    // corrupted by the reshape.
    A.setCoordinates( A.getCoordinates() );
    A.reshape({5,4*3});
    ASSERT_EQ( A.getSparse(), data );
    ASSERT_EQ( A.dimensions, vector<UInt>({5,12}) );
    ASSERT_ANY_THROW( A.reshape({ 2 }) );
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
    ASSERT_NE( a.getDense().data(), (const Byte*) vec.data()); // true copy not a reference
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

TEST(SdrTest, TestSetSparseVec) {
    SDR a({11, 10, 4});
    UInt *before = a.getSparse().data();
    auto vec = vector<UInt>(a.size, 1);
    UInt *data = vec.data();
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.setSparse( vec );
    UInt *after = a.getSparse().data();
    ASSERT_NE( before, after );
    ASSERT_EQ( after, data );
}

TEST(SdrTest, TestSetSparsePtr) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.zero();
    a.setSparse( (UInt*) vec.data(), a.size );
    ASSERT_EQ( a.getSparse(), vec );
    ASSERT_NE( a.getSparse().data(), vec.data()); // true copy not a reference
}

TEST(SdrTest, TestSetSparseInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getSparse();
    ASSERT_EQ( a_data, vector<UInt>(0) );
    a_data.push_back(0);
    a.setSparse( a_data );
    ASSERT_EQ( a.getSparse().data(), a.getSparse().data() );
    ASSERT_EQ( a.getSparse(),        a.getSparse() );
    ASSERT_EQ( a.getSparse().data(), a_data.data() );
    ASSERT_EQ( a.getSparse(),        a_data );
    ASSERT_EQ( a.getSparse(), vector<UInt>(1) );
    a_data.clear();
    a.setSparse( a_data );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 0) );
}

TEST(SdrTest, TestSetCoordinates) {
    SDR a({4, 1, 3});
    void *before = a.getCoordinates().data();
    auto vec = vector<vector<UInt>>({
        { 0, 1, 2, 0 },
        { 0, 0, 0, 0 },
        { 0, 1, 2, 1 } });
    void *data = vec.data();
    a.setCoordinates( vec );
    void *after = a.getCoordinates().data();
    ASSERT_EQ( after, data );
    ASSERT_NE( before, after );
}

TEST(SdrTest, TestSetCoordinatesCopy) {
    SDR a({ 3, 3 });
    void *before = a.getCoordinates().data();
    auto vec = vector<vector<Real>>({
        { 0.0f, 1.0f, 2.0f },
        { 1.0f, 1.0f, 2.0f } });
    void *data = vec.data();
    a.setCoordinates( vec );
    void *after = a.getCoordinates().data();
    ASSERT_EQ( before, after );  // Data copied from vec into sdr's buffer
    ASSERT_NE( after, data );   // Data copied from vec into sdr's buffer
    ASSERT_EQ( a.getSparse(), SDR_sparse_t({ 1, 4, 8 }));
}

TEST(SdrTest, TestSetCoordinatesInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getCoordinates();
    ASSERT_EQ( a_data, vector<vector<UInt>>({{}, {}}) );
    a_data[0].push_back(0);
    a_data[1].push_back(0);
    a_data[0].push_back(3);
    a_data[1].push_back(7);
    a_data[0].push_back(7);
    a_data[1].push_back(1);
    a.setCoordinates( a_data );
    ASSERT_EQ( a.getSum(), 3ul );
    // I think some of these check the same things but thats ok.
    ASSERT_EQ( (void*) a.getCoordinates().data(), (void*) a.getCoordinates().data() );
    ASSERT_EQ( a.getCoordinates(), a.getCoordinates() );
    ASSERT_EQ( a.getCoordinates().data(), a_data.data() );
    ASSERT_EQ( a.getCoordinates(),        a_data );
    ASSERT_EQ( a.getSparse(), vector<UInt>({0, 37, 71}) ); // Check data ok
    a_data[0].clear();
    a_data[1].clear();
    a.setCoordinates( a_data );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 0) );
}

TEST(SdrTest, TestSetSDR) {
    SDR a({5});
    SDR b({5});
    // Test dense assignment works
    a.setDense(SDR_dense_t({1, 1, 1, 1, 1}));
    b.setSDR(a);
    ASSERT_EQ( b.getSparse(), vector<UInt>({0, 1, 2, 3, 4}) );
    // Test sparse assignment works
    a.setSparse(SDR_sparse_t({0, 1, 2, 3, 4}));
    b.setSDR(a);
    ASSERT_EQ( b.getDense(), vector<Byte>({1, 1, 1, 1, 1}) );
    // Test coordinate assignment works
    a.setCoordinates(SDR_coordinate_t({{0, 1, 2, 3, 4}}));
    b.setSDR(a);
    ASSERT_EQ( b.getDense(), vector<Byte>({1, 1, 1, 1, 1}) );
    // Test equals override works
    a.zero();
    b = a;
    a.randomize( 1.0f );
    ASSERT_EQ( b.getSum(), 0u );
}

TEST(SdrTest, TestGetDenseFromSparse) {
    // Test zeros
    SDR z({4, 4});
    z.setSparse(SDR_sparse_t({}));
    ASSERT_EQ( z.getDense(), vector<Byte>(16, 0) );

    // Test ones
    SDR nz({4, 4});
    nz.setSparse(SDR_sparse_t(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
    ASSERT_EQ( nz.getDense(), vector<Byte>(16, 1) );

    // Test 1-D
    SDR d1({30});
    d1.setSparse(SDR_sparse_t({1, 4, 5, 7, 29}));
    vector<Byte> ans(30, 0);
    ans[1] = 1;
    ans[4] = 1;
    ans[5] = 1;
    ans[7] = 1;
    ans[29] = 1;
    ASSERT_EQ( d1.getDense(), ans );

    // Test 3-D
    SDR d3({10, 10, 10});
    d3.setSparse(SDR_sparse_t({0, 5, 50, 55, 500, 550, 555, 999}));
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

TEST(SdrTest, TestGetDenseFromCoordinates) {
    // Test simple 2-D
    SDR a({3, 3});
    a.setCoordinates(SDR_coordinate_t({{0, 1, 2}, {0, 2, 2}}));
    vector<Byte> ans(9, 0);
    ans[0] = 1;
    ans[5] = 1;
    ans[8] = 1;
    ASSERT_EQ( a.getDense(), ans );

    // Test zeros
    SDR z({99, 1});
    z.setCoordinates(SDR_coordinate_t({{}, {}}));
    ASSERT_EQ( z.getDense(), vector<Byte>(99, 0) );
}

TEST(SdrTest, TestGetSparseFromDense) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto dense = a.getDense();
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ(a.getSparse().at(0), 5ul);
    ASSERT_EQ(a.getSparse().at(1), 8ul);

    // Test zero'd SDR.
    a.setDense( vector<Byte>(a.size, 0) );
    ASSERT_EQ( a.getSparse().size(), 0ul );
}

TEST(SdrTest, TestGetSparseFromCoordinates) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto& index = a.getCoordinates();
    ASSERT_EQ( index.size(), 2ul );
    ASSERT_EQ( index[0].size(), 0ul );
    ASSERT_EQ( index[1].size(), 0ul );
    // Insert flat index 4
    index.at(0).push_back(1);
    index.at(1).push_back(1);
    // Insert flat index 8
    index.at(0).push_back(2);
    index.at(1).push_back(2);
    // Insert flat index 5
    index.at(0).push_back(1);
    index.at(1).push_back(2);
    a.setCoordinates( index );
    ASSERT_EQ(a.getSparse().at(0), 4ul);
    ASSERT_EQ(a.getSparse().at(1), 8ul);
    ASSERT_EQ(a.getSparse().at(2), 5ul);

    // Test zero'd SDR.
    a.setCoordinates(SDR_coordinate_t( {{}, {}} ));
    ASSERT_EQ( a.getSparse().size(), 0ul );
}

TEST(SdrTest, TestGetCoordinatesFromSparse) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto& index = a.getCoordinates();
    ASSERT_EQ( index.size(), 2ul );
    ASSERT_EQ( index[0].size(), 0ul );
    ASSERT_EQ( index[1].size(), 0ul );
    a.setSparse(SDR_sparse_t({ 4, 5, 8 }));
    ASSERT_EQ( a.getCoordinates(), vector<vector<UInt>>({
        { 1, 1, 2 },
        { 1, 2, 2 } }) );

    // Test zero'd SDR.
    a.setSparse(SDR_sparse_t( { } ));
    ASSERT_EQ( a.getCoordinates(), vector<vector<UInt>>({{}, {}}) );
}

TEST(SdrTest, TestGetCoordinatesFromDense) {
    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto dense = a.getDense();
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ( a.getCoordinates(), vector<vector<UInt>>({
        { 1, 2 },
        { 2, 2 }}) );

    // Test zero'd SDR.
    a.setDense( vector<Byte>(a.size, 0) );
    ASSERT_EQ( a.getCoordinates()[0].size(), 0ul );
    ASSERT_EQ( a.getCoordinates()[1].size(), 0ul );
}

TEST(SdrTest, TestAt) {
    SDR a({3, 3});
    a.setSparse(SDR_sparse_t( {4, 5, 8} ));
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
    str << a;
    // Use find so that trailing whitespace differences on windows/unix don't break it.
    ASSERT_NE( str.str().find( "SDR( 100 )" ), std::string::npos);

    stringstream str2;
    SDR b({ 9, 8 });
    str2 << b;
    ASSERT_NE( str2.str().find( "SDR( 9, 8 )" ), std::string::npos);

    stringstream str3;
    SDR sdr3({ 3, 3 });
    sdr3.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
    str3 << sdr3;
    ASSERT_NE( str3.str().find( "SDR( 3, 3 ) 1, 4, 8" ), std::string::npos);

    // Check that default aruments don't crash.
    cout << "PRINTING \"SDR( 3, 3 ) 1, 4, 8\" TO STDOUT: ";
    cout << sdr3;
}

TEST(SdrTest, TestGetOverlap) {
    SDR a({3, 3});
    a.setDense(SDR_dense_t({1, 1, 1, 1, 1, 1, 1, 1, 1}));
    SDR b(a);
    ASSERT_EQ( a.getOverlap( b ), 9ul );
    b.zero();
    ASSERT_EQ( a.getOverlap( b ), 0ul );
    b.setDense(SDR_dense_t({0, 1, 0, 0, 1, 0, 0, 0, 1}));
    ASSERT_EQ( a.getOverlap( b ), 3ul );
    a.zero(); b.zero();
    ASSERT_EQ( a.getOverlap( b ), 0ul );
}

TEST(SdrTest, TestRandomize) {
    // Test sparsity is OK
    SDR a({1000});
    a.randomize( 0. );
    ASSERT_EQ( a.getSum(), 0ul );
    a.randomize( .25 );
    ASSERT_EQ( a.getSum(), 250ul );
    a.randomize( .5 );
    ASSERT_EQ( a.getSum(), 500ul );
    a.randomize( .75 );
    ASSERT_EQ( a.getSum(), 750ul );
    a.randomize( 1. );
    ASSERT_EQ( a.getSum(), 1000ul );
    // Test RNG is deterministic
    SDR b(a);
    Random rng(77);
    Random rng2(77);
    a.randomize( 0.02f, rng );
    b.randomize( 0.02f, rng2 );
    ASSERT_TRUE( a == b);
    // Test different random number generators have different results.
    Random rng3( 1 );
    Random rng4( 2 );
    a.randomize( 0.02f, rng3 );
    b.randomize( 0.02f, rng4 );
    ASSERT_TRUE( a != b);
    // Test that this modifies RNG state and will generate different
    // distributions with the same RNG.
    Random rng5( 88 );
    a.randomize( 0.02f, rng5 );
    b.randomize( 0.02f, rng5 );
    ASSERT_TRUE( a != b);
    // Test default RNG has a different result every time
    a.randomize( 0.02f );
    b.randomize( 0.02f );
    ASSERT_TRUE( a != b);
    // Methodically test by running it many times and checking for an even
    // activation frequency at every bit.
    SDR af_test({ 97 /* prime number */ });
    UInt iterations = 10000;
    Real sparsity   = .25f;
    vector<Real> af( af_test.size, 0 );
    for( UInt i = 0; i < iterations; i++ ) {
        af_test.randomize( sparsity );
        for( auto idx : af_test.getSparse() )
            af[ idx ] += 1;
    }
    for( auto f : af ) {
        f = f / iterations / sparsity;
        ASSERT_GT( f, 0.90f );
        ASSERT_LT( f, 1.10f );
    }
}

TEST(SdrTest, TestAddNoise) {
    SDR a({1000});
    a.randomize( 0.10f );
    SDR b(a);
    SDR c(a);
    // Test seed is deteministic
    b.setSDR(a);
    c.setSDR(a);
    Random b_rng( 44 );
    Random c_rng( 44 );
    b.addNoise( 0.5, b_rng );
    c.addNoise( 0.5, c_rng );
    ASSERT_TRUE( b == c );
    ASSERT_FALSE( a == b );
    // Test different seed generates different distributions
    b.setSDR(a);
    c.setSDR(a);
    Random rng1( 1 );
    Random rng2( 2 );
    b.addNoise( 0.5, rng1 );
    c.addNoise( 0.5, rng2 );
    ASSERT_TRUE( b != c );
    // Test addNoise changes PRNG state so two consequtive calls yeild different
    // results.
    Random prng( 55 );
    b.setSDR(a);
    b.addNoise( 0.5, prng );
    SDR b_cpy(b);
    b.setSDR(a);
    b.addNoise( 0.5, prng );
    ASSERT_TRUE( b_cpy != b );
    // Test default seed works ok
    b.setSDR(a);
    c.setSDR(a);
    b.addNoise( 0.5 );
    c.addNoise( 0.5 );
    ASSERT_TRUE( b != c );
    // Methodically test for every overlap.
    for( UInt x = 0; x <= 100; x++ ) {
        b.setSDR( a );
        b.addNoise( (Real)x / 100.0f );
        ASSERT_EQ( a.getOverlap( b ), 100 - x );
        ASSERT_EQ( b.getSum(), 100ul );
    }
}

TEST(SdrTest, TestIntersectionExampleUsage) {
    // Setup 2 SDRs to hold the inputs.
    SDR A({ 10 });
    SDR B({ 10 });
    SDR C({ 10 });
    A.setSparse(SDR_sparse_t{0, 1, 2, 3});
    B.setSparse(SDR_sparse_t      {2, 3, 4, 5});
    // Calculate the logical intersection
    C.intersection(A, B);
    ASSERT_EQ(C.getSparse(), SDR_sparse_t({2, 3}));
}

TEST(SdrTest, TestIntersection) {
    SDR A({1000});
    SDR B(A.dimensions);
    SDR X(A.dimensions);
    A.randomize(.5);
    B.randomize(.5);

    // Test basic functionality
    X.intersection(A, B);
    X.getDense();
    ASSERT_GT( X.getSparsity(), .25 / 2. );
    ASSERT_LT( X.getSparsity(), .25 * 2. );
    A.zero();
    X.intersection(A, B);
    ASSERT_EQ( X.getSum(), 0u );
}

TEST(SdrTest, TestUnionExampleUsage) {
    // Setup 2 SDRs to hold the inputs.
    SDR A({ 10 });
    SDR B({ 10 });
    SDR C({ 10 });
    A.setSparse(SDR_sparse_t{0, 1, 2, 3});
    B.setSparse(SDR_sparse_t      {2, 3, 4, 5});
    // Calculate the logical union
    C.set_union(A, B);
    ASSERT_EQ(C.getSparse(), SDR_sparse_t({0, 1, 2, 3, 4, 5}));
}

TEST(SdrTest, TestUnion) {
    SDR A({1000});
    SDR B(A.dimensions);
    SDR U(A.dimensions);
    A.randomize(.5);
    B.randomize(.5);

    // Test basic functionality
    U.set_union(A, B);
    U.getDense();
    ASSERT_GT( U.getSparsity(), 1 - .25 * 2. );
    ASSERT_LT( U.getSparsity(), 1 - .25 / 2. );
    A.zero();
    U.set_union(A, B);
    ASSERT_EQ( U.getSparsity(), .5 );
}

TEST(SdrTest, TestConcatenationExampleUsage) {
    SDR A({ 10 });
    SDR B({ 10 });
    SDR C({ 20 });
    A.setSparse(SDR_sparse_t{ 0, 1, 2 });
    B.setSparse(SDR_sparse_t{ 0, 1, 2 });
    C.concatenate( A, B );
    ASSERT_EQ(C.getSparse(), SDR_sparse_t({0, 1, 2, 10, 11, 12}));
}

TEST(SdrTest, TestConcatenation) {
    SDR A({10});
    SDR B({10});
    SDR C({20});
    SDR D({20});
    SDR E({30});
    A.randomize( 0.30f );
    B.randomize( 0.70f );
    
    //concat 2 members
    EXPECT_NO_THROW(C.concatenate( A, B ));
    ASSERT_EQ( C.getSum(), 10u );

    //concat a vector
    EXPECT_NO_THROW(D.concatenate({&A, &B}));
    ASSERT_EQ( D.getSum(), 10u );

    //concat explicit vector
    std::vector<const SDR*> AB;
    AB.push_back(&A);
    AB.push_back(&B);
    AB.push_back(&A);
    EXPECT_NO_THROW(E.concatenate(AB));
    ASSERT_EQ(E.getSum(), 13u);
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
    test_cases.back()->setSparse(SDR_sparse_t({0,}));
    test_cases.push_back( new SDR({ 3, 3 }));
    test_cases.back()->setSparse(SDR_sparse_t({3, 4, 6}));

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

    SDR zero({ 3, 3 });
    SDR dense({ 3, 3 });
    SDR sparse({ 3, 3 });
    SDR coord({ 3, 3 });
    {
      cereal::JSONOutputArchive json_out(outfile);

      // Test zero value
      zero.save_ar( json_out );

      // Test dense data
      dense.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
      dense.save_ar( json_out );

      // Test sparse data
      sparse.setSparse(SDR_sparse_t({ 1, 4, 8 }));
      sparse.save_ar( json_out );

      // Test coordinate data
      coord.setCoordinates(SDR_coordinate_t({
              { 0, 1, 2 },
              { 1, 1, 2 }}));
      coord.save_ar( json_out );
    } // forces Cereal to flush to stream.

    // Now load all of the data back into SDRs.
    outfile.close();
    ifstream infile( filename );

    if( verbose ) {
        // Print the file's contents
        std::stringstream buffer; buffer << infile.rdbuf();
        VERBOSE << buffer.str() << "EOF" << endl;
        infile.seekg( 0 ); // rewind to start of file.
    }

    cereal::JSONInputArchive json_in(infile);

    SDR zero_2;
    zero_2.load_ar( json_in );
    SDR dense_2;
    dense_2.load_ar( json_in );
    SDR sparse_2;
    sparse_2.load_ar( json_in );
    SDR coord_2;
    coord_2.load_ar( json_in );
    
    infile.close();
    int ret = ::remove( filename );
    ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

    // Check that all of the data is OK
    ASSERT_TRUE( zero    == zero_2 );
    ASSERT_TRUE( dense   == dense_2 );
    ASSERT_TRUE( sparse  == sparse_2 );
    ASSERT_TRUE( coord   == coord_2 );

    dense.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
    stringstream ss;
    dense.save(ss, SerializableFormat::BINARY);
    dense_2.load(ss, SerializableFormat::BINARY);
    ASSERT_TRUE( dense   == dense_2 );

}

TEST(SdrTest, TestCallbacks) {

    SDR A({ 10, 20 });
    // Add and remove these callbacks a bunch of times, and then check they're
    // called the correct number of times.
    int count1 = 0;
    SDR_callback_t call1 = [&](){ count1++; };
    int count2 = 0;
    SDR_callback_t call2 = [&](){ count2++; };
    int count3 = 0;
    SDR_callback_t call3 = [&](){ count3++; };

    A.zero();   // No effect on callbacks
    A.zero();   // No effect on callbacks
    A.zero();   // No effect on callbacks

    UInt handle1 = A.addCallback( call1 );
    UInt handle2 = A.addCallback( call2 );
    UInt handle3 = A.addCallback( call3 );

    // Remove call 2 and add it back in.
    A.removeCallback( handle2 );
    A.zero();
    handle2 = A.addCallback( call2 );

    A.zero();
    ASSERT_EQ( count1, 2 );
    ASSERT_EQ( count2, 1 );
    ASSERT_EQ( count3, 2 );

    // Remove call 1
    A.removeCallback( handle1 );
    A.zero();
    ASSERT_EQ( count1, 2 );
    ASSERT_EQ( count2, 2 );
    ASSERT_EQ( count3, 3 );

    UInt handle2_2 = A.addCallback( call2 );
    UInt handle2_3 = A.addCallback( call2 );
    UInt handle2_4 = A.addCallback( call2 );
    UInt handle2_5 = A.addCallback( call2 );
    UInt handle2_6 = A.addCallback( call2 );
    UInt handle2_7 = A.addCallback( call2 );
    UInt handle2_8 = A.addCallback( call2 );
    A.zero();
    ASSERT_EQ( count1, 2 );
    ASSERT_EQ( count2, 10 );
    ASSERT_EQ( count3, 4 );

    A.removeCallback( handle2_2 );
    A.removeCallback( handle2_3 );
    A.removeCallback( handle2_4 );
    A.removeCallback( handle2_7 );
    A.removeCallback( handle2_6 );
    A.removeCallback( handle2_5 );
    A.removeCallback( handle2_8 );
    A.removeCallback( handle3 );
    A.removeCallback( handle2 );

    // Test removing junk handles.
    ASSERT_ANY_THROW( A.removeCallback( 99 ) );
    // Test callbacks are not copied.
    handle1 = A.addCallback( call1 );
    SDR B(A);
    ASSERT_ANY_THROW( B.removeCallback( handle1 ) );
    ASSERT_ANY_THROW( B.removeCallback( 0 ) );
}


TEST(SdrTest, TestAssignmentOperator) 
{
  SDR a({10, 10});
  a.setSparse<UInt>({1, 3, 5, 7});
  SDR copy; //notice the no dimensions
  ASSERT_EQ(copy.dimensions.size(), 0u);
  EXPECT_NO_THROW(copy = a);
  EXPECT_EQ(a, copy);
  EXPECT_EQ(a.getSparse(), copy.getSparse());
  EXPECT_EQ(a.dimensions, copy.dimensions);
}

} // End namespace testing
