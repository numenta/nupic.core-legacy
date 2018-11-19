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

typedef vector<Byte> dense_vec;

void ASSERT_SDR_NO_VALUE(SDR &sdr) {
    EXPECT_FALSE( sdr.dense_valid );
    EXPECT_FALSE( sdr.flatIndex_valid );
    EXPECT_FALSE( sdr.index_valid );
    EXPECT_ANY_THROW( sdr.getDense() );
    EXPECT_ANY_THROW( sdr.getFlatIndex() );
    EXPECT_ANY_THROW( sdr.getIndex() );
}

void ASSERT_SDR_HAS_VALUE(SDR &sdr) {
    sdr.getDense();
    sdr.getFlatIndex();
    sdr.getIndex();
    EXPECT_TRUE( sdr.dense_valid );
    EXPECT_TRUE( sdr.flatIndex_valid );
    EXPECT_TRUE( sdr.index_valid );
}

/* This also tests the size and dimensions are correct */
TEST(SdrTest, TestConstructor) {
    // Test 0-D
    vector<UInt> a_dims = {};
    SDR a(a_dims);
    ASSERT_EQ( a.size, 0 );
    ASSERT_EQ( a.dimensions, a_dims );
    ASSERT_SDR_NO_VALUE(a);
    a.zero();
    ASSERT_EQ( a.getIndex().size(), 0 );

    // Test 1-D
    vector<UInt> b_dims = {3};
    SDR b(b_dims);
    ASSERT_EQ( b.size, 3 );
    ASSERT_EQ( b.dimensions, b_dims );
    ASSERT_SDR_NO_VALUE(b);
    b.zero();
    ASSERT_EQ( b.getIndex().size(), 1 );

    // Test 3-D
    vector<UInt> c_dims = {11, 15, 3};
    SDR c(c_dims);
    ASSERT_EQ( c.size, 11 * 15 * 3 );
    ASSERT_EQ( c.dimensions, c_dims );
    ASSERT_SDR_NO_VALUE(c);
    c.zero();
    ASSERT_EQ( c.getIndex().size(), 3 );

    // Test 4-D w/ zero size
    vector<UInt> d_dims = {10, 20, 0, 3};
    SDR d(d_dims);
    ASSERT_EQ( d.size, 0 );
    ASSERT_EQ( d.dimensions, d_dims );
    ASSERT_SDR_NO_VALUE(d);
    d.zero();
    ASSERT_EQ( d.getIndex().size(), 4 );
    // Test dimensions are copied not referenced
    d_dims.push_back(7);
    vector<UInt> copy({10, 20, 0, 3});
    ASSERT_EQ( d.dimensions, copy );

    // Test default arguments
    SDR x;
    ASSERT_EQ( x.size, 0 );
    ASSERT_EQ( x.dimensions.size(), 0 );
    ASSERT_SDR_NO_VALUE(x);
}

TEST(DISABLED_SdrTest, TestConstructorCopy) {
    // Test value/no-value is preserved
    // SDR x({100});
    // SDR x_copy(x);
    // ASSERT_SDR_NO_VALUE( x_copy );
    // x.zero();
    // ASSERT_SDR_NO_VALUE( x_copy );
    // SDR X_copy = SDR(x);
    // ASSERT_SDR_HAS_VALUE( x_copy );

    // // Test simple zero data
    // SDR a({5, 5});
    // a.zero();
    // SDR b(a);

    // const Byte *raw = a.getDense();

    // Test larger SDR w/ data
}

TEST(SdrTest, TestClear) {
    SDR a({32, 32});
    ASSERT_SDR_NO_VALUE(a);
    a.zero();
    ASSERT_SDR_HAS_VALUE(a);
    a.clear();
    ASSERT_SDR_NO_VALUE(a);
}

TEST(SdrTest, TestZero) {
    SDR a({4, 4});
    a.setDense( vector<Byte>(16, 1) );
    a.zero();
    ASSERT_EQ( vector<Byte>(16, 0), a.getDense() );
    ASSERT_EQ( a.getFlatIndex().size(),  0);
    ASSERT_EQ( a.getIndex().size(),  2);
    ASSERT_EQ( a.getIndex().at(0).size(),  0);
    ASSERT_EQ( a.getIndex().at(1).size(),  0);
}

TEST(SdrTest, TestSetDenseVec) {
    SDR a({11, 10, 4});
    auto vec = vector<Byte>(440, 1);
    a.zero();
    a.setDense( vec );
    ASSERT_EQ( a.getDense(), vec );
    ASSERT_NE( a.getDense().data(), vec.data() ); // true copy not a reference
    // Test size zero doesn't crash
    SDR b( vector<UInt>(0) );
    b.setDense( vector<Byte>(0) );
    SDR c( {11, 0} );
    c.setDense( vector<Byte>(0) );
}

TEST(SdrTest, TestSetDenseByte) {
    SDR a({11, 10, 4});
    auto vec = vector<Byte>(a.size, 1);
    a.zero();
    a.setDense( (Byte*) vec.data());
    ASSERT_EQ( a.getDense(), vec );
    ASSERT_NE( ((vector<Byte>&) a.getDense()).data(), vec.data() ); // true copy not a reference
    // Test size zero doesn't crash
    SDR b( vector<UInt>(0) );
    Byte data;
    b.setDense( (Byte*) &data );
    SDR c( {11, 0} );
    c.setDense( (Byte*) &data );
}

TEST(SdrTest, TestSetDenseUInt) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    a.zero();
    a.setDense( (UInt*) vec.data() );
    ASSERT_EQ( a.getDense(), vector<Byte>(a.size, 1) );
    ASSERT_NE( a.getDense().data(), (const char*) vec.data()); // true copy not a reference
    // Test size zero doesn't crash
    SDR b( vector<UInt>(0) );
    UInt data;
    b.setDense( (Byte*) &data );
    SDR c( {11, 0} );
    c.setDense( (Byte*) &data );
}

TEST(DISABLED_SdrTest, TestSetDenseArray) {
    // Overload is not implemented ...
}

TEST(SdrTest, TestSetDenseMutableInplace) {
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getDenseMutable();
    ASSERT_EQ( a_data, dense_vec(100, 0) );
    a.clear();
    ASSERT_SDR_NO_VALUE(a);
    a_data.assign( a.size, 1 );
    a.setDenseInplace();
    ASSERT_SDR_HAS_VALUE(a);
    ASSERT_EQ( a.getDense().data(), a.getDenseMutable().data() );
    ASSERT_EQ( a.getDense().data(), a_data.data() );
    ASSERT_EQ( a.getDense(), dense_vec(a.size, 1) );
    ASSERT_EQ( a.getDense(), a_data );
}

TEST(SdrTest, TestSetFlatIndexVec) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.zero();
    a.setFlatIndex( vec );
    ASSERT_EQ( a.getFlatIndex(), vec );
    ASSERT_NE( a.getFlatIndex().data(), vec.data()); // true copy not a reference
    // Test size zero doesn't crash
    SDR b( vector<UInt>(0) );
    b.setDense( vector<Byte>(0) );
    SDR c( {11, 0} );
    c.setDense( vector<Byte>(0) );   
}

TEST(SdrTest, TestSetFlatIndexPtr) {
    SDR a({11, 10, 4});
    auto vec = vector<UInt>(a.size, 1);
    for(UInt i = 0; i < a.size; i++)
        vec[i] = i;
    a.zero();
    a.setFlatIndex( (UInt*) vec.data(), a.size );
    ASSERT_EQ( a.getFlatIndex(), vec );
    ASSERT_NE( a.getFlatIndex().data(), vec.data()); // true copy not a reference
    // Test size zero doesn't crash
    SDR b( vector<UInt>(0) );
    b.setDense( vector<Byte>(0) );
    SDR c( {11, 0} );
    c.setDense( vector<Byte>(0) );
}

TEST(SdrTest, TestSetFlatIndexMutableInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
    SDR a({10, 10});
    a.zero();
    auto& a_data = a.getFlatIndexMutable();
    ASSERT_EQ( a_data, vector<UInt>(0) );
    a.clear();
    ASSERT_SDR_NO_VALUE(a);
    a_data.push_back(0);
    a.setFlatIndexInplace();
    ASSERT_SDR_HAS_VALUE(a);
    ASSERT_EQ( a.getFlatIndex().data(), a.getFlatIndexMutable().data() );
    ASSERT_EQ( a.getFlatIndex(),        a.getFlatIndexMutable() );
    ASSERT_EQ( a.getFlatIndex().data(), a_data.data() );
    ASSERT_EQ( a.getFlatIndex(),        a_data );
    ASSERT_EQ( a.getFlatIndex(), vector<UInt>(1) );
    a_data.clear();
    a.setFlatIndexInplace();
    ASSERT_EQ( a.getDense(), dense_vec(a.size, 0) );
}

TEST(DISABLED_SdrTest, TestSetIndex) {}

TEST(DISABLED_SdrTest, TestSetIndexMutableInplace) {
    // Test both mutable & inplace methods at the same time, which is the intended use case.
}

TEST(DISABLED_SdrTest, TestSetSDR) {
    // This method has many code paths...
}

TEST(SdrTest, TestGetDenseFromFlatIndex) {
    // Test zeros
    SDR z({4, 4});
    z.setFlatIndex({});
    ASSERT_EQ( z.getDense(), vector<Byte>(16, 0) );

    // Test ones
    SDR nz({4, 4});
    nz.setFlatIndex({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    ASSERT_EQ( nz.getDense(), vector<Byte>(16, 1) );

    // Test 0-D
    SDR foobar;
    foobar.setFlatIndex({});
    ASSERT_EQ( foobar.getDense(), vector<Byte>(0) );

    // Test 1-D
    SDR d1({30});
    d1.setFlatIndex({1, 29, 4, 5, 7});
    vector<Byte> ans(30, 0);
    ans[1] = 1;
    ans[29] = 1;
    ans[4] = 1;
    ans[5] = 1;
    ans[7] = 1;
    ASSERT_EQ( d1.getDense(), ans );

    // Test 3-D
    SDR d3({10, 10, 10});
    d3.setFlatIndex({0, 5, 50, 55, 500, 550, 555, 999});
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

TEST(SdrTest, TestGetDenseFromIndex) {
    // Test simple 2-D
    SDR a({3, 3});
    a.setIndex({{1, 0, 2}, {2, 0, 2}});
    vector<Byte> ans(9, 0);
    ans[0] = 1;
    ans[5] = 1;
    ans[8] = 1;
    ASSERT_EQ( a.getDense(), ans );

    // Test zeros
    SDR z({99, 1});
    z.setIndex({{}, {}});
    ASSERT_EQ( z.getDense(), vector<Byte>(99, 0) );
}

TEST(SdrTest, TestGetFlatIndexFromDense) {
    // Test zero sized SDR.
    SDR z;
    Byte data;
    z.setDense( (Byte*) &data );
    ASSERT_EQ( z.getFlatIndex().size(), 0 );

    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto dense = a.getDenseMutable();
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ(a.getFlatIndex().at(0), 5);
    ASSERT_EQ(a.getFlatIndex().at(1), 8);

    // Test zero'd SDR.
    a.setDense( vector<Byte>(a.size, 0) );
    ASSERT_EQ( a.getFlatIndex().size(), 0 );
}

TEST(SdrTest, TestGetFlatIndexFromIndex) {
    // Test zero sized SDR.
    SDR z;
    z.setIndex( { } );
    ASSERT_EQ( z.getFlatIndex().size(), 0 );

    // Test simple 2-D SDR.
    SDR a({3, 3}); a.zero();
    auto& index = a.getIndexMutable();
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
    a.setIndexInplace();
    ASSERT_EQ(a.getFlatIndex().at(0), 4);
    ASSERT_EQ(a.getFlatIndex().at(1), 8);
    ASSERT_EQ(a.getFlatIndex().at(2), 5);

    // Test zero'd SDR.
    a.setIndex( {{}, {}} );
    ASSERT_EQ( a.getFlatIndex().size(), 0 );
}

TEST(DISABLED_SdrTest, TestGetIndexFromFlat) {}

TEST(DISABLED_SdrTest, TestGetIndexFromDense) {}

TEST(SdrTest, TestAt) {
    SDR a({3, 3});
    a.setFlatIndex( {4, 5, 8} );
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
    a.zero();
    auto& dense = a.getDenseMutable();
    for(UInt i = 0; i < a.size; i++) {
        ASSERT_EQ( i, a.getSum() );
        EXPECT_FLOAT_EQ( (Real) i / a.size, a.getSparsity() );
        dense[i] = 1;
        a.setDenseInplace();
    }
    ASSERT_EQ( a.size, a.getSum() );
    ASSERT_FLOAT_EQ( 1, a.getSparsity() );
}

TEST(DISABLED_SdrTest, TestOverlap) {
    // TODO: THIS METHOD IS NOT IN HEADER
    // TODO: THIS METHOD IS NOT IN IMPLEMENTATION
}

TEST(DISABLED_SdrTest, TestRandomize) {}

TEST(DISABLED_SdrTest, TestAddNoise) {}

TEST(DISABLED_SdrTest, TestSaveLoad) {}

TEST(DISABLED_SdrTest, TestCallbacks) {}
