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

void ASSERT_SDR_NO_VALUE(SDR &sdr) {
    ASSERT_ANY_THROW( sdr.getDense() );
    ASSERT_ANY_THROW( sdr.getFlatIndex() );
    ASSERT_ANY_THROW( sdr.getIndex() );
}

void ASSERT_SDR_HAS_VALUE(SDR &sdr) {
    sdr.getDense();
    sdr.getFlatIndex();
    sdr.getIndex();
}

TEST(SdrTest, TestConstructorNoArgs) {
    SDR a;
    ASSERT_EQ( a.getSize(), 0 );
    ASSERT_EQ( a.getDimensions().size(), 0 );
    ASSERT_SDR_NO_VALUE(a);
}

TEST(SdrTest, TestConstructor) {
    // Test 0-D
    vector<UInt> a_dims = {};
    SDR a(a_dims);
    ASSERT_EQ( a.getSize(), 0 );
    ASSERT_EQ( a.getDimensions(), a_dims );
    ASSERT_SDR_NO_VALUE(a);

    // Test 1-D
    vector<UInt> b_dims = {3};
    SDR b(b_dims);
    ASSERT_EQ( b.getSize(), 3 );
    ASSERT_EQ( b.getDimensions(), b_dims );
    ASSERT_SDR_NO_VALUE(b);

    // Test 3-D
    vector<UInt> c_dims = {11, 15, 3};
    SDR c(c_dims);
    ASSERT_EQ( c.getSize(), 11 * 15 * 3 );
    ASSERT_EQ( c.getDimensions(), c_dims );
    ASSERT_SDR_NO_VALUE(c);

    // Test 4-D w/ zero size
    vector<UInt> d_dims = {10, 20, 0, 3};
    SDR d(d_dims);
    ASSERT_EQ( d.getSize(), 0 );
    ASSERT_EQ( d.getDimensions(), d_dims );
    ASSERT_SDR_NO_VALUE(d);
}

TEST(DISABLED_SdrTest, TestConstructorCopy) {
    // Test value/no-value is preserved
    SDR x({100});
    SDR x_copy(x);
    ASSERT_SDR_NO_VALUE( x_copy );
    x.zero();
    ASSERT_SDR_NO_VALUE( x_copy );
    SDR X_copy = SDR(x);
    ASSERT_SDR_HAS_VALUE( x_copy );

    // Test simple zero data
    SDR a({5, 5});
    a.zero();
    SDR b(a);

    const Byte *raw = a.getDense()->data();

    // Test larger SDR w/ data

}

TEST(SdrTest, TestGetDimensions) {
    // Test 0-D
    vector<UInt> a_dims = {};
    SDR a(a_dims);
    ASSERT_EQ(a.getDimensions(), a_dims);

    // Test 1-D
    vector<UInt> b_dims = {77};
    SDR b(b_dims);
    ASSERT_EQ(b.getDimensions(), b_dims);

    // Test 7-D
    vector<UInt> c_dims = {3,4,5,6,7,8,9};
    SDR c(c_dims);
    ASSERT_EQ(c.getDimensions(), c_dims);
}

TEST(SdrTest, TestGetSize) {
    // Test 0-D
    SDR a;
    ASSERT_EQ(a.getSize(), 0);

    // Test 1-D
    SDR b({77});
    ASSERT_EQ(b.getSize(), 77);

    // Test 7-D
    SDR c({3,4,5,6,7,8,9});
    ASSERT_EQ(c.getSize(), 181440);
}

TEST(SdrTest, TestClear) {
    SDR a({32, 32});
    a.zero();
    a.getDense();
    a.getFlatIndex();
    a.getIndex();
    a.clear();
    ASSERT_SDR_NO_VALUE(a);
}

TEST(SdrTest, TestZero) {
    SDR a({4, 4});
    vector<Byte> dense(16, 1);
    a.setDense( dense );
    a.zero();
    ASSERT_EQ( *a.getDense(),  vector<Byte>(16, 0));
    ASSERT_EQ( a.getFlatIndex()->size(),  0);
    ASSERT_EQ( a.getIndex()->size(),  2);
    ASSERT_EQ( a.getIndex()->at(0).size(),  0);
    ASSERT_EQ( a.getIndex()->at(1).size(),  0);
}

TEST(DISABLED_SdrTest, TestSetDenseVec) {}

TEST(DISABLED_SdrTest, TestSetDensePtr) {}

TEST(DISABLED_SdrTest, TestSetDenseArray) {}

TEST(DISABLED_SdrTest, TestSetFlatIndexVec) {}

TEST(DISABLED_SdrTest, TestSetFlatIndexPtr) {}

TEST(DISABLED_SdrTest, TestSetIndex) {}

TEST(DISABLED_SdrTest, TestAssign) {
    // This method has many code paths...
}

TEST(SdrTest, TestGetDenseFromFlatIndex) {
    // Test zeros
    SDR z({4, 4});
    z.setFlatIndex({});
    ASSERT_EQ( *z.getDense(), vector<Byte>(16, 0) );

    // Test ones
    SDR nz({4, 4});
    nz.setFlatIndex({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    ASSERT_EQ( *nz.getDense(), vector<Byte>(16, 1) );

    // Test 0-D
    SDR foobar;
    foobar.setFlatIndex({});
    ASSERT_EQ( *foobar.getDense(), vector<Byte>(0) );

    // Test 1-D
    SDR d1({30});
    d1.setFlatIndex({1, 29, 4, 5, 7});
    vector<Byte> ans(30, 0);
    ans[1] = 1;
    ans[29] = 1;
    ans[4] = 1;
    ans[5] = 1;
    ans[7] = 1;
    ASSERT_EQ( *d1.getDense(), ans );

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
    ASSERT_EQ( *d3.getDense(), ans2 );
}

TEST(SdrTest, TestGetDenseFromIndex) {
    // Test simple 2-D
    SDR a({3, 3});
    a.setIndex({{1, 0, 2}, {2, 0, 2}});
    vector<Byte> ans(9, 0);
    ans[0] = 1;
    ans[5] = 1;
    ans[8] = 1;
    ASSERT_EQ( *a.getDense(), ans );

    // Test zeros
    SDR z({99, 1});
    z.setIndex({{}, {}});
    ASSERT_EQ( *z.getDense(), vector<Byte>(99, 0) );
}

TEST(SdrTest, TestGetFlatIndexFromDense) {
    // Test zero sized SDR.
    FAIL();

    // Test simple 1-D SDR.
    FAIL();

    // Test simple 2-D SDR.
    SDR a({3, 3});
    vector<Byte> dense(9, 0);
    dense[5] = 1;
    dense[8] = 1;
    a.setDense(dense);
    ASSERT_EQ(a.getFlatIndex()->at(0), 5);
    ASSERT_EQ(a.getFlatIndex()->at(1), 8);

    // Test zero'd SDR.
    FAIL();

    // Test converting random SDRs.
    FAIL();
}

TEST(DISABLED_SdrTest, TestGetFlatIndexFromIndex) {}

TEST(DISABLED_SdrTest, TestGetIndex) {}

TEST(DISABLED_SdrTest, TestAt) {}

TEST(DISABLED_SdrTest, TestCopy) {}

TEST(SdrTest, TestSparsity) {
    SDR a({31, 17, 3});
    a.zero();
    ASSERT_FLOAT_EQ( 0, a.getSparsity() );
    auto dense = vector<Byte>(*a.getDense());
    for(UInt i = 0; i < a.getSize(); i++) {
        dense[i] = 1;
        a.setDense( dense );
        ASSERT_FLOAT_EQ( (Real) (i + 1) / a.getSize(), a.getSparsity() );
    }
}

TEST(DISABLED_SdrTest, TestOverlap) {
    // TODO: THIS METHOD IS NOT IN HEADER
    // TODO: THIS METHOD IS NOT IN IMPLEMENTATION
}

TEST(DISABLED_SdrTest, TestSum) {
    // TODO: THIS METHOD IS NOT IN HEADER
    // TODO: THIS METHOD IS NOT IN IMPLEMENTATION
}

TEST(DISABLED_SdrTest, TestRandomize) {}

TEST(DISABLED_SdrTest, TestAddNoise) {}

TEST(DISABLED_SdrTest, TestSave) {}

TEST(DISABLED_SdrTest, TestLoad) {}
