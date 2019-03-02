/* ---------------------------------------------------------------------
 * Copyright (C) 2019, David McDougall.
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

#include <vector>
#include <gtest/gtest.h>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/SdrProxy.hpp>

using namespace std;
using namespace nupic;

TEST(SdrReshapeTest, TestReshapeExamples) {
    SDR         A(    { 4, 4 });
    SDR_Reshape B( A, { 8, 2 });
    A.setCoordinates(SDR_coordinate_t({{1, 1, 2}, {0, 1, 2}}));
    auto coords = B.getCoordinates();
    ASSERT_EQ(coords, SDR_coordinate_t({{2, 2, 5}, {0, 1, 0}}));
}

TEST(SdrReshapeTest, TestReshapeConstructor) {
    SDR           A({ 11 });
    SDR_Reshape   B( A );
    ASSERT_EQ( A.dimensions, B.dimensions );
    SDR_Reshape   C( A, { 11 });
    SDR           D({ 5, 4, 3, 2, 1 });
    SDR_Reshape   E( D, {1, 1, 1, 120, 1});
    SDR_Reshape   F( D, { 20, 6 });
    SDR_Reshape   X( (SDR&) F );

    // Test that proxies can be safely made and destroyed.
    SDR_Reshape *G = new SDR_Reshape( A );
    SDR_Reshape *H = new SDR_Reshape( A );
    SDR_Reshape *I = new SDR_Reshape( A );
    A.zero();
    H->getDense();
    delete H;
    I->getDense();
    A.zero();
    SDR_Reshape *J = new SDR_Reshape( A );
    J->getDense();
    SDR_Reshape *K = new SDR_Reshape( A );
    delete K;
    SDR_Reshape *L = new SDR_Reshape( A );
    L->getCoordinates();
    delete L;
    delete G;
    I->getCoordinates();
    delete I;
    delete J;
    A.getDense();

    // Test invalid dimensions
    ASSERT_ANY_THROW( new SDR_Reshape( A, {2, 5}) );
    ASSERT_ANY_THROW( new SDR_Reshape( A, {11, 0}) );
}

TEST(SdrReshapeTest, TestReshapeDeconstructor) {
    SDR       *A = new SDR({12});
    SDR_Reshape *B = new SDR_Reshape( *A );
    SDR_Reshape *C = new SDR_Reshape( *A, {3, 4} );
    SDR_Reshape *D = new SDR_Reshape( *C, {4, 3} );
    SDR_Reshape *E = new SDR_Reshape( *C, {2, 6} );
    D->getDense();
    E->getCoordinates();
    // Test subtree deletion
    delete C;
    ASSERT_ANY_THROW( D->getDense() );
    ASSERT_ANY_THROW( E->getCoordinates() );
    ASSERT_ANY_THROW( new SDR_Reshape( *E ) );
    delete D;
    // Test rest of tree is OK.
    B->getSparse();
    A->zero();
    B->getSparse();
    // Test delete root.
    delete A;
    ASSERT_ANY_THROW( B->getDense() );
    ASSERT_ANY_THROW( E->getCoordinates() );
    // Cleanup remaining Proxies.
    delete B;
    delete E;
}

TEST(SdrReshapeTest, TestReshapeThrows) {
    SDR A({10});
    SDR_Reshape B(A, {2, 5});
    SDR *C = &B;

    ASSERT_ANY_THROW( C->setDense( SDR_dense_t( 10, 1 ) ));
    ASSERT_ANY_THROW( C->setCoordinates( SDR_coordinate_t({ {0}, {0} }) ));
    ASSERT_ANY_THROW( C->setSparse( SDR_sparse_t({ 0, 1, 2 }) ));
    SDR X({10});
    ASSERT_ANY_THROW( C->setSDR( X ));
    ASSERT_ANY_THROW( C->randomize(0.10f) );
    ASSERT_ANY_THROW( C->addNoise(0.10f) );
}

TEST(SdrReshapeTest, TestReshapeGetters) {
    SDR A({ 2, 3 });
    SDR_Reshape B( A, { 3, 2 });
    SDR *C = &B;
    // Test getting dense
    A.setDense( SDR_dense_t({ 0, 1, 0, 0, 1, 0 }) );
    ASSERT_EQ( C->getDense(), SDR_dense_t({ 0, 1, 0, 0, 1, 0 }) );

    // Test getting flat sparse
    A.setCoordinates( SDR_coordinate_t({ {0, 1}, {0, 1} }));
    ASSERT_EQ( C->getCoordinates(), SDR_coordinate_t({ {0, 2}, {0, 0} }) );

    // Test getting sparse
    A.setSparse( SDR_sparse_t({ 2, 3 }));
    ASSERT_EQ( C->getSparse(), SDR_sparse_t({ 2, 3 }) );

    // Test getting sparse, a second time.
    A.setSparse( SDR_sparse_t({ 2, 3 }));
    ASSERT_EQ( C->getCoordinates(), SDR_coordinate_t({ {1, 1}, {0, 1} }) );

    // Test getting sparse, when the parent SDR already has sparse computed and
    // the dimensions are the same.
    A.zero();
    SDR_Reshape D( A );
    SDR *E = &D;
    A.setCoordinates( SDR_coordinate_t({ {0, 1}, {0, 1} }));
    ASSERT_EQ( E->getCoordinates(), SDR_coordinate_t({ {0, 1}, {0, 1} }) );
}

TEST(SdrReshapeTest, TestSaveLoad) {
    const char *filename = "SdrReshapeSerialization.tmp";
    ofstream outfile;
    outfile.open(filename);

    // Test zero value
    SDR zero({ 3, 3 });
    SDR_Reshape z( zero );
    z.save( outfile );

    // Test dense data
    SDR dense({ 3, 3 });
    SDR_Reshape d( dense );
    dense.setDense(SDR_dense_t({ 0, 1, 0, 0, 1, 0, 0, 0, 1 }));
    Serializable &ser = d;
    ser.save( outfile );

    // Test flat data
    SDR flat({ 3, 3 });
    SDR_Reshape f( flat );
    flat.setSparse(SDR_sparse_t({ 1, 4, 8 }));
    f.save( outfile );

    // Test index data
    SDR index({ 3, 3 });
    SDR_Reshape x( index );
    index.setCoordinates(SDR_coordinate_t({
            { 0, 1, 2 },
            { 1, 1, 2 }}));
    x.save( outfile );

    // Now load all of the data back into SDRs.
    outfile.close();
    ifstream infile( filename );

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
    EXPECT_TRUE(ret == 0) << "Failed to delete " << filename;

    // Check that all of the data is OK
    ASSERT_TRUE( zero    == zero_2 );
    ASSERT_TRUE( dense   == dense_2 );
    ASSERT_TRUE( flat    == flat_2 );
    ASSERT_TRUE( index   == index_2 );
}

