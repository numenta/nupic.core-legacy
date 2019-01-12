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
#include <nupic/ntypes/Sdr.hpp>
#include <nupic/ntypes/SdrProxy.hpp>

using namespace std;
using namespace nupic;

TEST(SdrTest, TestProxyExamples) {
    SDR       A(    { 4, 4 });
    SDR_Proxy B( A, { 8, 2 });
    A.setSparse(SDR_sparse_t({{1, 1, 2}, {0, 1, 2}}));
    auto sparse = B.getSparse();
    ASSERT_EQ(sparse, SDR_sparse_t({{2, 2, 5}, {0, 1, 0}}));
}

TEST(SdrTest, TestProxyConstructor) {
    SDR         A({ 11 });
    SDR_Proxy   B( A );
    ASSERT_EQ( A.dimensions, B.dimensions );
    SDR_Proxy   C( A, { 11 });
    SDR         D({ 5, 4, 3, 2, 1 });
    SDR_Proxy   E( D, {1, 1, 1, 120, 1});
    SDR_Proxy   F( D, { 20, 6 });

    // Test that proxies can be safely made and destroyed.
    SDR_Proxy *G = new SDR_Proxy( A );
    SDR_Proxy *H = new SDR_Proxy( A );
    SDR_Proxy *I = new SDR_Proxy( A );
    A.zero();
    H->getDense();
    delete H;
    I->getDense();
    A.zero();
    SDR_Proxy *J = new SDR_Proxy( A );
    J->getDense();
    SDR_Proxy *K = new SDR_Proxy( A );
    delete K;
    SDR_Proxy *L = new SDR_Proxy( A );
    L->getSparse();
    delete L;
    delete G;
    I->getSparse();
    delete I;
    delete J;
    A.getDense();

    // Test invalid dimensions
    ASSERT_ANY_THROW( new SDR_Proxy( A, {2, 5}) );
    ASSERT_ANY_THROW( new SDR_Proxy( A, {11, 0}) );
}

TEST(SdrTest, TestProxyDeconstructor) {
    SDR       *A = new SDR({12});
    SDR_Proxy *B = new SDR_Proxy( *A );
    SDR_Proxy *C = new SDR_Proxy( *A, {3, 4} );
    SDR_Proxy *D = new SDR_Proxy( *C, {4, 3} );
    SDR_Proxy *E = new SDR_Proxy( *C, {2, 6} );
    D->getDense();
    E->getSparse();
    // Test subtree deletion
    delete C;
    ASSERT_ANY_THROW( D->getDense() );
    ASSERT_ANY_THROW( E->getSparse() );
    ASSERT_ANY_THROW( new SDR_Proxy( *E ) );
    delete D;
    // Test rest of tree is OK.
    B->getFlatSparse();
    A->zero();
    B->getFlatSparse();
    // Test delete root.
    delete A;
    ASSERT_ANY_THROW( B->getDense() );
    ASSERT_ANY_THROW( E->getSparse() );
    // Cleanup remaining Proxies.
    delete B;
    delete E;
}

TEST(SdrTest, TestProxyThrows) {
    SDR A({10});
    SDR_Proxy B(A, {2, 5});
    SDR *C = &B;

    ASSERT_ANY_THROW( C->setDense( SDR_dense_t( 10, 1 ) ));
    ASSERT_ANY_THROW( C->setSparse( SDR_sparse_t({ {0}, {0} }) ));
    ASSERT_ANY_THROW( C->setFlatSparse( SDR_flatSparse_t({ 0, 1, 2 }) ));
    SDR X({10});
    ASSERT_ANY_THROW( C->setSDR( X ));
    ASSERT_ANY_THROW( C->randomize(0.10f) );
    ASSERT_ANY_THROW( C->addNoise(0.10f) );
}

TEST(SdrTest, TestProxyGetters) {
    SDR A({ 2, 3 });
    SDR_Proxy B( A, { 3, 2 });
    SDR *C = &B;
    // Test getting dense
    A.setDense( SDR_dense_t({ 0, 1, 0, 0, 1, 0 }) );
    ASSERT_EQ( C->getDense(), SDR_dense_t({ 0, 1, 0, 0, 1, 0 }) );

    // Test getting flat sparse
    A.setSparse( SDR_sparse_t({ {0, 1}, {0, 1} }));
    ASSERT_EQ( C->getSparse(), SDR_sparse_t({ {0, 2}, {0, 0} }) );

    // Test getting sparse
    A.setFlatSparse( SDR_flatSparse_t({ 2, 3 }));
    ASSERT_EQ( C->getFlatSparse(), SDR_flatSparse_t({ 2, 3 }) );

    // Test getting sparse, a second time.
    A.setFlatSparse( SDR_flatSparse_t({ 2, 3 }));
    ASSERT_EQ( C->getSparse(), SDR_sparse_t({ {1, 1}, {0, 1} }) );

    // Test getting sparse, when the parent SDR already has sparse computed and
    // the dimensions are the same.
    A.zero();
    SDR_Proxy D( A );
    SDR *E = &D;
    A.setSparse( SDR_sparse_t({ {0, 1}, {0, 1} }));
    ASSERT_EQ( E->getSparse(), SDR_sparse_t({ {0, 1}, {0, 1} }) );
}

