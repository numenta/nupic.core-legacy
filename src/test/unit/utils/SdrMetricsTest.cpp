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

#include <gtest/gtest.h>
#include <htm/types/Sdr.hpp>
#include <htm/utils/SdrMetrics.hpp>
#include <vector>
#include <random>

namespace testing { 

static bool verbose = false;
#define VERBOSE if(verbose) std::cerr << "[          ]"

using namespace std;
using namespace htm;

/**
 * Sparsity
 * Test that it creates & destroys, and that nothing crashes.
 */
TEST(SdrMetricsTest, TestSparsityConstruct) {
    SDR *A = new SDR({1});
    Sparsity S( *A, 1000u );
    ASSERT_ANY_THROW( Sparsity S( *A, 0u ) ); // Period > 0!
    A->zero();
    A->zero();
    A->zero();
    delete A; // Test use after freeing the parent SDR.
    S.min();
    S.max();
    S.mean();
    S.std();
    ASSERT_EQ( S.sparsity, 0.0f );
}

TEST(SdrMetricsTest, TestSparsityExample) {
    SDR A( { 1000u } );
    Sparsity B( A, 1000u );
    A.randomize( 0.01f );
    A.randomize( 0.15f );
    A.randomize( 0.05f );
    ASSERT_EQ( B.sparsity, 0.05f);
    ASSERT_EQ( B.min(),    0.01f);
    ASSERT_EQ( B.max(),    0.15f);
    ASSERT_NEAR( B.mean(),   0.07f, 0.005f );
    ASSERT_NEAR( B.std(),    0.06f, 0.005f );
}

/*
 * Sparsity
 * Verify that the initial 10 values of metric are OK.
 */
TEST(SdrMetricsTest, TestSparsityShortTerm) {
    SDR A({1});
    Real period = 10u;
    Real alpha  = 1.0f / period;
    Sparsity S( A, (UInt)period );

    A.setDense(SDR_dense_t{ 1 });
    ASSERT_FLOAT_EQ( S.sparsity, 1.0f );
    ASSERT_NEAR( S.min(),  1.0f, alpha );
    ASSERT_NEAR( S.max(),  1.0f, alpha );
    ASSERT_NEAR( S.mean(), 1.0f, alpha );
    ASSERT_NEAR( S.std(),  0.0f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_FLOAT_EQ( S.sparsity, 0.0f );
    ASSERT_NEAR( S.min(),  0.0f, alpha );
    ASSERT_NEAR( S.max(),  1.0f, alpha );
    ASSERT_NEAR( S.mean(), 1.0f / 2, alpha );
    ASSERT_NEAR( S.std(),  0.5f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_FLOAT_EQ( S.sparsity, 0.0f );
    ASSERT_NEAR( S.min(),  0.0f, alpha );
    ASSERT_NEAR( S.max(),  1.0f, alpha );
    ASSERT_NEAR( S.mean(), 1.0f / 3, alpha );
    // Standard deviation was computed in python with numpy.std([ 1, 0, 0 ])
    ASSERT_NEAR( S.std(),  0.47140452079103168f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 4, alpha );
    ASSERT_NEAR( S.std(),  0.4330127018922193f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 5, alpha );
    ASSERT_NEAR( S.std(),  0.40000000000000008f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 6, alpha );
    ASSERT_NEAR( S.std(),  0.372677996249965f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 7, alpha );
    ASSERT_NEAR( S.std(),  0.34992710611188266f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 8, alpha );
    ASSERT_NEAR( S.std(),  0.33071891388307384f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 9, alpha );
    ASSERT_NEAR( S.std(),  0.31426968052735443f, alpha );

    A.setDense(SDR_dense_t{ 0 });
    ASSERT_NEAR( S.mean(), 1.0f / 10, alpha );
    ASSERT_NEAR( S.std(),  0.30000000000000004f, alpha );
}

/*
 * Sparsity
 * Verify that the longer run values of the Sparsity metric are OK.
 * Test Protocol:
 *      instantaneous-sparsity = Sample random distribution
 *      for iteration in range( 1,000 ):
 *          SDR.randomize( instantaneous-sparsity )
 *      ASSERT_NEAR( SparsityMetric.mean(), true_mean )
 *      ASSERT_NEAR( SparsityMetric.std(),  true_std )
 */
TEST(SdrMetricsTest, TestSparsityLongTerm) {
    auto period     = 100u;
    auto iterations = 1000u;

    SDR A({1000u});
    Sparsity S( A, period );

    vector<Real> test_means{ 0.01f,  0.05f,  0.20f, 0.50f, 0.50f, 0.75f, 0.99f };
    vector<Real> test_stdev{ 0.001f, 0.025f, 0.10f, 0.33f, 0.01f, 0.15f, 0.01f };

    std::default_random_engine generator;
    for(auto test = 0u; test < test_means.size(); test++) {
        const auto mean = test_means[test];
        const auto stdv = test_stdev[test];
        auto dist = std::normal_distribution<float>(mean, stdv);
        for(UInt i = 0; i < iterations; i++) {
            Real sparsity;
            do {
                sparsity = dist( generator );
            } while( sparsity < 0.0f || sparsity > 1.0f);
            A.randomize( sparsity );
            EXPECT_NEAR( S.sparsity, sparsity, 0.501f / A.size );
        }
        EXPECT_NEAR( S.mean(), mean, stdv );
        EXPECT_NEAR( S.std(),  stdv, stdv / 2.0f );
    }
}

TEST(SdrMetricsTest, TestSparsityPrint) {
    // Test passes if it does not crash.  The exact strings are checked by
    // python unit tests.
    std::stringstream ss;
    SDR A({ 2000u });
    Sparsity S( A, 10u );

    A.randomize( 0.30f );
    A.randomize( 0.70f );
    ss << S;

    A.randomize( 0.123456789f );
    A.randomize( 1.0f - 0.123456789f );
    ss << S;
    ss << endl;
    VERBOSE << ss.str() << std::endl;
}

/**
 * ActivationFrequency
 * Test that it creates & destroys, and that no methods crash.
 */
TEST(SdrMetricsTest, TestAF_Construct) {
    // Test creating it.
    SDR *A = new SDR({ 5 });
    ActivationFrequency F( *A, 100 );
    ASSERT_ANY_THROW( ActivationFrequency F( *A, 0u ) ); // Period > 0!
    // Test nothing crashes with no data.
    F.min();
    F.mean();
    F.std();
    F.max();
    ASSERT_EQ( F.activationFrequency.size(), A->size );

    // Test with junk data.
    A->zero(); A->randomize( 0.5f ); A->randomize( 1.0f ); A->randomize( 0.5f );
    F.min();
    F.mean();
    F.std();
    F.max();
    ASSERT_EQ( F.activationFrequency.size(), A->size );

    // Test use after freeing parent SDR.
    auto A_size = A->size;
    delete A;
    F.min();
    F.mean();
    F.std();
    F.max();
    ASSERT_EQ( F.activationFrequency.size(), A_size );
}

/**
 * ActivationFrequency
 * Verify that the first few data points are ok.
 */
TEST(SdrMetricsTest, TestAF_Example) {
    SDR A({ 2u });
    ActivationFrequency F( A, 10u );

    A.setDense(SDR_dense_t{ 0, 0 });
    ASSERT_EQ( F.activationFrequency, vector<Real>({ 0.0f, 0.0f }));

    A.setDense(SDR_dense_t{ 1, 1 });
    ASSERT_EQ( F.activationFrequency, vector<Real>({ 0.5f, 0.5f }));

    A.setDense(SDR_dense_t{ 0, 1 });
    ASSERT_NEAR( F.activationFrequency[0], 0.3333333333333333f, 0.001f );
    ASSERT_NEAR( F.activationFrequency[1], 0.6666666666666666f, 0.001f );
    ASSERT_EQ( F.min(), F.activationFrequency[0] );
    ASSERT_EQ( F.max(), F.activationFrequency[1] );
    ASSERT_FLOAT_EQ( F.mean(), 0.5f );
    ASSERT_NEAR( F.std(), 0.16666666666666666f, 0.001f );
    ASSERT_NEAR( F.entropy(), 0.9182958340544896f, 0.001f );
}

/*
 * ActivationFrequency
 * Verify that the longer run values of this metric are OK.
 */
TEST(SdrMetricsTest, TestAF_LongTerm) {
    const auto period  =  1000u;
    const auto runtime = 10000u;
    SDR A({ 20u });
    ActivationFrequency F( A, period );


    vector<Real> test_sparsity{ 0.0f, 0.05f, 1.0f, 0.25f, 0.5f };

    for(const auto &sparsity : test_sparsity) {
        for(UInt i = 0; i < runtime; i++)
            A.randomize( sparsity );

        const auto epsilon = 0.10f;
        EXPECT_GT( F.min(), sparsity - epsilon );
        EXPECT_LT( F.max(), sparsity + epsilon );
        EXPECT_NEAR( F.mean(), sparsity, epsilon );
        EXPECT_NEAR( F.std(),  0.0f,     epsilon );
    }
}

TEST(SdrMetricsTest, TestAF_Entropy) {
    const auto size    = 1000u; // Num bits in SDR.
    const auto period  =  100u; // For activation frequency exp-rolling-avg
    const auto runtime = 1000u; // Train time for each scenario
    const auto tolerance = 0.02f;

    // Extact tests:
    // Test all zeros.
    SDR A({ size });
    ActivationFrequency F( A, period );
    A.zero();
    EXPECT_FLOAT_EQ( F.entropy(), 0.0f );

    // Test all ones.
    SDR B({ size });
    ActivationFrequency G( B, period );
    B.randomize( 1.0f );
    EXPECT_FLOAT_EQ( G.entropy(), 0.0f );

    // Probabilistic tests:
    // Start with random SDRs, verify 100% entropy. Then disable cells and
    // verify that the entropy decreases.  Disable cells by freezing their value
    // so that the resulting SDR keeps the same sparsity.  Progresively disable
    // more cells and verify that the entropy monotonically decreases.  Finally
    // verify 0% entropy when all cells are disabled.
    SDR C({ size });
    ActivationFrequency H( C, period );
    auto last_entropy = -1.0f;
    const UInt incr = size / 10u; // NOTE: This MUST divide perfectly, with no remainder!
    for(auto nbits_disabled = 0u; nbits_disabled <= size; nbits_disabled += incr) {
        for(auto i = 0u; i < runtime; i++) {
            SDR scratch({size});
            scratch.randomize( 0.05f );
            // Freeze bits, such that nbits remain alive.
            for(auto z = 0u; z < nbits_disabled; z++)
                scratch.getDense()[z] = C.getDense()[z];
            C.setDense( scratch.getDense() );
        }
        const auto entropy = H.entropy();
        if( nbits_disabled == 0u ) {
            // Expect 100% entropy
            EXPECT_GT( entropy, 1.0f - tolerance );
        }
        else{
            // Expect less entropy than last time, when fewer bits were disabled.
            ASSERT_LT( entropy, last_entropy );
        }
        last_entropy = entropy;
    }
    // Expect 0% entropy.
    EXPECT_LT( last_entropy, tolerance );
}

TEST(SdrMetricsTest, TestAF_Print) {
    // Test passes if it does not crash.  The exact strings are checked by
    // python unit tests.
    std::stringstream ss;
    const auto period  =  100u;
    const auto runtime = 1000u;
    SDR A({ 2000u });
    ActivationFrequency F( A, period );

    vector<Real> sparsity{ 0.0f, 0.02f, 0.05f, 0.50f, 0.0f };
    for(const auto sp : sparsity) {
        for(auto i = 0u; i < runtime; i++)
            A.randomize( sp );
        ss << F;
        ss << endl;
    }
    VERBOSE << ss.str() << std::endl;
}

TEST(SdrMetricsTest, TestOverlap_Construct) {
    SDR *A = new SDR({ 1000u });
    Overlap V( *A, 100u );
    ASSERT_ANY_THROW( new Overlap( *A, 0 ) ); // Period > 0!
    // Check that it doesn't crash, when uninitialized.
    V.min();
    V.mean();
    V.std();
    V.max();
    // If no data, have obviously wrong result.
    ASSERT_FALSE( V.overlap >= 0.0f and V.overlap <= 1.0f );

    // Check that it doesn't crash with half enough data.
    A->randomize( 0.20f );
    V.min();
    V.mean();
    V.std();
    V.max();
    ASSERT_FALSE( V.overlap >= 0.0f and V.overlap <= 1.0f );

    // Check no crash with data.
    A->addNoise( 0.50f );
    V.min();
    V.mean();
    V.std();
    V.max();
    ASSERT_EQ( V.overlap, 0.50f );

    // Check overlap metric is valid after parent SDR is deleted.
    delete A;
    V.min();
    V.mean();
    V.std();
    V.max();
    ASSERT_EQ( V.overlap, 0.50f );
}

TEST(SdrMetricsTest, TestOverlap_Example) {
    SDR A({ 10000u });
    Overlap B( A, 1000u );
    A.randomize( 0.05f );
    A.addNoise( 0.95f );         //   5% overlap
    A.addNoise( 0.55f );         //  45% overlap
    A.addNoise( 0.72f );         //  28% overlap
    ASSERT_EQ( B.overlap,  0.28f );
    ASSERT_EQ( B.min(),    0.05f );
    ASSERT_EQ( B.max(),    0.45f );
    ASSERT_NEAR( B.mean(), 0.26f, 0.005f );
    ASSERT_NEAR( B.std(),  0.16f, 0.005f );
}

TEST(SdrMetricsTest, TestOverlap_ShortTerm) {
    SDR     A({ 1000u });
    Overlap V( A, 10u );

    A.randomize( 0.20f ); // Initial value is taken after Overlap is created

    // Add overlap 50% to metric tracker.
    A.addNoise(  0.50f );
    ASSERT_FLOAT_EQ( V.overlap, 0.50f );
    ASSERT_FLOAT_EQ( V.min(),   0.50f );
    ASSERT_FLOAT_EQ( V.max(),   0.50f );
    ASSERT_FLOAT_EQ( V.mean(),  0.50f );
    ASSERT_FLOAT_EQ( V.std(),   0.0f );

    // Add overlap 80% to metric tracker.
    A.addNoise(  0.20f );
    ASSERT_FLOAT_EQ( V.overlap, 0.80f );
    ASSERT_FLOAT_EQ( V.min(),   0.50f );
    ASSERT_FLOAT_EQ( V.max(),   0.80f );
    ASSERT_FLOAT_EQ( V.mean(),  0.65f );
    ASSERT_FLOAT_EQ( V.std(),   0.15f );

    // Add overlap 25% to metric tracker.
    A.addNoise(  0.75f );
    ASSERT_FLOAT_EQ( V.overlap, 0.25f );
    ASSERT_FLOAT_EQ( V.min(),   0.25f );
    ASSERT_FLOAT_EQ( V.max(),   0.80f );
    ASSERT_FLOAT_EQ( V.mean(),  0.51666666666666672f ); // Source: python numpy.mean
    ASSERT_FLOAT_EQ( V.std(),   0.22484562605386735f ); // Source: python numpy.std
}

TEST(SdrMetricsTest, TestOverlap_LongTerm) {
    const auto runtime = 1000u;
    const auto period  =  100u;
    SDR A({ 500u });
    Overlap V( A, period );
    A.randomize( 0.45f );

    vector<Real> mean_ovlp{ 0.0f, 1.0f,
                            0.5f, 0.25f,
                            0.85f, 0.95f };

    vector<Real> std_ovlp{  0.01f, 0.01f,
                            0.33f, 0.05f,
                            0.05f, 0.02f };

    std::default_random_engine generator;
    for(auto i = 0u; i < mean_ovlp.size(); i++) {
        auto dist = std::normal_distribution<float>(mean_ovlp[i], std_ovlp[i]);

        for(auto z = 0u; z < runtime; z++) {
            Real ovlp;
            do {
                ovlp = dist( generator );
            } while( ovlp < 0.0f || ovlp > 1.0f );
            A.addNoise( 1.0f - ovlp );
            EXPECT_NEAR( V.overlap, ovlp, 0.501f / A.getSum() );
        }
        EXPECT_NEAR( V.mean(), mean_ovlp[i], std_ovlp[i] );
        EXPECT_NEAR( V.std(),  std_ovlp[i],  std_ovlp[i] / 2.0f );
    }
}

TEST(SdrMetricsTest, TestOverlap_Print) {
    // Test passes if it does not crash.  The exact strings are checked by
    // python unit tests.
    stringstream ss;
    SDR A({ 2000u });
    Overlap V( A, 100u );
    A.randomize( 0.02f );

    vector<Real> overlaps{ 0.02f, 0.05f, 0.0f, 0.50f, 0.0f };
    for(const auto ovlp : overlaps) {
        for(auto i = 0u; i < 1000u; i++)
            A.addNoise( 1.0f - ovlp );
        ss << V;
    }
    for(auto i = 0u; i < 1000u; i++)
        A.randomize( 0.02f );
    ss << V;
    ss << endl;
    VERBOSE << ss.str() << std::endl;
}

/**
 * Metrics
 *
 */
TEST(SdrMetricsTest, TestAllMetrics_Construct) {
    // Test that it constructs.
    SDR *A = new SDR({ 100u });
    Metrics M( *A, 10u );

    A->randomize( 0.05f );
    A->randomize( 0.05f );
    A->randomize( 0.05f );

    // Test use after freeing data source.
    delete A;
    stringstream devnull;
    devnull << M;

    // Test delete Metric and keep using SDR.
    A = new SDR({ 100u });
    Metrics *B = new Metrics( *A, 99u );
    Metrics *C = new Metrics( *A, 98u );
    A->randomize( 0.20f );
    A->randomize( 0.20f );
    devnull << *B;
    delete B;                   // First deletion
    A->randomize( 0.20f );
    A->addNoise( 0.20f );
    B = new Metrics( *A, 99u );    // Remove & Recreate
    A->randomize( 0.20f );
    A->randomize( 0.20f );
    devnull << *A;
    delete A;
    devnull << *C;
    delete C;
    devnull << *B;
    delete B;
}

/**
 * Metrics prints OK.
 */
TEST(SdrMetricsTest, TestAllMetrics_Print) {
    // Test passes if it does not crash.  The exact strings are checked by
    // python unit tests.
  stringstream ss;
    SDR A({ 4097u });
    Metrics M( A, 100u );

    vector<Real> sparsity{ 0.02f, 0.15f, 0.06f, 0.50f, 0.0f };
    vector<Real> overlaps{ 0.02f, 0.05f, 0.10f, 0.50f, 0.0f };
    for(auto test = 0u; test < sparsity.size(); test++) {
        A.randomize( sparsity[test] );
        for(auto i = 0u; i < 1000u; i++)
            A.addNoise( 1.0f - overlaps[test] );
        ss << M;
        ss << endl;
    }
    VERBOSE << ss.str() << std::endl;
}

/**
 * Test constructor with dimensions instead of SDR,
 * Test addData() methods.
 */
TEST(SdrMetricsTest, TestAddData) {
    Metrics M( {10u, 5u, 2u}, 100u );

    // Check error checking
    SDR badDims({2u, 5u, 10u});
    ASSERT_ANY_THROW( M.addData(badDims) );
    // Check that addData() only works when using the correct initializer.
    SDR A({100u});
    Metrics otherInit(A, 100u);
    ASSERT_ANY_THROW( otherInit.addData(A) );
    SDR B(A);
    ASSERT_ANY_THROW( otherInit.addData(B) );

    // Sanity check that data is used, not discarded.
    SDR C( M.dimensions );
    C.randomize( 0.20f );
    for(auto i = 0u; i < 10u; i++) {
        C.addNoise( 0.5f );
        M.addData( C );
    }
    ASSERT_NEAR( M.sparsity.mean(), 0.2f, 0.01f );
    ASSERT_NEAR( M.overlap.mean(),  0.5f, 0.01f );
    ASSERT_NEAR( M.activationFrequency.mean(), 0.2f, 0.01f );
}
}
