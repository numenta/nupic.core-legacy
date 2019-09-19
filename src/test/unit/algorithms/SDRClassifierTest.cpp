/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Implementation of unit tests for Classifier & Predictor
 */

#include <cmath> // isnan
#include <iostream>
#include <limits> // numeric_limits
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <htm/algorithms/SDRClassifier.hpp>
#include <htm/utils/Log.hpp>

using namespace std;
using namespace htm;

namespace testing {


TEST(SDRClassifierTest, ExampleUsageClassifier)
{
  // Make a random SDR and associate it with the category B.
  SDR inputData({ 1000u });
      inputData.randomize( 0.02f );
  enum Category { A, B, C, D };
  Classifier clsr;
  clsr.learn( inputData, { Category::B } );
  ASSERT_EQ( argmax( clsr.infer( inputData ) ),  Category::B );

  // Estimate a scalar value.  The Classifier only accepts categories, so
  // put real valued inputs into bins (AKA buckets) by subtracting the
  // minimum value and dividing by a resolution.
  double scalar = 567.8f;
  double minimum = 500.0f;
  double resolution = 10.0f;
  clsr.learn( inputData, { (UInt)((scalar - minimum) / resolution) } );
  ASSERT_EQ( argmax( clsr.infer( inputData ) ) * resolution + minimum,  560.0f );
}


TEST(SDRClassifierTest, ExampleUsagePredictor)
{
  // Predict 1 and 2 time steps into the future.

  // Make a sequence of 4 random SDRs. Each SDR has 1000 bits and 2% sparsity.
  vector<SDR> sequence( 4u, vector<UInt>{ 1000u } );
  for( SDR & inputData : sequence ) {
      inputData.randomize( 0.02f );
  }
  // Make category labels for the sequence.
  vector<UInt> labels = { 4, 5, 6, 7 };

  // Make a Predictor and train it.
  Predictor pred( vector<UInt>{ 1, 2 } );
  pred.learn( 0, sequence[0], { labels[0] } );
  pred.learn( 1, sequence[1], { labels[1] } );
  pred.learn( 2, sequence[2], { labels[2] } );
  pred.learn( 3, sequence[3], { labels[3] } );

  // Give the predictor partial information, and make predictions
  // about the future.
  pred.reset();
  Predictions A = pred.infer( sequence[0] );
  ASSERT_EQ( argmax( A[1] ),  labels[1] );
  ASSERT_EQ( argmax( A[2] ),  labels[2] );

  Predictions B = pred.infer( sequence[1] );
  ASSERT_EQ( argmax( B[1] ),  labels[2] );
  ASSERT_EQ( argmax( B[2] ),  labels[3] );
}


TEST(SDRClassifierTest, SingleValue) {
  // Feed the same input 10 times, the corresponding probability should be
  // very high
  vector<UInt> steps{1u};
  Predictor c(steps, 0.1f);

  // Create a vector of input bit indices
  SDR input1({10u}); input1.setSparse(SDR_sparse_t({ 1u, 5u, 9u }));
  vector<UInt> bucketIdxList{4u};
  for (UInt i = 0u; i < 10u; ++i) {
    c.learn( i, input1, bucketIdxList );
  }
  Predictions result1 = c.infer( input1 );

  ASSERT_EQ( argmax( result1[1u] ), 4u )
      << "Incorrect prediction for bucket 4";

  ASSERT_EQ( result1.size(), 1u );
}


TEST(SDRClassifierTest, ComputeComplex) {
  // More complex classification
  // This test is ported from the Python unit test
  Predictor c({1u}, 1.0f);

  // Create a input vector
  SDR input1({ 20u });
  input1.setSparse(SDR_sparse_t({ 1u, 5u, 9u }));
  vector<UInt> bucketIdxList1{ 4u };

  // Create a input vector
  SDR input2({ 20u });
  input2.setSparse(SDR_sparse_t({ 0u, 6u, 9u, 11u }));
  vector<UInt> bucketIdxList2{ 5u };

  // Create input vectors
  SDR input3({ 20u });
  input3.setSparse(SDR_sparse_t({ 6u, 9u }));
  vector<UInt> bucketIdxList3{ 5u };
  vector<UInt> bucketIdxList4{ 4u };
  vector<UInt> bucketIdxList5{ 4u };

  c.learn(0, input1, bucketIdxList1);
  c.learn(1, input2, bucketIdxList2);
  c.learn(2, input3, bucketIdxList3);
  c.learn(3, input1, bucketIdxList4);
  auto result = c.infer(input1);

  // Check the one-step prediction
  ASSERT_EQ(result.size(), 1u)
    << "Result should only have 1 key.";
  ASSERT_EQ(6ul, result[1u].size()) << "Expected six bucket predictions";
  ASSERT_LT(fabs(result[1u].at(0u) - 0.034234f), 0.000001f)
      << "Incorrect prediction for bucket 0";
  ASSERT_LT(fabs(result[1u].at(1u) - 0.034234f), 0.000001f)
      << "Incorrect prediction for bucket 1";
  ASSERT_LT(fabs(result[1u].at(2u) - 0.034234f), 0.000001f)
      << "Incorrect prediction for bucket 2";
  ASSERT_LT(fabs(result[1u].at(3u) - 0.034234f), 0.000001f)
      << "Incorrect prediction for bucket 3";
  ASSERT_LT(fabs(result[1u].at(4u) - 0.093058f), 0.000001f)
      << "Incorrect prediction for bucket 4";
  ASSERT_LT(fabs(result[1u].at(5u) - 0.770004f), 0.000001f)
      << "Incorrect prediction for bucket 5";
}


TEST(SDRClassifierTest, MultipleCategories) {
  // Test multiple category classification with single compute calls
  // This test is ported from the Python unit test
  Classifier c(1.0f);

  // Create a input vectors
  SDR input1({ 10 });
  input1.setSparse(SDR_sparse_t({ 1u, 3u, 5u }));
  vector<UInt> bucketIdxList1{ 0u, 1u };

  // Create a input vectors
  SDR input2({ 10 });
  input2.setSparse(SDR_sparse_t({ 2u, 4u, 6u }));
  vector<UInt> bucketIdxList2{ 2u, 3u };

  // Train
  for (auto i = 0u; i < 1000u; i++) {
    c.learn( input1, bucketIdxList1 );
    c.learn( input2, bucketIdxList2 );
  }

  // Test
  PDF result1 = c.infer( input1 );
  PDF result2 = c.infer( input2 );

  ASSERT_LT(fabs(result1.at(0u) - 0.5f), 0.1f)
      << "Incorrect prediction for bucket 0 (expected=0.5)";
  ASSERT_LT(fabs(result1.at(1u) - 0.5f), 0.1f)
      << "Incorrect prediction for bucket 1 (expected=0.5)";

  ASSERT_LT(fabs(result2.at(2u) - 0.5f), 0.1f)
      << "Incorrect prediction for bucket 2 (expected=0.5)";
  ASSERT_LT(fabs(result2.at(3u) - 0.5f), 0.1f)
      << "Incorrect prediction for bucket 3 (expected=0.5)";
}


TEST(SDRClassifierTest, SaveLoad) {
  vector<UInt> steps{ 1u };
  Predictor c1(steps, 0.1f);

  // Train a Predictor with a few different things.
  SDR A({ 100u }); A.randomize( 0.10f );
  for(UInt i = 0; i < 10u; i++)
    { c1.learn(i, A, {4u}); }
  c1.reset();
  A.addNoise( 1.0f ); // Change every bit.
  for(UInt i = 0; i < 10u; i++)
    { c1.learn(i, A, {3u, 5u}); }
  // Measure and save some output.
  A.addNoise( 0.20f ); // Change two bits.
  c1.reset();
  const auto c1_out = c1.infer( A );

  // Save and load.
  stringstream ss;
  EXPECT_NO_THROW(c1.save(ss));
  Predictor c2;
  EXPECT_NO_THROW(c2.load(ss));

  // Expect identical results.
  const auto c2_out = c2.infer( A );
  ASSERT_EQ(c1_out, c2_out);
}


TEST(SDRClassifierTest, testSoftmaxOverflow) {
  PDF values({ numeric_limits<Real>::max() });
  softmax(values.begin(), values.end());
  auto result = values[0u];
  ASSERT_FALSE(std::isnan(result));
}


TEST(SDRClassifierTest, testSoftmax) {
  PDF values {0.0f, 1.0f, 1.337f, 2.018f, 1.1f, 0.5f, 0.9f};
  const PDF exp {
    0.045123016137150938f,
    0.12265707481088166f,
    0.17181055613150184f,
    0.3394723335640627f,
    0.13555703197721547f,
    0.074395276503465876f,
    0.11098471087572169f};

  softmax(values.begin(), values.end());

  for(auto i = 0u; i < exp.size(); i++) {
    EXPECT_NEAR(values[i], exp[i], 0.000001f) << "softmax ["<< i <<"]";
  }
}

} // end namespace
