/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
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
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for SDRClassifier
 */

#include <cmath> // isnan
#include <iostream>
#include <limits> // numeric_limits
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
namespace algorithms {
namespace sdr_classifier {

// SDRClassifier friend class used to access private members
class SDRClassifierTest : public ::testing::Test {
protected:
  typedef std::vector<double>::iterator Iterator;
  void softmax_(SDRClassifier *self, Iterator begin, Iterator end) {
    self->softmax_(begin, end);
  };
};
} // namespace sdr_classifier
} // namespace algorithms
} // namespace nupic

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::cla_classifier;
using namespace nupic::algorithms::sdr_classifier;
namespace {
TEST_F(SDRClassifierTest, Basic) {
  vector<UInt> steps{1u};
  SDRClassifier c = SDRClassifier(steps, 0.1f, 0.1f, 0u);

  // Create a vector of input bit indices
  vector<UInt> input1{1u, 5u, 9u};
  vector<UInt> bucketIdxList1{4u};
  vector<Real64> actValueList1{34.7f};
  ClassifierResult result1;
  c.compute(0u, input1, bucketIdxList1, actValueList1, false, true, true, &result1);

  // Create a vector of input bit indices
  vector<UInt> input2{1u, 5u, 9u};
  vector<UInt> bucketIdxList2{4u};
  vector<Real64> actValueList2{ 34.7f };
  ClassifierResult result2;
  c.compute(1u, input2, bucketIdxList2, actValueList2, false, true, true, &result2);

  {
    bool foundMinus1 = false;
    bool found1 = false;
    for (auto it = result2.begin(); it != result2.end(); ++it) {
      if (it->first == -1) {
        // The -1 key is used for the actual values
        ASSERT_FALSE(foundMinus1) << "Already found key -1 in classifier result";
        foundMinus1 = true;
        ASSERT_EQ(5ul, it->second->size())
            << "Expected five buckets since it has only seen bucket 4 (so it "
            << "Has buckets 0-4).";
        ASSERT_LT(fabs(it->second->at(4) - 34.7f), 0.000001f)
                                      << "Incorrect actual value for bucket 4";
      } else if (it->first == 1) {
        // Check the one-step prediction
        ASSERT_FALSE(found1) << "Already found key 1 in classifier result";
        found1 = true;
        ASSERT_EQ(5ul, it->second->size()) << "Expected five bucket predictions";
        ASSERT_NEAR(it->second->at(0u), 0.2f, 0.000001f) << "Incorrect prediction for bucket 0";
        ASSERT_NEAR(it->second->at(1u), 0.2f, 0.000001f) << "Incorrect prediction for bucket 1";
        ASSERT_NEAR(it->second->at(2u), 0.2f, 0.000001f) << "Incorrect prediction for bucket 2";
        ASSERT_NEAR(it->second->at(3u), 0.2f, 0.000001f) << "Incorrect prediction for bucket 3";
        ASSERT_NEAR(it->second->at(4u), 0.2f, 0.000001f) << "Incorrect prediction for bucket 4";
      }
    }
    ASSERT_TRUE(foundMinus1) << "Key -1 not found in classifier result";
    ASSERT_TRUE(found1) << "key 1 not found in classifier result";
  }
}

TEST_F(SDRClassifierTest, SingleValue) {
  // Feed the same input 10 times, the corresponding probability should be
  // very high
  vector<UInt> steps{1u};
  SDRClassifier c = SDRClassifier(steps, 0.1f, 0.1f, 0u);

  // Create a vector of input bit indices
  vector<UInt> input1{1u, 5u, 9u};
  vector<UInt> bucketIdxList{4u};
  vector<Real64> actValueList{34.7f};
  ClassifierResult result1;
  for (UInt i = 0u; i < 10u; ++i) {
    ClassifierResult result1;
    c.compute(i, input1, bucketIdxList, actValueList, false, true, true, &result1);
  }

  {
    for (auto it = result1.begin(); it != result1.end(); ++it) {
      if (it->first == -1) {
        ASSERT_LT(fabs(it->second->at(4u) - 10.0f), 0.000001f)
            << "Incorrect actual value for bucket 4";
      } else if (it->first == 1) {
        ASSERT_GT(it->second->at(4u), 0.9f)
            << "Incorrect prediction for bucket 4";
      }
    }
  }
}


TEST_F(SDRClassifierTest, ComputeComplex) {
  // More complex classification
  // This test is ported from the Python unit test
  vector<UInt> steps{1u};
  SDRClassifier c = SDRClassifier(steps, 1.0f, 0.1f, 0u);

  // Create a input vector
  vector<UInt> input1{ 1u, 5u, 9u };
  vector<UInt> bucketIdxList1{ 4u };
  vector<Real64> actValueList1{ 34.7f };

  // Create a input vector
  vector<UInt> input2{ 0u, 6u, 9u, 11u };
  vector<UInt> bucketIdxList2{ 5u };
  vector<Real64> actValueList2{ 41.7f };

  // Create input vectors
  vector<UInt> input3{ 6u, 9u };
  vector<UInt> bucketIdxList3{ 5u };
  vector<Real64> actValueList3{ 44.9f };

  vector<UInt> bucketIdxList4{ 4u };
  vector<Real64> actValueList4{ 42.9f };

  vector<UInt> bucketIdxList5{ 4u };
  vector<Real64> actValueList5{ 34.7f };

  ClassifierResult result1;
  c.compute(0, input1, bucketIdxList1, actValueList1, false, true, true,
            &result1);

  ClassifierResult result2;
  c.compute(1, input2, bucketIdxList2, actValueList2, false, true, true,
            &result2);

  ClassifierResult result3;
  c.compute(2, input3, bucketIdxList3, actValueList3, false, true, true,
            &result3);

  ClassifierResult result4;
  c.compute(3, input1, bucketIdxList4, actValueList4, false, true, true,
            &result4);

  ClassifierResult result5;
  c.compute(4, input1, bucketIdxList5, actValueList5, false, true, true,
            &result5);

  {
    bool foundMinus1 = false;
    bool found1 = false;
    for (auto it = result5.begin(); it != result5.end(); ++it) {
      ASSERT_TRUE(it->first == -1 || it->first == 1)
          << "Result vector should only have -1 or 1 as key";
      if (it->first == -1) {
        // The -1 key is used for the actual values
        ASSERT_FALSE(foundMinus1)
            << "Already found key -1 in classifier result";
        foundMinus1 = true;
        ASSERT_EQ(6ul, it->second->size())
            << "Expected six buckets since it has only seen bucket 4-5 (so it "
            << "has buckets 0-5).";
        ASSERT_LT(fabs(it->second->at(4u) - 35.520000457763672f), 0.000001f)
            << "Incorrect actual value for bucket 4";
        ASSERT_LT(fabs(it->second->at(5u) - 42.020000457763672f), 0.000001f)
            << "Incorrect actual value for bucket 5";
      } else if (it->first == 1) {
        // Check the one-step prediction
        ASSERT_FALSE(found1) << "Already found key 1 in classifier result";
        found1 = true;

        ASSERT_EQ(6ul, it->second->size()) << "Expected six bucket predictions";
        ASSERT_LT(fabs(it->second->at(0u) - 0.034234f), 0.000001f)
            << "Incorrect prediction for bucket 0";
        ASSERT_LT(fabs(it->second->at(1u) - 0.034234f), 0.000001f)
            << "Incorrect prediction for bucket 1";
        ASSERT_LT(fabs(it->second->at(2u) - 0.034234f), 0.000001f)
            << "Incorrect prediction for bucket 2";
        ASSERT_LT(fabs(it->second->at(3u) - 0.034234f), 0.000001f)
            << "Incorrect prediction for bucket 3";
        ASSERT_LT(fabs(it->second->at(4u) - 0.093058f), 0.000001f)
            << "Incorrect prediction for bucket 4";
        ASSERT_LT(fabs(it->second->at(5u) - 0.770004f), 0.000001f)
            << "Incorrect prediction for bucket 5";
      }
    }
    ASSERT_TRUE(foundMinus1) << "Key -1 not found in classifier result";
    ASSERT_TRUE(found1) << "Key 1 not found in classifier result";
  }
}

TEST_F(SDRClassifierTest, MultipleCategory) {
  // Test multiple category classification with single compute calls
  // This test is ported from the Python unit test
  vector<UInt> steps{ 0u };
  SDRClassifier c = SDRClassifier(steps, 1.0f, 0.1f, 0u);

  // Create a input vectors
  vector<UInt> input1{ 1u, 3u, 5u };
  vector<UInt> bucketIdxList1{ 0u, 1u };
  vector<Real64> actValueList1{ 0u, 1u };

  // Create a input vectors
  vector<UInt> input2{ 2u, 4u, 6u };
  vector<UInt> bucketIdxList2{ 2u, 3u };
  vector<Real64> actValueList2{ 2.0f, 3.0f };

  auto recordNum = 0u;
  for (auto i = 0u; i < 1000u; i++) {
    ClassifierResult result1;
    ClassifierResult result2;
    c.compute(recordNum, input1, bucketIdxList1, actValueList1, false, true,
              true, &result1);
    recordNum += 1u;
    c.compute(recordNum, input2, bucketIdxList2, actValueList2, false, true,
              true, &result2);
    recordNum += 1u;
  }

  ClassifierResult result1;
  ClassifierResult result2;
  c.compute(recordNum, input1, bucketIdxList1, actValueList1, false, true, true,
            &result1);
  recordNum += 1u;
  c.compute(recordNum, input2, bucketIdxList2, actValueList2, false, true, true,
            &result2);
  recordNum += 1u;

  for (auto it = result1.begin(); it != result1.end(); ++it) {
    if (it->first == 0) {
      ASSERT_LT(fabs(it->second->at(0u) - 0.5f), 0.1f)
          << "Incorrect prediction for bucket 0 (expected=0.5)";
      ASSERT_LT(fabs(it->second->at(1u) - 0.5f), 0.1f)
          << "Incorrect prediction for bucket 1 (expected=0.5)";
    }
  }

  for (auto it = result2.begin(); it != result2.end(); ++it) {
    if (it->first == 0) {
      ASSERT_LT(fabs(it->second->at(2u) - 0.5f), 0.1f)
          << "Incorrect prediction for bucket 2 (expected=0.5)";
      ASSERT_LT(fabs(it->second->at(3u) - 0.5f), 0.1f)
          << "Incorrect prediction for bucket 3 (expected=0.5)";
    }
  }
}

TEST_F(SDRClassifierTest, SaveLoad) {
  vector<UInt> steps{ 1u };
  SDRClassifier c1 = SDRClassifier(steps, 0.1f, 0.1f, 0u);
  SDRClassifier c2 = SDRClassifier(steps, 0.1f, 0.1f, 0u);

  // Create a vector of input bit indices
  vector<UInt> input1{ 1u, 5u, 9u };
  vector<UInt> bucketIdxList1{4u};
  vector<Real64> actValueList1{34.7f};
  ClassifierResult result;
  c1.compute(0u, input1, bucketIdxList1, actValueList1, false, true, true, &result);

  {
    stringstream ss;
    EXPECT_NO_THROW(c1.save(ss));
    EXPECT_NO_THROW(c2.load(ss));
  }
  ASSERT_EQ(c1, c2);

  ClassifierResult result1, result2;
  c1.compute(1u, input1, bucketIdxList1, actValueList1, false, true, true, &result1);
  c2.compute(1u, input1, bucketIdxList1, actValueList1, false, true, true, &result2);

  ASSERT_EQ(result1, result2);
}


TEST_F(SDRClassifierTest, testSoftmaxOverflow) {
  SDRClassifier c = SDRClassifier({1u}, 0.5f, 0.5f, 0u);
  std::vector<Real64> values = {numeric_limits<Real64>::max()};
  softmax_(&c, values.begin(), values.end());
  Real64 result = values[0u];
  ASSERT_FALSE(std::isnan(result));
}


TEST_F(SDRClassifierTest, testSoftmax) {
  SDRClassifier c = SDRClassifier({1u}, 0.1f, 0.3f, 0u);
  std::vector<Real64> values {0.0f, 1.0f, 1.337f, 2.018f, 1.1f, 0.5f, 0.9f};
  const std::vector<Real64> exp {
	  0.045123016137150938f,
	  0.12265707481088166f,
	  0.17181055613150184f,
	  0.3394723335640627f,
	  0.13555703197721547f,
	  0.074395276503465876f,
	  0.11098471087572169f};

  softmax_(&c, values.begin(), values.end());

  for(auto i = 0u; i < exp.size(); i++) {
    EXPECT_NEAR(values[i], exp[i], 0.000001f) << "softmax ["<< i <<"]";
  }
}

} // end namespace
