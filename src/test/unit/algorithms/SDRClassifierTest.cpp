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

#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::cla_classifier;
using namespace nupic::algorithms::sdr_classifier;

namespace
{

  TEST(SDRClassifierTest, Basic)
  {
    vector<UInt> steps;
    steps.push_back(1);
    SDRClassifier c = SDRClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    vector<UInt> bucketIdxList1;
    bucketIdxList1.push_back(4);
    vector<Real64> actValueList1;
    actValueList1.push_back(34.7);
    ClassifierResult result1;
    c.compute(0, input1, bucketIdxList1, actValueList1, false, true, true, &result1);

    // Create a vector of input bit indices
    vector<UInt> input2;
    input2.push_back(1);
    input2.push_back(5);
    input2.push_back(9);
    vector<UInt> bucketIdxList2;
    bucketIdxList2.push_back(4);
    vector<Real64> actValueList2;
    actValueList2.push_back(34.7);
    ClassifierResult result2;
    c.compute(1, input2, bucketIdxList2, actValueList2, false, true, true, &result2);

    {
      bool foundMinus1 = false;
      bool found1 = false;
      for (auto it = result2.begin();
           it != result2.end(); ++it)
      {
        if (it->first == -1)
        {
          // The -1 key is used for the actual values
          ASSERT_EQ(false, foundMinus1)
            << "Already found key -1 in classifier result";
          foundMinus1 = true;
          ASSERT_EQ(5, it->second->size())
            << "Expected five buckets since it has only seen bucket 4 (so it "
            << "Has buckets 0-4).";
          ASSERT_TRUE(fabs(it->second->at(4) - 34.7) < 0.000001)
            << "Incorrect actual value for bucket 4";
        } else if (it->first == 1) {
          // Check the one-step prediction
          ASSERT_EQ(false, found1)
            << "Already found key 1 in classifier result";
          found1 = true;
          ASSERT_EQ(5, it->second->size())
            << "Expected five bucket predictions";
          ASSERT_LT(fabs(it->second->at(0) - 0.2), 0.000001)
            << "Incorrect prediction for bucket 0";
          ASSERT_LT(fabs(it->second->at(1) - 0.2), 0.000001)
            << "Incorrect prediction for bucket 1";
          ASSERT_LT(fabs(it->second->at(2) - 0.2), 0.000001)
            << "Incorrect prediction for bucket 2";
          ASSERT_LT(fabs(it->second->at(3) - 0.2), 0.000001)
            << "Incorrect prediction for bucket 3";
          ASSERT_LT(fabs(it->second->at(4) - 0.2), 0.000001)
            << "Incorrect prediction for bucket 4";
        }
      }
      ASSERT_TRUE(foundMinus1) << "Key -1 not found in classifier result";
      ASSERT_TRUE(found1) << "key 1 not found in classifier result";
    }
  }

  TEST(SDRClassifierTest, SingleValue)
  {
    // Feed the same input 10 times, the corresponding probability should be
    // very high
    vector<UInt> steps;
    steps.push_back(1);
    SDRClassifier c = SDRClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    vector<UInt> bucketIdxList;
    bucketIdxList.push_back(4);
    vector<Real64> actValueList;
    actValueList.push_back(34.7);
    ClassifierResult result1;
    for (UInt i = 0; i < 10; ++i)
    {
      ClassifierResult result1;
      c.compute(i, input1, bucketIdxList, actValueList, false, true, true, &result1);
    }

    {
      for (auto it = result1.begin();
           it != result1.end(); ++it)
      {
        if (it->first == -1)
        {
          ASSERT_TRUE(fabs(it->second->at(4) - 10.0) < 0.000001)
            << "Incorrect actual value for bucket 4";
        } else if (it->first == 1) {
          ASSERT_GT(it->second->at(4), 0.9)
            << "Incorrect prediction for bucket 4";
        }
      }
    }

  }

  TEST(SDRClassifierTest, ComputeComplex)
  {
    // More complex classification
    // This test is ported from the Python unit test
    vector<UInt> steps;
    steps.push_back(1);
    SDRClassifier c = SDRClassifier(steps, 1.0, 0.1, 0);

    // Create a input vector
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    vector<UInt> bucketIdxList1;
    bucketIdxList1.push_back(4);
    vector<Real64> actValueList1;
    actValueList1.push_back(34.7);

    // Create a input vector
    vector<UInt> input2;
    input2.push_back(0);
    input2.push_back(6);
    input2.push_back(9);
    input2.push_back(11);
    vector<UInt> bucketIdxList2;
    bucketIdxList2.push_back(5);
    vector<Real64> actValueList2;
    actValueList2.push_back(41.7);

    // Create input vectors
    vector<UInt> input3;
    input3.push_back(6);
    input3.push_back(9);
    vector<UInt> bucketIdxList3;
    bucketIdxList3.push_back(5);
    vector<Real64> actValueList3;
    actValueList3.push_back(44.9);

    vector<UInt> bucketIdxList4;
    bucketIdxList4.push_back(4);
    vector<Real64> actValueList4;
    actValueList4.push_back(42.9);

    vector<UInt> bucketIdxList5;
    bucketIdxList5.push_back(4);
    vector<Real64> actValueList5;
    actValueList5.push_back(34.7);

    ClassifierResult result1;
    c.compute(0, input1, bucketIdxList1, actValueList1, false, true, true, &result1);

    ClassifierResult result2;
    c.compute(1, input2, bucketIdxList2, actValueList2, false, true, true, &result2);

    ClassifierResult result3;
    c.compute(2, input3, bucketIdxList3, actValueList3, false, true, true, &result3);

    ClassifierResult result4;
    c.compute(3, input1, bucketIdxList4, actValueList4, false, true, true, &result4);

    ClassifierResult result5;
    c.compute(4, input1, bucketIdxList5, actValueList5, false, true, true, &result5);

    {
      bool foundMinus1 = false;
      bool found1 = false;
      for (auto it = result5.begin(); it != result5.end(); ++it)
      {
        ASSERT_TRUE(it->first == -1 || it->first == 1)
          << "Result vector should only have -1 or 1 as key";
        if (it->first == -1)
        {
          // The -1 key is used for the actual values
          ASSERT_EQ(false, foundMinus1)
            << "Already found key -1 in classifier result";
          foundMinus1 = true;
          ASSERT_EQ(6, it->second->size())
            << "Expected six buckets since it has only seen bucket 4-5 (so it "
            << "has buckets 0-5).";
          ASSERT_TRUE(fabs(it->second->at(4) - 35.520000457763672) < 0.000001)
            << "Incorrect actual value for bucket 4";
          ASSERT_TRUE(fabs(it->second->at(5) - 42.020000457763672) < 0.000001)
            << "Incorrect actual value for bucket 5";
        } else if (it->first == 1) {
          // Check the one-step prediction
          ASSERT_EQ(false, found1)
            << "Already found key 1 in classifier result";
          found1 = true;

          ASSERT_EQ(6, it->second->size())
            << "Expected six bucket predictions";
          ASSERT_LT(fabs(it->second->at(0) - 0.034234), 0.000001)
            << "Incorrect prediction for bucket 0";
          ASSERT_LT(fabs(it->second->at(1) - 0.034234), 0.000001)
            << "Incorrect prediction for bucket 1";
          ASSERT_LT(fabs(it->second->at(2) - 0.034234), 0.000001)
            << "Incorrect prediction for bucket 2";
          ASSERT_LT(fabs(it->second->at(3) - 0.034234), 0.000001)
            << "Incorrect prediction for bucket 3";
          ASSERT_LT(fabs(it->second->at(4) - 0.093058), 0.000001)
            << "Incorrect prediction for bucket 4";
          ASSERT_LT(fabs(it->second->at(5) - 0.770004), 0.000001)
            << "Incorrect prediction for bucket 5";
        }
      }
      ASSERT_TRUE(foundMinus1) << "Key -1 not found in classifier result";
      ASSERT_TRUE(found1) << "Key 1 not found in classifier result";
    }

  }

  TEST(SDRClassifierTest, MultipleCategory)
  {
    // Test multiple category classification with single compute calls
    // This test is ported from the Python unit test
    vector<UInt> steps;
    steps.push_back(0);
    SDRClassifier c = SDRClassifier(steps, 1.0, 0.1, 0);

    // Create a input vectors
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(3);
    input1.push_back(5);
    vector<UInt> bucketIdxList1;
    bucketIdxList1.push_back(0);
    bucketIdxList1.push_back(1);
    vector<Real64> actValueList1;
    actValueList1.push_back(0);
    actValueList1.push_back(1);

    // Create a input vectors
    vector<UInt> input2;
    input2.push_back(2);
    input2.push_back(4);
    input2.push_back(6);
    vector<UInt> bucketIdxList2;
    bucketIdxList2.push_back(2);
    bucketIdxList2.push_back(3);
    vector<Real64> actValueList2;
    actValueList2.push_back(2);
    actValueList2.push_back(3);

    int recordNum=0;
    for(int i=0; i<1000; i++)
    {
      ClassifierResult result1;
      ClassifierResult result2;
      c.compute(recordNum, input1, bucketIdxList1, actValueList1, false, true, true, &result1);
      recordNum += 1;
      c.compute(recordNum, input2, bucketIdxList2, actValueList2, false, true, true, &result2);
      recordNum += 1;
    }

    ClassifierResult result1;
    ClassifierResult result2;
    c.compute(recordNum, input1, bucketIdxList1, actValueList1, false, true, true, &result1);
    recordNum += 1;
    c.compute(recordNum, input2, bucketIdxList2, actValueList2, false, true, true, &result2);
    recordNum += 1;

    for (auto it = result1.begin(); it != result1.end(); ++it)
    {
      if (it->first == 0) {
        ASSERT_LT(fabs(it->second->at(0) - 0.5), 0.1)
        << "Incorrect prediction for bucket 0 (expected=0.5)";
        ASSERT_LT(fabs(it->second->at(1) - 0.5), 0.1)
        << "Incorrect prediction for bucket 1 (expected=0.5)";
      }
    }

    for (auto it = result2.begin(); it != result2.end(); ++it)
    {
      if (it->first == 0) {
        ASSERT_LT(fabs(it->second->at(2) - 0.5), 0.1)
        << "Incorrect prediction for bucket 2 (expected=0.5)";
        ASSERT_LT(fabs(it->second->at(3) - 0.5), 0.1)
        << "Incorrect prediction for bucket 3 (expected=0.5)";
      }
    }

  }

  TEST(SDRClassifierTest, SaveLoad)
  {
    vector<UInt> steps;
    steps.push_back(1);
    SDRClassifier c1 = SDRClassifier(steps, 0.1, 0.1, 0);
    SDRClassifier c2 = SDRClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    vector<UInt> bucketIdxList1;
    bucketIdxList1.push_back(4);
    vector<Real64> actValueList1;
    actValueList1.push_back(34.7);
    ClassifierResult result;
    c1.compute(0, input1, bucketIdxList1, actValueList1, false, true, true, &result);

    {
      stringstream ss;
      c1.save(ss);
      c2.load(ss);
    }

    ASSERT_TRUE(c1 == c2);

    ClassifierResult result1, result2;
    c1.compute(1, input1, bucketIdxList1, actValueList1, false, true, true, &result1);
    c2.compute(1, input1, bucketIdxList1, actValueList1, false, true, true, &result2);

    ASSERT_TRUE(result1 == result2);
  }

  TEST(SDRClassifierTest, WriteRead)
  {
    vector<UInt> steps;
    steps.push_back(1);
    steps.push_back(2);
    SDRClassifier c1 = SDRClassifier(steps, 0.1, 0.1, 0);
    SDRClassifier c2 = SDRClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    vector<UInt> bucketIdxList1;
    bucketIdxList1.push_back(4);
    vector<Real64> actValueList1;
    actValueList1.push_back(34.7);
    ClassifierResult trainResult1;
    c1.compute(0, input1, bucketIdxList1, actValueList1, false, true, true, &trainResult1);

        // Create a vector of input bit indices
    vector<UInt> input2;
    input2.push_back(0);
    input2.push_back(8);
    input2.push_back(9);
    vector<UInt> bucketIdxList2;
    bucketIdxList2.push_back(2);
    vector<Real64> actValueList2;
    actValueList2.push_back(24.7);
    ClassifierResult trainResult2;
    c1.compute(1, input2, bucketIdxList2, actValueList2, false, true, true, &trainResult2);

    {
      stringstream ss;
      c1.write(ss);
      c2.read(ss);
    }

    ASSERT_TRUE(c1 == c2);

    ClassifierResult result1, result2;
    c1.compute(2, input1, bucketIdxList1, actValueList1, false, true, true, &result1);
    c2.compute(2, input1, bucketIdxList1, actValueList1, false, true, true, &result2);

    ASSERT_TRUE(result1 == result2);
  }

} // end namespace
