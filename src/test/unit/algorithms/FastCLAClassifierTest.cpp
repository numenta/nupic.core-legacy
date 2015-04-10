/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Implementation of unit tests for NearestNeighbor
 */

#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/FastClaClassifier.hpp>
#include <nupic/math/StlIo.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::cla_classifier;

namespace
{

  TEST(FastCLAClassifierTest, Basic)
  {
    vector<UInt> steps;
    steps.push_back(1);
    FastCLAClassifier c = FastCLAClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    ClassifierResult result1;
    c.fastCompute(0, input1, 4, 34.7, false, true, true, &result1);

    // Create a vector of input bit indices
    vector<UInt> input2;
    input2.push_back(1);
    input2.push_back(5);
    input2.push_back(9);
    ClassifierResult result2;
    c.fastCompute(1, input2, 4, 34.7, false, true, true, &result2);

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
            << "already found key -1 in classifier result";
          foundMinus1 = true;
          ASSERT_EQ((long unsigned int)5, it->second->size())
            << "Expected five buckets since it has only seen bucket 4 (so it "
            << "has buckets 0-4).";
          ASSERT_TRUE(fabs(it->second->at(4) - 34.7) < 0.000001)
            << "Incorrect actual value for bucket 4";
        } else if (it->first == 1) {
          // Check the one-step prediction
          ASSERT_EQ(false, found1)
            << "already found key 1 in classifier result";
          found1 = true;
          ASSERT_EQ((long unsigned int)5, it->second->size())
            << "expected five bucket predictions";
          ASSERT_LT(fabs(it->second->at(0) - 0.2), 0.000001)
            << "incorrect prediction for bucket 0";
          ASSERT_LT(fabs(it->second->at(1) - 0.2), 0.000001)
            << "incorrect prediction for bucket 1";
          ASSERT_LT(fabs(it->second->at(2) - 0.2), 0.000001)
            << "incorrect prediction for bucket 2";
          ASSERT_LT(fabs(it->second->at(3) - 0.2), 0.000001)
            << "incorrect prediction for bucket 3";
          ASSERT_LT(fabs(it->second->at(4) - 0.2), 0.000001)
            << "incorrect prediction for bucket 4";
        }
      }
      ASSERT_TRUE(foundMinus1) << "key -1 not found in classifier result";
      ASSERT_TRUE(found1) << "key 1 not found in classifier result";
    }
  }

  TEST(FastCLAClassifierTest, SaveLoad)
  {
    vector<UInt> steps;
    steps.push_back(1);
    FastCLAClassifier c1 = FastCLAClassifier(steps, 0.1, 0.1, 0);
    FastCLAClassifier c2 = FastCLAClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    ClassifierResult result;
    c1.fastCompute(0, input1, 4, 34.7, false, true, true, &result);

    {
      stringstream ss;
      c1.save(ss);
      c2.load(ss);
    }

    ASSERT_TRUE(c1 == c2);

    ClassifierResult result1, result2;
    c1.fastCompute(1, input1, 4, 35.7, false, true, true, &result1);
    c2.fastCompute(1, input1, 4, 35.7, false, true, true, &result2);

    ASSERT_TRUE(result1 == result2);
  }

  TEST(FastCLAClassifierTest, WriteRead)
  {
    vector<UInt> steps;
    steps.push_back(1);
    FastCLAClassifier c1 = FastCLAClassifier(steps, 0.1, 0.1, 0);
    FastCLAClassifier c2 = FastCLAClassifier(steps, 0.1, 0.1, 0);

    // Create a vector of input bit indices
    vector<UInt> input1;
    input1.push_back(1);
    input1.push_back(5);
    input1.push_back(9);
    ClassifierResult result;
    c1.fastCompute(0, input1, 4, 34.7, false, true, true, &result);

    {
      stringstream ss;
      c1.write(ss);
      c2.read(ss);
    }

    ASSERT_TRUE(c1 == c2);

    ClassifierResult result1, result2;
    c1.fastCompute(1, input1, 4, 35.7, false, true, true, &result1);
    c2.fastCompute(1, input1, 4, 35.7, false, true, true, &result2);

    ASSERT_TRUE(result1 == result2);
  }

} // end namespace
