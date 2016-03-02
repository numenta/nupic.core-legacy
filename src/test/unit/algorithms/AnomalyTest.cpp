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



#include <vector>
#include <algorithm>
#include "nupic/algorithms/Anomaly.hpp"
#include "gtest/gtest.h"

using namespace nupic::algorithms::anomaly;

TEST(ComputeRawAnomalyScore, NoActiveOrPredicted) 
{
  std::vector<int> active{};
  std::vector<int> predicted{};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0);
};


TEST(ComputeRawAnomalyScore, NoActive)
{
  std::vector<int> active {};
  std::vector<int> predicted {3, 5};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0);
};


TEST(ComputeRawAnomalyScore, PerfectMatch)
{
  std::vector<int> active {3, 5, 7};
  std::vector<int> predicted {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0);
};


TEST(ComputeRawAnomalyScore, NoMatch)
{
  std::vector<int> active {2, 4, 6};
  std::vector<int> predicted {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 1.0);
};


TEST(ComputeRawAnomalyScore, PartialMatch)
{
  std::vector<int> active {2, 3, 6};
  std::vector<int> predicted {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 2 / float(3));
};


TEST(Anomaly, ComputeScoreNoActiveOrPredicted)
{
  std::vector<int> active {};
  std::vector<int> predicted {};
  Anomaly a{};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 0.0);
}


TEST(Anomaly, ComputeScoreNoActive)
{
  std::vector<int> active {};
  std::vector<int> predicted {3, 5};
  Anomaly a{};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 0.0);
}


TEST(Anomaly, ComputeScorePerfectMatch)
{
  std::vector<int> active {3, 5, 7};
  std::vector<int> predicted {3, 5, 7};
  Anomaly a{};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 0.0);
}


TEST(Anomaly, ComputeScoreNoMatch)
{
  std::vector<int> active {2, 4, 6};
  std::vector<int> predicted {3, 5, 7};
  Anomaly a{};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 1.0);
}


TEST(Anomaly, ComputeScorePartialMatch)
{
  std::vector<int> active {2, 3, 6};
  std::vector<int> predicted {3, 5, 7};
  Anomaly a{};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 2.0 / 3.0);
}


TEST(Anomaly, Cumulative)
{
  const int TEST_COUNT = 9;
  Anomaly a{3};
  std::vector<std::vector<int>> preds(TEST_COUNT);
  std::generate(preds.begin(), preds.end(), []() { return std::vector<int> {1, 2, 6}; });

  std::vector<std::vector<int>> acts {
    std::vector<int> {1, 2, 6}, std::vector<int> {1, 2, 6}, std::vector<int> {1, 4, 6},
    std::vector<int> {10, 11, 6}, std::vector<int> {10, 11, 12}, std::vector<int> {10, 11, 12},
    std::vector<int> {10, 11, 12}, std::vector<int> {1, 2, 6}, std::vector<int> {1, 2, 6}
  };

  std::vector<float> expected {0.0, 0.0, 1.0/9.0, 3.0/9.0, 2.0/3.0, 8.0/9.0, 1.0, 2.0/3.0, 1.0/3.0};

  for (int index = 0; index < TEST_COUNT; index++) {
    ASSERT_FLOAT_EQ(a.compute(acts[index],  preds[index]), expected[index]);
  }
}


TEST(Anomaly, SelectModePure)
{
  Anomaly a{0, AnomalyMode::PURE, 0};
  std::vector<int> active {2, 3, 6};
  std::vector<int> predicted {3, 5, 7};
  ASSERT_FLOAT_EQ(a.compute(active, predicted), 2.0 / 3.0);
};

// Not implemented.
/*

  def testEquals(self):
    an = Anomaly()
    anP = Anomaly()
    self.assertEqual(an, anP, "default constructors equal")

    anN = Anomaly(mode=Anomaly.MODE_LIKELIHOOD)
    self.assertNotEqual(an, anN)
    an = Anomaly(mode=Anomaly.MODE_LIKELIHOOD)
    self.assertEqual(an, anN)

    an = Anomaly(slidingWindowSize=5, mode=Anomaly.MODE_WEIGHTED, binaryAnomalyThreshold=0.9)
    anP = Anomaly(slidingWindowSize=5, mode=Anomaly.MODE_WEIGHTED, binaryAnomalyThreshold=0.9)
    anN = Anomaly(slidingWindowSize=4, mode=Anomaly.MODE_WEIGHTED, binaryAnomalyThreshold=0.9)
    self.assertEqual(an, anP)
    self.assertNotEqual(an, anN)
    anN = Anomaly(slidingWindowSize=5, mode=Anomaly.MODE_WEIGHTED, binaryAnomalyThreshold=0.5)
    self.assertNotEqual(an, anN)
*/
