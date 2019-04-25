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

#include <algorithm>
#include <vector>
#include <sstream>

#include "gtest/gtest.h"

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/types/Types.hpp"

namespace testing {

using namespace nupic::algorithms::anomaly;
using namespace nupic;

TEST(ComputeRawAnomalyScore, NoActiveOrPredicted) {
  std::vector<UInt> active;
  std::vector<UInt> predicted;
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0);
};

TEST(ComputeRawAnomalyScore, NoActive) {
  std::vector<UInt> active;
  std::vector<UInt> predicted = {3, 5};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0f);
};

TEST(ComputeRawAnomalyScore, PerfectMatch) {
  std::vector<UInt> active = {3, 5, 7};
  std::vector<UInt> predicted = {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 0.0f);
};

TEST(ComputeRawAnomalyScore, NoMatch) {
  std::vector<UInt> active = {2, 4, 6};
  std::vector<UInt> predicted = {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 1.0f);
};

TEST(ComputeRawAnomalyScore, PartialMatch) {
  std::vector<UInt> active = {2, 3, 6};
  std::vector<UInt> predicted = {3, 5, 7};
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 2.0f / 3.0f);
};

TEST(ComputeRawAnomalyScore, PartialMatchSDR) {
  sdr::SDR active({20});       active.setSparse(std::vector<UInt>{2, 3, 6});
  sdr::SDR predicted({20}); predicted.setSparse(std::vector<UInt>{3, 5, 7});
  ASSERT_FLOAT_EQ(computeRawAnomalyScore(active, predicted), 2.0f / 3.0f);
};

}
