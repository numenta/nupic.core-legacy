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
#include <numeric>
#include <algorithm>
#include <iterator>
#include <set>

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/utils/Log.hpp"
#include "nupic/utils/MovingAverage.hpp"

using namespace std;

namespace nupic
{

namespace algorithms
{

namespace anomaly
{


Real32 computeRawAnomalyScore(const vector<UInt>& active,
                              const vector<UInt>& predicted)
{
  // Return 0 if no active columns are present
  if (active.size() == 0)
  {
    return 0.0f;
  }

  set<UInt> active_{active.begin(), active.end()};
  set<UInt> predicted_{predicted.begin(), predicted.end()};
  vector<UInt> predictedActiveCols;

  // Calculate and return percent of active columns that were not predicted.
  set_intersection(active_.begin(), active_.end(),
                   predicted_.begin(), predicted_.end(),
                   back_inserter(predictedActiveCols));

  return (active.size() - predictedActiveCols.size()) / Real32(active.size());
}


Anomaly::Anomaly(UInt slidingWindowSize, AnomalyMode mode,
                 Real32 binaryAnomalyThreshold)
    : binaryThreshold_(binaryAnomalyThreshold)
{
  NTA_ASSERT(binaryAnomalyThreshold >= 0 && binaryAnomalyThreshold <= 1)
      << "binaryAnomalyThreshold must be within [0.0,1.0]";
  mode_ = mode;
  if (slidingWindowSize > 0)
  {
    movingAverage_.reset(new nupic::util::MovingAverage(slidingWindowSize));
  }

  // Not implemented. Fail.
  NTA_ASSERT(mode_ == AnomalyMode::PURE)
      << "C++ Anomaly implemented only for PURE mode!";
}


Real32 Anomaly::compute(
    const vector<UInt>& active, const vector<UInt>& predicted,
    Real64 inputValue, UInt timestamp)
{
  Real32 anomalyScore = computeRawAnomalyScore(active, predicted);
  Real32 score = anomalyScore;
  switch(mode_)
  {
    case AnomalyMode::PURE:
      score = anomalyScore;
      break;
    case AnomalyMode::LIKELIHOOD:
    case AnomalyMode::WEIGHTED:
      // Not implemented. Fail
      NTA_ASSERT(mode_ == AnomalyMode::PURE)
          << "C++ Anomaly implemented only for PURE mode!";
      break;
  }

  if (movingAverage_)
  {
    score = movingAverage_->compute(score);
  }

  if (binaryThreshold_)
  {
    score = (score >= binaryThreshold_) ? 1.0 : 0.0;
  }

  return score;
}

} // namespace anomaly

} // namespace algorithms

} // namespace nupic
