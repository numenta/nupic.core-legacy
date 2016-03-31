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


float computeRawAnomalyScore(const vector<UInt>& active, const vector<UInt>& predicted)
{
  	
  // Return 0 if no active columns are present
  if (active.size() == 0)
  {
    return 0.0f;
  }

  set<UInt> active_ {active.begin(), active.end()};
  set<UInt> predicted_ {predicted.begin(), predicted.end()};
  vector<UInt> predictedActiveCols;

  // Calculate and return percent of active columns that were not predicted.
  set_intersection(active_.begin(), active_.end(),
                   predicted_.begin(), predicted_.end(),
                   back_inserter(predictedActiveCols));

  return (active.size() - predictedActiveCols.size()) / float(active.size());
}


Anomaly::Anomaly(UInt slidingWindowSize, AnomalyMode mode, float binaryAnomalyThreshold) :
                 binaryThreshold_(binaryAnomalyThreshold) /*, moving_average(nullptr) */
{
  NTA_ASSERT(binaryAnomalyThreshold >= 0 && binaryAnomalyThreshold <= 1) 
    << "binaryAnomalyThreshold must be within [0.0,1.0]";
  this->mode_ = mode;
  if (slidingWindowSize > 0) 
  {
    this->movingAverage_.reset(new nupic::util::MovingAverage(slidingWindowSize));
  }

  // Not implemented. Fail.
  NTA_ASSERT(this->mode_ == AnomalyMode::PURE) << "C++ Anomaly implemented only for PURE mode!";
}


float Anomaly::compute(const vector<UInt>& active, const vector<UInt>& predicted,
                       int inputValue, int timestamp)
{
  float anomalyScore = computeRawAnomalyScore(active, predicted);
  float score = anomalyScore;
  switch(this->mode_) 
  {
    case AnomalyMode::PURE:
      score = anomalyScore;
      break;
    case AnomalyMode::LIKELIHOOD:
    case AnomalyMode::WEIGHTED:
      // Not implemented. Fail
      NTA_ASSERT(this->mode_ == AnomalyMode::PURE) << "C++ Anomaly implemented only for PURE mode!";
      break;
  }

  if (this->movingAverage_) 
  {
    this->movingAverage_->next(score);
    score = this->movingAverage_->getCurrentAvg();
  }

  if (this->binaryThreshold_) 
  {
    score = (score >= this->binaryThreshold_) ? 1.0 : 0.0;
  }

  return score;
}

}}} //namespaces
