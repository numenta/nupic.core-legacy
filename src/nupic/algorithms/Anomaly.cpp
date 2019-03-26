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
#include <iterator>
#include <numeric>
#include <set>
#include <vector>

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/utils/Log.hpp"
#include "nupic/utils/MovingAverage.hpp"
#include "nupic/types/SdrTools.hpp" // sdr::Intersection

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::anomaly;
using namespace nupic::sdr;

namespace nupic {
namespace algorithms {
namespace anomaly {
Real32 computeRawAnomalyScore(SDR& active,
                              SDR& predicted) {

  // Return 0 if no active columns are present
  if (active.getSum() == 0) {
    return 0.0f;
  }

  NTA_CHECK(active.dimensions == predicted.dimensions);

  // Calculate and return percent of active columns that were not predicted.
  Intersection both(active, predicted); //TODO allow Intersection to act on const SDR, so active,pred can be const

  cout << "act " << active << endl;
  cout << "pred " << predicted << endl;
  cout << "x " << both;

  return (active.getSum() - both.getSum()) / Real(active.getSum());
}

Real32 computeRawAnomalyScore(const vector<UInt>& active,
                              const vector<UInt>& pred) {
  SDR a({static_cast<UInt>(active.size())});
  a.setSparse(active);

  SDR p({static_cast<UInt>(pred.size())});
  p.setSparse(pred);

  return computeRawAnomalyScore(a, p);
}

}}} //end ns

Anomaly::Anomaly(UInt slidingWindowSize, AnomalyMode mode, Real binaryAnomalyThreshold)
    : binaryThreshold_(binaryAnomalyThreshold)
{
  NTA_CHECK(binaryAnomalyThreshold >= 0 && binaryAnomalyThreshold <= 1) << "binaryAnomalyThreshold must be within [0.0,1.0]";
  mode_ = mode;
  if (slidingWindowSize > 0) {
    movingAverage_.reset(new nupic::util::MovingAverage(slidingWindowSize));
  }
}


Real Anomaly::compute(const vector<UInt>& active, const vector<UInt>& predicted, int timestamp)
{
  Real anomalyScore = computeRawAnomalyScore(active, predicted);
  Real likelihood = 0.5;
  Real score = anomalyScore;
  switch(mode_)
  {
    case AnomalyMode::PURE:
      score = anomalyScore;
      break;
    case AnomalyMode::LIKELIHOOD:
      likelihood = likelihood_.anomalyProbability(anomalyScore, timestamp);
      score = 1 - likelihood;
      break;
    case AnomalyMode::WEIGHTED:
      likelihood = likelihood_.anomalyProbability(anomalyScore, timestamp);
      score = anomalyScore * (1 - likelihood);
      break;
  }

  if (movingAverage_) {
    score = movingAverage_->compute(score);
  }

  if (binaryThreshold_) {
    score = (score >= binaryThreshold_) ? 1.0f : 0.0f;
  }

  return score;
}

