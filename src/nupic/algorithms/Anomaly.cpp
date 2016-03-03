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
//#define NTA_ASSERT(condition) if (condition) {} else throw "Error";

using namespace std;

namespace nupic
{
  namespace algorithms
  {
    namespace anomaly
    { //FIXME for some reason googletests fail if using namespace nupic::algorithms::anomaly; is used instead!


/**
Computes the raw anomaly score.

The raw anomaly score is the fraction of active columns not predicted.

@param activeColumns: array of active column indices
@param prevPredictedColumns: array of columns indices predicted in prev step
@return anomaly score 0..1 (float)
*/
float computeRawAnomalyScore(const vector<UInt>& active, const vector<UInt>& predicted)
{

  	
  // Return 0 if no active columns are present
  if (active.size() == 0)
  {
    return 0.0f;
  }

  set<UInt> active_set {active.begin(), active.end()};
  set<UInt> predicted_set{predicted.begin(), predicted.end()};
  vector<UInt> res;

  // Calculate and return percent of active columns that were not predicted.
  set_intersection(active_set.begin(), active_set.end(),
		           predicted_set.begin(), predicted_set.end(),
                   back_inserter(res));

  return (active.size() - res.size()) / float(active.size());
}


/**
Utility class for generating anomaly scores in different ways.

  Supported modes:
    MODE_PURE - the raw anomaly score as computed by computeRawAnomalyScore
    MODE_LIKELIHOOD - uses the AnomalyLikelihood class on top of the raw
        anomaly scores
    MODE_WEIGHTED - multiplies the likelihood result with the raw anomaly score
        that was used to generate the likelihood

    @param slidingWindowSize (optional) - how many elements are summed up;
        enables moving average on final anomaly score; int >= 0
    @param mode (optional) - (string) how to compute anomaly;
        possible values are:
          - "pure" - the default, how much anomal the value is;
              float 0..1 where 1=totally unexpected
          - "likelihood" - uses the anomaly_likelihood code;
              models probability of receiving this value and anomalyScore
          - "weighted" - "pure" anomaly weighted by "likelihood"
              (anomaly * likelihood)
    @param binaryAnomalyThreshold (optional) - if set [0,1] anomaly score
         will be discretized to 1/0 (1 if >= binaryAnomalyThreshold)
         The transformation is applied after moving average is computed.
*/
Anomaly::Anomaly(int slidingWindowSize, AnomalyMode mode, float binaryAnomalyThreshold) :
				binaryThreshold_(binaryAnomalyThreshold) /*, moving_average(nullptr) */
{
  NTA_ASSERT(binaryAnomalyThreshold >= 0 && binaryAnomalyThreshold <= 1) << "binaryAnomalyThreshold must be within [0.0,1.0]";
  this->mode_ = mode;
  if (slidingWindowSize > 0) 
  {
    this->movingAverage_.reset(new nupic::util::MovingAverage(slidingWindowSize));
  }

  if (this->mode_  == AnomalyMode::LIKELIHOOD || this->mode_ == AnomalyMode::WEIGHTED)
  {
    // Not implemented. Fail.
    NTA_ASSERT(this->mode_ == AnomalyMode::PURE) << "C++ Anomaly implemented only for PURE mode!";
  }
}

/**
Compute the anomaly score as the percent of active columns not predicted.

    @param activeColumns: array of active column indices
    @param predictedColumns: array of columns indices predicted in this step
                             (used for anomaly in step T+1)
    @param inputValue: (optional) value of current input to encoders
                                (eg "cat" for category encoder)
                                (used in anomaly-likelihood)
    @param timestamp: (optional) date timestamp when the sample occured
                                (used in anomaly-likelihood)
    @return the computed anomaly score; float 0..1
*/
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
    score = this->movingAverage_->next(score);
  }

  if (this->binaryThreshold_) 
  {
    score = (score >= this->binaryThreshold_) ? 1.0 : 0.0;
  }

  return score;
}

}}} //namespaces
