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
#include "nupic/algorithms/Anomaly.hpp"

using namespace std;
using namespace nupic::algorithms::anomaly;

/**
Computes the raw anomaly score.

The raw anomaly score is the fraction of active columns not predicted.

@param activeColumns: array of active column indices
@param prevPredictedColumns: array of columns indices predicted in prev step
@return anomaly score 0..1 (float)
*/
float computeRawAnomalyScore(const vector<int>& active, const vector<int>& predicted)
{
  // Return 0 if no active columns are present
  if (active.size() == 0) {
    return 0.0f;
  }
	
  // Count active columns that were predicted
  int score = accumulate(active.begin(), active.end(), 0,
			 [predicted](int accum, int curr) -> int { 
			   return ((find(predicted.begin(), predicted.end(), curr) != predicted.end()) 
						? accum + 1 : accum); 
			        }
			 );
			
  // Calculate and return percent of active columns that were not predicted.
  return (active.size() - score) / float(active.size());
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
Anomaly::Anomaly(int slidingWindowSize, AnomalyMode mode, float binary_anomaly_threshold) :
				binary_threshold(binary_anomaly_threshold) /*, moving_average(nullptr) */
{
// TODO: Raise an Exception if the threshold value is not > 0 && < 1
  this->mode = mode;
  if (slidingWindowSize != 0) {
    this->moving_average.reset(new nupic::util::MovingAverage(slidingWindowSize));
  }

  if (this->mode  == AnomalyMode::LIKELIHOOD || this->mode == AnomalyMode::WEIGHTED) {
  // Not implemented
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
float Anomaly::compute(vector<int>& active, vector<int>& predicted,
		       int inputValue, int timestamp)
{
  float score;
  float anomaly_score = computeRawAnomalyScore(active, predicted);
  switch(this->mode) {
    case AnomalyMode::PURE:
      score = anomaly_score;
      break;
    case AnomalyMode::LIKELIHOOD:
    case AnomalyMode::WEIGHTED:
      // Not implemented
      break;
}

  if (this->moving_average) {
    score = this->moving_average->next(score);
  }

  if (this->binary_threshold) {
    score = (score >= this->binary_threshold) ? 1.0 : 0.0;
  }

  return score;
}
