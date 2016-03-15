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

#ifndef ANOMALY_HPP
#define ANOMALY_HPP


#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <nupic/types/Types.hpp>
#include "nupic/utils/MovingAverage.hpp"

using namespace std;
using namespace nupic;

namespace nupic {
  namespace algorithms {
    namespace anomaly {

      /**
       * Computes the raw anomaly score.
       *
       * The raw anomaly score is the fraction of active columns not predicted.
       *
       * @param activeColumns: array of active column indices
       * @param prevPredictedColumns: array of columns indices predicted in prev step
       * @return anomaly score 0..1 (float)
       */
      float computeRawAnomalyScore(const vector<UInt>& active, const vector<UInt>& predicted);


      enum class AnomalyMode { PURE, LIKELIHOOD, WEIGHTED };


      class Anomaly
      {
        private:
          AnomalyMode mode_;
          float binaryThreshold_;
          unique_ptr<nupic::util::MovingAverage> movingAverage_;
        public:
         /**
          * Utility class for generating anomaly scores in different ways.
          *
          * Supported modes:
          *    PURE - the raw anomaly score as computed by computeRawAnomalyScore
          *    LIKELIHOOD - uses the AnomalyLikelihood class on top of the raw
          *        anomaly scores (not implemented in C++)
          *    WEIGHTED - multiplies the likelihood result with the raw anomaly score
          *        that was used to generate the likelihood (not implemented in C++)
          *
          *    @param slidingWindowSize (optional) - how many elements are summed up;
          *        enables moving average on final anomaly score; int >= 0
          *    @param mode (optional) - (enum) how to compute anomaly;
          *        possible values are AnomalyMode::
          *          - PURE - the default, how much anomal the value is;
          *              float 0..1 where 1=totally unexpected
          *          - LIKELIHOOD - uses the anomaly_likelihood code;
          *              models probability of receiving this value and anomalyScore
          *          - WEIGHTED - "pure" anomaly weighted by "likelihood"
          *              (anomaly * likelihood)
          *    @param binaryAnomalyThreshold (optional) - if set [0,1] anomaly score
          *         will be discretized to 1/0 (1 if >= binaryAnomalyThreshold)
          *         The transformation is applied after moving average is computed.
          */
          Anomaly(int slidingWindowSize=0, AnomalyMode mode=AnomalyMode::PURE, 
                  float binaryAnomalyThreshold=0);

         /**
          * Compute the anomaly score as the percent of active columns not predicted.
          *
          * @param active: array of active column indices
          * @param predicted: array of columns indices predicted in this step
          *        (used for anomaly in step T+1)
          * @param inputValue: (optional) value of current input to encoders
          *                    (eg "cat" for category encoder)
          *                    (used in anomaly-likelihood)
          * @param timestamp: (optional) date timestamp when the sample occured
          *                   (used in anomaly-likelihood)
          * @return the computed anomaly score; float 0..1
          */
          float compute(const vector<UInt>& active, const vector<UInt>& predicted, 
                        int inputValue=0, int timestamp=0);
      };
    }
  }
}

#endif
