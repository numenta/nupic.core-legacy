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

      float computeRawAnomalyScore(const vector<UInt>& active, const vector<UInt>& predicted);


      enum class AnomalyMode { PURE, LIKELIHOOD, WEIGHTED };


      class Anomaly
      {
        private:
          AnomalyMode mode_;
          float binaryThreshold_;
          unique_ptr<nupic::util::MovingAverage> movingAverage_;
        public:
          Anomaly(int slidingWindowSize=0, AnomalyMode mode=AnomalyMode::PURE, 
                  float binaryAnomalyThreshold=0);
          float compute(const vector<UInt>& active, const vector<UInt>& predicted, 
                        int inputValue=0, int timestamp=0);
      };
    }
  }
}

#endif
