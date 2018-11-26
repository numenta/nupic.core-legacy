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

#include "nupic/utils/MovingAverage.hpp"
#include "nupic/utils/Log.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;
using namespace nupic;
using namespace nupic::util;


MovingAverage::MovingAverage(UInt wSize, const vector<Real>& historicalValues)
    : slidingWindow_(wSize, begin(historicalValues), end(historicalValues)) {
 const std::vector<Real>&  window = slidingWindow_.getData();
  total_ = Real(accumulate(begin(window), end(window), 0));
}


MovingAverage::MovingAverage(UInt wSize) : slidingWindow_(wSize), total_(0) {}


Real MovingAverage::compute(Real newVal) {
  Real droppedVal = 0.0;
  const bool hasDropped = slidingWindow_.append(newVal, &droppedVal);
  if(hasDropped) { total_ -= droppedVal; }
  total_ += newVal;
  return getCurrentAvg();
}
