/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

#include "htm/utils/MovingAverage.hpp"
#include "htm/utils/Log.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath> //isnan

using namespace std;
using namespace htm;


MovingAverage::MovingAverage(UInt wSize, const vector<Real>& historicalValues)
    : slidingWindow_(wSize, begin(historicalValues), end(historicalValues)) {
 const std::vector<Real>&  window = slidingWindow_.getData();
  total_ = Real(accumulate(begin(window), end(window), 0.0f));
}


MovingAverage::MovingAverage(UInt wSize) : slidingWindow_(wSize), total_(0) {}


Real MovingAverage::compute(Real newVal) {
  NTA_CHECK(not std::isnan(newVal));

  Real droppedVal = 0.0;
  const bool hasDropped = slidingWindow_.append(newVal, &droppedVal);
  if(hasDropped) { total_ -= droppedVal; }
  total_ += newVal;
  return getCurrentAvg();
}

Real MovingAverage::getCurrentAvg() const {
  if(slidingWindow_.size() == 0) {
    return 0.0f; //avoid division by zero/nan! 
  }
  return total_ / static_cast<Real>(slidingWindow_.size());
}

