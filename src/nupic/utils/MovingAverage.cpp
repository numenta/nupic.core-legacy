/*
 * Copyright 2016 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#include "nupic/utils/MovingAverage.hpp"
#include "nupic/utils/Log.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

using namespace std;
using namespace ::nupic;
using namespace nupic::util;

MovingAverage::MovingAverage(UInt wSize, const vector<Real32> &historicalValues)
    : windowSize_(wSize) {
  if (historicalValues.size() != 0) {
    copy(historicalValues.begin() + historicalValues.size() - wSize,
         historicalValues.end(), back_inserter(slidingWindow_));
  }
  total_ = Real32(accumulate(slidingWindow_.begin(), slidingWindow_.end(), 0));
}

MovingAverage::MovingAverage(UInt wSize) : windowSize_(wSize), total_(0) {}

Real32 MovingAverage::compute(Real32 newVal) {
  if (windowSize_ == slidingWindow_.size()) {
    total_ -= slidingWindow_.front();
    slidingWindow_.erase(slidingWindow_.begin()); // pop front element
  }

  slidingWindow_.push_back(newVal);
  total_ += newVal;
  return getCurrentAvg();
}

std::vector<Real32> MovingAverage::getSlidingWindow() const {
  return slidingWindow_;
}

Real32 MovingAverage::getCurrentAvg() const {
  return Real32(total_) / Real32(slidingWindow_.size());
}

bool MovingAverage::operator==(const MovingAverage &r2) const {
  return (windowSize_ == r2.windowSize_ &&
          slidingWindow_ == r2.slidingWindow_ && total_ == r2.total_);
}

bool MovingAverage::operator!=(const MovingAverage &r2) const {
  return !operator==(r2);
}

Real32 MovingAverage::getTotal() const { return total_; }
