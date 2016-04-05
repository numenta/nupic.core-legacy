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
#include <numeric>

using namespace std;
using namespace::nupic;
using namespace nupic::util;


MovingAverage::MovingAverage(UInt wSize, const vector<Real32>& historicalValues) :
  windowSize_(wSize)
{
  if (historicalValues.size() != 0)
  {
    copy(
      historicalValues.begin() + historicalValues.size() - wSize,
      historicalValues.end(),
      back_inserter(this->slidingWindow_));
  }
  this->total_ = Real32(accumulate(this->slidingWindow_.begin(), this->slidingWindow_.end(), 0)); 
}


MovingAverage::MovingAverage(UInt wSize) : windowSize_(wSize), total_(0) {}


Real32 MovingAverage::compute(Real32 newVal)
{
  if (windowSize_ == slidingWindow_.size())
  {
    total_ -= slidingWindow_.front();
    slidingWindow_.erase(slidingWindow_.begin()); // pop front element.
  }

  slidingWindow_.push_back(newVal);
  total_ += newVal;
  return this->getCurrentAvg();
}


std::vector<Real32> MovingAverage::getSlidingWindow() const
{
  return this->slidingWindow_;
}


Real32 MovingAverage::getCurrentAvg() const
{
  return Real32(this->total_) / float(this->slidingWindow_.size());
}


bool MovingAverage::operator==(const MovingAverage& r2) const
{
  return (this->windowSize_ == r2.windowSize_ &&
          this->slidingWindow_ == r2.slidingWindow_ &&
          this->total_ == r2.total_);
}


bool MovingAverage::operator!=(const MovingAverage& r2) const
{
  return !(*this == r2);
}


Real32 MovingAverage::getTotal() const
{
  return this->total_;
}