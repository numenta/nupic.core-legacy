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
using namespace::nupic;
using namespace nupic::util;


MovingAverage::MovingAverage(UInt wSize, const vector<Real32>& historicalValues)
    : slidingWindow_(wSize, historicalValues)
{
 auto window = slidingWindow_.getData();
  total_ = Real32(accumulate(begin(window), end(window), 0));
}


MovingAverage::MovingAverage(UInt wSize) : 
  slidingWindow_(wSize), total_(0) {}


Real32 MovingAverage::compute(Real32 newVal)
{

  auto popped = slidingWindow_.append(newVal);
  total_ -= popped;
  total_ += newVal;
  return getCurrentAvg();
}


std::vector<Real32> MovingAverage::getData() const
{
  return slidingWindow_.getData();
}

SlidingWindow MovingAverage::getSlidingWindow() const
{
  return slidingWindow_;
}


Real32 MovingAverage::getCurrentAvg() const
{
  return Real32(total_) / Real32(slidingWindow_.size());
}


bool MovingAverage::operator==(const MovingAverage& r2) const
{
  return (slidingWindow_ == r2.slidingWindow_ &&
          total_ == r2.total_);
}


bool MovingAverage::operator!=(const MovingAverage& r2) const
{
  return !operator==(r2);
}


Real32 MovingAverage::getTotal() const
{
  return total_;
}
