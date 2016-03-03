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
using namespace nupic::util;


MovingAverage::MovingAverage(int wSize, const vector<float>& historicalValues) :
	windowSize_(wSize)
{
  NTA_ASSERT(wSize > 0) << "wSize must be > 0";
  if (historicalValues.size() != 0)
  {
    copy(
      historicalValues.begin() + historicalValues.size() - wSize,
      historicalValues.end(),
      back_inserter(this->slidingWindow_));
  }
  this->total_ = float(accumulate(this->slidingWindow_.begin(), this->slidingWindow_.end(), 0)); 
}


MovingAverage::MovingAverage(int wSize) : windowSize_(wSize), total_(0) {}


std::tuple<float, float> MovingAverage::compute(std::vector<float>& slidingWindow, 
  float total, float newVal, unsigned int windowSize)
{
  if (slidingWindow.size() == windowSize) 
  {
    total -= slidingWindow.front();
    slidingWindow.erase(slidingWindow.begin()); // pop front element.
  }

  slidingWindow.push_back(newVal);
  total += newVal;

  return std::make_tuple(total / float(slidingWindow.size()), total);
}

std::vector<float> MovingAverage::getSlidingWindow()
{
  return this->slidingWindow_;
}

float MovingAverage::getCurrentAvg()
{
  return float(this->total_) / float(this->slidingWindow_.size());
}

float MovingAverage::next(float newValue)
{
  float newAverage;
  std::tie(newAverage, this->total_) = this->compute(this->slidingWindow_,
    this->total_, newValue, this->windowSize_);
  return newAverage;
}

bool MovingAverage::operator==(MovingAverage& r2)
{
  // TODO
  return true;
}

float MovingAverage::getTotal()
{
  return this->total_;
}


