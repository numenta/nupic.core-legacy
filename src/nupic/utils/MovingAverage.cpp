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
#include <algorithm>

using namespace std;
using namespace nupic::util;


MovingAverage::MovingAverage(int w_size, const vector<float>& historical_values) :
	window_size(w_size)
{
  // TODO: Fail if window size is negative.
  if (historical_values.size() != 0) {
    copy(
      historical_values.begin() + historical_values.size() - w_size,
      historical_values.end(),
      back_inserter(this->sliding_window));
  }
  this->total = float(accumulate(this->sliding_window.begin(), this->sliding_window.end(), 0)); 
}


MovingAverage::MovingAverage(int w_size) : window_size(w_size), total(0) {}


std::tuple<float, float> MovingAverage::compute(std::vector<float>& sliding_window, 
  float total, float new_val, unsigned int window_size)
{
  if (sliding_window.size() == window_size) {
    total -= sliding_window.front();
    sliding_window.erase(sliding_window.begin()); // pop front element.
  }

  sliding_window.push_back(new_val);
  total += new_val;

  return std::make_tuple(total / float(sliding_window.size()), total);
}

std::vector<float> MovingAverage::get_sliding_window()
{
  return this->sliding_window;
}

float MovingAverage::get_current_avg()
{
  return float(this->total) / float(this->sliding_window.size());
}

float MovingAverage::next(float newValue)
{
  float newAverage;
  std::tie(newAverage, this->total) = this->compute(this->sliding_window,
    this->total, newValue, this->window_size);
  return newAverage;
}

bool MovingAverage::operator==(MovingAverage& r2)
{
  // TODO
  return true;
}

float MovingAverage::get_total()
{
  return this->total;
}

/*
  def __eq__(self, o):
    return (isinstance(o, MovingAverage) and
            o.slidingWindow == self.slidingWindow and
            o.total == self.total and
            o.windowSize == self.windowSize)

  def __call__(self, value):
    return self.next(value)

  @classmethod
  def read(cls, proto):
    movingAverage = object.__new__(cls)
    movingAverage.windowSize = proto.windowSize
    movingAverage.slidingWindow = list(proto.slidingWindow)
    movingAverage.total = proto.total
    return movingAverage

  def write(self, proto):
    proto.windowSize = self.windowSize
    proto.slidingWindow = self.slidingWindow
    proto.total = self.total
*/
