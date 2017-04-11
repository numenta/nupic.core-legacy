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

#include <iostream>

using namespace std;
using namespace::nupic;
using namespace nupic::util;


MovingAverage::MovingAverage(UInt wSize, const vector<Real32>& historicalValues)
    : slidingWindow_(wSize)
{
 for(auto v: historicalValues) {
  slidingWindow_.push_back(v);
 }
 auto window = this->getData();
  total_ = Real32(accumulate(begin(window), end(window), 0));
}


MovingAverage::MovingAverage(UInt wSize) : 
  slidingWindow_(wSize), total_(0) {}


Real32 MovingAverage::compute(Real32 newVal)
{
  if(slidingWindow_.full()) {
    total_ -= slidingWindow_.front();
    slidingWindow_.pop_front();
  }
  slidingWindow_.push_back(newVal);
  total_ += newVal;
  return getCurrentAvg();
}


std::vector<Real32> MovingAverage::getData() const 
{
//  slidingWindow_.linearize();
  auto d1 = slidingWindow_.array_one();
  vector<Real32> data(d1.first, d1.first+d1.second); //TODO improve this copy data
  auto d2 = slidingWindow_.array_two();
  data.insert(end(data), d2.first, d2.first+d2.second);

  return data;
}

boost::circular_buffer<Real32> MovingAverage::getSlidingWindow() const
{
  return slidingWindow_;
}


Real32 MovingAverage::getCurrentAvg() const
{
  return Real32(total_) / Real32(slidingWindow_.size());
}


bool MovingAverage::operator==(const MovingAverage& r2) const
{

 for(UInt i=0; i< slidingWindow_.size(); i++) 
   cout << i << ": " << slidingWindow_[i] << " vs " << r2.slidingWindow_[i] << endl;

  return ( //! slidingWindow_ == r2.slidingWindow_ &&
        //  total_ == r2.total_ && 
       //   slidingWindow_.front() == r2.slidingWindow_.front() && 
      this->getData() == r2.getData() );
}


bool MovingAverage::operator!=(const MovingAverage& r2) const
{
  return !operator==(r2);
}


Real32 MovingAverage::getTotal() const
{
  return total_;
}
