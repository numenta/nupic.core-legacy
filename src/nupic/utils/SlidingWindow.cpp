/* Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

#include <cmath>

#include <nupic/utils/SlidingWindow.hpp> 
#include <nupic/utils/Log.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::util;

SlidingWindow::SlidingWindow(UInt maxCapacity) :
  maxCapacity(maxCapacity) {
  NTA_CHECK(maxCapacity > 0);
  buffer_.reserve(maxCapacity);
  idxNext_ = 0;
}

SlidingWindow::SlidingWindow(UInt maxCapacity, vector<Real> initialData)  :
  SlidingWindow(maxCapacity) {
  NTA_CHECK(initialData.size() <= maxCapacity); 
  buffer_.insert(begin(buffer_), begin(initialData), end(initialData));
  idxNext_ = initialData.size();
}

UInt SlidingWindow::size() const {
  NTA_ASSERT(buffer_.size() <= maxCapacity);
  return buffer_.size();
}

Real SlidingWindow::append(Real newValue) {
  Real old = NEUTRAL_VALUE;
  if(firstRun_ && size() == maxCapacity) {
    firstRun_ = false;
  }

  if(firstRun_) { 
    buffer_.emplace_back(newValue);
  } else {
    old = buffer_[idxNext_];  //FIXME this IF is here only because size() wouldn't work w/o it or similar hack
    buffer_[idxNext_] = newValue;
  }
  idxNext_ = ++idxNext_ % maxCapacity;
  return old;
}

vector<Real> SlidingWindow::getData() const {
  return buffer_;
}

bool SlidingWindow::operator==(const SlidingWindow& r2) const
{
  return ((this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity) && this->getData()==r2.getData()); 
}


bool SlidingWindow::operator!=(const SlidingWindow& r2) const
{
  return !operator==(r2);
}

