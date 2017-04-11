/* Numenta Platform for Intelligent Computing (NuPIC)
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

#include <cmath>
#include <cassert>

#include <nupic/utils/SlidingWindow.hpp> 

using namespace std;
using namespace nupic;
using namespace nupic::util;

SlidingWindow::SlidingWindow(UInt maxCapacity) :
  maxCapacity(maxCapacity) {}

SlidingWindow::SlidingWindow(UInt maxCapacity, vector<Real> initialData)  :
  SlidingWindow(maxCapacity) {
  this->buffer.insert(std::end(buffer), std::begin(initialData), std::end(initialData));
  UInt toRemove = max(0ul, initialData.size() - maxCapacity); //crop to maxCapacity
  if(toRemove > 0) {
    buffer.erase(begin(buffer), end(buffer) + toRemove);
  }
  assert(this->size() <= this->maxCapacity);
}

UInt SlidingWindow::size() const {
  return buffer.size();
}

Real SlidingWindow::append(Real newValue) {
  buffer.push_back(newValue);
  Real old = NEUTRAL_VALUE;

  if(size() > maxCapacity) { //pop oldest to remove space
    old = buffer.front();
    buffer.erase(begin(buffer)); 
  }
  return old;
}

vector<Real> SlidingWindow::getData() const {
  return buffer;
}

bool SlidingWindow::operator==(const SlidingWindow& r2) const
{
  return ((this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity) && this->getData()==r2.getData()); 
}


bool SlidingWindow::operator!=(const SlidingWindow& r2) const
{
  return !operator==(r2);
}

