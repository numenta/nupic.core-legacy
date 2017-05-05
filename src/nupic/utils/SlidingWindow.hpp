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


#ifndef NUPIC_UTIL_SLIDING_WINDOW_HPP
#define NUPIC_UTIL_SLIDING_WINDOW_HPP

#include <vector>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
  namespace util {

    template<class T> 
    class SlidingWindow {
      public:
        SlidingWindow(UInt maxCapacity);
        SlidingWindow(UInt maxCapacity, std::vector<T> initialData);
        const UInt maxCapacity;
        const T NEUTRAL_VALUE = 0.0;
        UInt size() const;
        /** append new value to the end of the buffer and handle the "overflows"-may pop the first element if full. 
         :return addition(+) neutral value (that is 0) when size()< maxCapacity; when full - return the value of the dropped element  
        */
        T append(T newValue);
        /**
        :return unordered content (data ) of this sl. window; call linearize() if you need them oredered from oldest->newest
        */
        const std::vector<T>& getData() const;

        bool operator==(const SlidingWindow& r2) const;
        bool operator!=(const SlidingWindow& r2) const;

    // HELPER:
     static std::vector<T> getLastNValues(std::vector<T> biggerData, nupic::UInt n) {
      NTA_CHECK(n >=0 && n <= biggerData.size()); 
    std::vector<T> v;
    v.reserve(n);
    v.insert(begin(v), end(biggerData) - n, end(biggerData));
    NTA_ASSERT(v.size() == n);
    return v;
    };

      private:
        std::vector<T> buffer_;
        UInt idxNext_;
        bool firstRun_ = true; //fill be false once "first run" (=size reaches maxCapacity for the first time) is completed
}; 
}} //end ns

/// IMPLEMENTATION
#include <cmath>

using namespace std;
using namespace nupic;
using namespace nupic::util;

template<class T>
SlidingWindow<T>::SlidingWindow(UInt maxCapacity) :
  maxCapacity(maxCapacity) {
  NTA_CHECK(maxCapacity > 0);
  buffer_.reserve(maxCapacity);
  idxNext_ = 0;
}


template<class T>
SlidingWindow<T>::SlidingWindow(UInt maxCapacity, vector<T> initialData)  :
  SlidingWindow(maxCapacity) {
  NTA_CHECK(initialData.size() <= maxCapacity);
  buffer_.insert(begin(buffer_), begin(initialData), end(initialData));
  idxNext_ = initialData.size();
}


template<class T>
UInt SlidingWindow<T>::size() const {
  NTA_ASSERT(buffer_.size() <= maxCapacity);
  return buffer_.size();
}


template<class T>
T SlidingWindow<T>::append(T newValue) {
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


template<class T>
const vector<T>& SlidingWindow<T>::getData() const {
  return buffer_;
}


template<class T>
bool SlidingWindow<T>::operator==(const SlidingWindow& r2) const //FIXME review the ==, on my machine it randomly passes/fails the test!
{
  bool sameData = this->getData()== r2.getData();
  return ((this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity) && sameData);
}


template<class T>
bool SlidingWindow<T>::operator!=(const SlidingWindow& r2) const
{
  return !operator==(r2);
}

#endif //header
