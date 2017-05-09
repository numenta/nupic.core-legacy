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


#ifndef NUPIC_UTIL_SLIDING_WINDOW_HPP
#define NUPIC_UTIL_SLIDING_WINDOW_HPP

#include <vector>
#include <algorithm>

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
        size_t size() const;
        /** append new value to the end of the buffer and handle the "overflows"-may pop the first element if full. 
        */
        void append(T newValue);

        /** like append, but return the dropped value. isValid indicates if the return value is valid (not while size()< maxCapacity)
          :param T newValue - new value to append to the sliding window
          :param bool isValid - a return pass-by-value that indicates validity of the return T value. for first maxCapacity items it is false, later always true.
          :return T dropped value (past oldest element) if isValid; if not valid, this field holds the oldest value (but still contained in the window!)
        */
//        T append(T newValue, const bool& isValid);

        /**
        * :return unordered content (data ) of this sl. window; call getLinearizedData() if you need them oredered from oldest->newest
        * This direct access method is fast.
        */
        const std::vector<T>& getData() const;

        /** linearize method for the internal buffer; this is slower than the pure getData() but ensures that the data are ordered (oldest at the 
         * beginning, newest at the end of the vector
         * This handles case of |5,6;1,2,3,4| => |1,2,3,4,5,6|
         * :return new linearized vector
        */
        std::vector<T> getLinearizedData() const;

        bool operator==(const SlidingWindow& r2) const;
        bool operator!=(const SlidingWindow& r2) const;
        T& operator[](UInt index);
        const T& operator[](UInt index) const;


      private:
        std::vector<T> buffer_;
        UInt idxNext_;
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
  buffer_.insert(begin(buffer_), end(initialData) - std::min(initialData.size(), (size_t)maxCapacity), end(initialData));
  idxNext_ = initialData.size();
}


template<class T>
size_t SlidingWindow<T>::size() const {
  NTA_ASSERT(buffer_.size() <= maxCapacity);
  return buffer_.size();
}


template<class T>
void SlidingWindow<T>::append(T newValue) {
  if(buffer_.size() < maxCapacity) {
    buffer_.emplace_back(newValue); //FIXME this IF is here only because size() wouldn't work w/o it or similar hack
  } else {
    buffer_[idxNext_++ % maxCapacity] = newValue;
  }
}


template<class T>
const vector<T>& SlidingWindow<T>::getData() const {
  return buffer_;
}


template<class T>
vector<T> SlidingWindow<T>::getLinearizedData() const {
  vector<T> lin;
  lin.reserve(buffer_.size());

  lin.insert(begin(lin), begin(buffer_) + idxNext_, end(buffer_)); //insert the "older" part at the beginning
  lin.insert(end(lin), begin(buffer_), begin(buffer_) + idxNext_); //append the "newer" part to the end of the constructed vect
 return lin;
}


template<class T>
bool SlidingWindow<T>::operator==(const SlidingWindow& r2) const //FIXME review the ==, on my machine it randomly passes/fails the test!
{
  return ((this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity) && 
    (this->getData()== r2.getData()) );
}


template<class T>
bool SlidingWindow<T>::operator!=(const SlidingWindow& r2) const
{
  return !operator==(r2);
}


template<class T> 
T& SlidingWindow<T>::operator[](UInt index) {
  NTA_ASSERT(index <= size());
  return &buffer_[index];
}

template<class T>
const T& SlidingWindow<T>::operator[](UInt index) const {
  return this->operator[](index); //call the overloaded operator[] above 
}

#endif //header
