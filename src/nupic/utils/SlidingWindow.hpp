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
#include <iterator>
#include <cmath>
#include <string>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/math/Math.hpp> //for macro ASSERT_INPUT_ITERATOR
#include <nupic/utils/VectorHelpers.hpp> //printVector

using nupic::utils::VectorHelpers;

namespace nupic {
  namespace util {

template<class T> 
class SlidingWindow {
  public:
    SlidingWindow(UInt maxCapacity, std::string id="SlidingWindow", int debug=0) : 
      maxCapacity(maxCapacity),
      ID(id),
      DEBUG(debug)
{
      buffer_.reserve(maxCapacity);
      idxNext_ = 0;
} 


    template<class IteratorT> 
    SlidingWindow(UInt maxCapacity, IteratorT initialData_begin, 
      IteratorT initialData_end, std::string id="SlidingWindow", int debug=0): 
      SlidingWindow(maxCapacity, id, debug) {
      // Assert that It obeys the STL forward iterator concept
      ASSERT_INPUT_ITERATOR(IteratorT);
      for(IteratorT it = initialData_begin; it != initialData_end; ++it) {
        append(*it);
      }
      idxNext_ = buffer_.size() % maxCapacity; //TODO remove, not needed
    }

    const UInt maxCapacity;
    const std::string ID; //name of this object
    const int DEBUG;

    size_t size() const {
      NTA_ASSERT(buffer_.size() <= maxCapacity);
      return buffer_.size();
    }


    /** append new value to the end of the buffer and handle the 
       "overflows"-may pop the oldest element if full. 
      */
    void append(T newValue) {
      if(size() < maxCapacity) {
        buffer_.push_back(newValue);
      } else {
        buffer_[idxNext_] = newValue;
      }
      //the assignment must be out of the [] above, so not [idxNext_++%maxCap],
      // because we want to store the value %maxCap, not only ++
      idxNext_ = (idxNext_ +1 ) %maxCapacity;
    }


      /** like append, but return the dropped value if it was dropped.
        :param T newValue - new value to append to the sliding window
        :param T& - a return pass-by-value with the removed element,
          if this function returns false, this value will remain unchanged.
        :return bool if some value has been dropped (and updated as 
          droppedValue) 
      */
      bool append(T newValue, T& droppedValue) {
        //only in this case we drop oldest; this happens always after
        //first maxCap steps ; must be checked before append()
        bool isFull = (buffer_.size()==maxCapacity);
        if(isFull) {
          droppedValue = buffer_[idxNext_];
        }
        append(newValue);
        return isFull;
      }


      /**
        :return unordered content (data ) of this sl. window; 
          call getLinearizedData() if you need them oredered from 
          oldest->newest
        This direct access method is fast.
      */
      const std::vector<T>& getData() const {
        return buffer_;
      }


      /** linearize method for the internal buffer; this is slower than 
        the pure getData() but ensures that the data are ordered (oldest at
        the beginning, newest at the end of the vector
        This handles case of |5,6;1,2,3,4| => |1,2,3,4,5,6|
        :return new linearized vector
      */
      std::vector<T> getLinearizedData() const {
        std::vector<T> lin;
        lin.reserve(buffer_.size());

        //insert the "older" part at the beginning
        lin.insert(std::begin(lin), std::begin(buffer_) + idxNext_, std::end(buffer_));
        //append the "newer" part to the end of the constructed vect
        lin.insert(std::end(lin), std::begin(buffer_), std::begin(buffer_) + idxNext_);
        return lin;
      }


      bool operator==(const SlidingWindow& r2) const {
        return ((this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity) &&
          (this->getData()== r2.getData()) && getLinearizedData() == r2.getLinearizedData());
        //FIXME review the ==, on my machine it randomly passes/fails the test!
      }


      bool operator!=(const SlidingWindow& r2) const {
        return !operator==(r2);
      }


      /** operator[] provides fast access to the elements indexed relatively
        to the oldest element. So slidingWindow[0] returns oldest element, 
        slidingWindow[size()] returns the newest. 
      :param UInt index - index/offset from the oldest element, values 0..size()
      :return T - i-th oldest value in the buffer
      :throws 0<=index<=size()
      */ 
      T& operator[](UInt index) {
        NTA_ASSERT(index <= size());
        NTA_ASSERT(size() > 0);
        //get last updated position, "current"+index(offset)
        //avoid calling getLinearizeData() as it involves copy()
        if (size() == maxCapacity) {
          return &buffer_[(idxNext_ + index) % maxCapacity];
        } else {
          return &buffer_[index];
        }
      }


      const T& operator[](UInt index) const {
        return this->operator[](index); //call the overloaded operator[] above
      }


      std::ostream& operator<<(std::ostream& os) {
        os << ID;
        if(DEBUG>0) os << "["<<VectorHelpers::printVector(buffer_) <<"]";
        os << std::endl;
        return os;
      }



      private:
        std::vector<T> buffer_;
        UInt idxNext_;
}; 
}} //end ns
#endif //header
