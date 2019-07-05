#ifndef HTM_UTIL_SLIDING_WINDOW_HPP
#define HTM_UTIL_SLIDING_WINDOW_HPP

#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <string>

#include <htm/types/Serializable.hpp>
#include <htm/types/Types.hpp>
#include <htm/utils/Log.hpp>

namespace htm {

template<class T> 
class SlidingWindow : public Serializable {
  // Note: member veriables need to be declared before constructor.
  //       Otherwise we get "will be initialized after [-Werror=reorder]"
  public: 
    const UInt maxCapacity;
    const std::string ID; //name of this object
    const int DEBUG;
  private:
    std::vector<T> buffer_;
    UInt idxNext_;
				
  public:
    SlidingWindow(UInt max_capacity, std::string id="SlidingWindow", int debug=0) : 
      maxCapacity(max_capacity),
      ID(id),
      DEBUG(debug)
    {
      buffer_.reserve(max_capacity);
      idxNext_ = 0;
    } 


    template<class IteratorT> 
    SlidingWindow(UInt max_capacity, IteratorT initialData_begin, 
      IteratorT initialData_end, std::string id="SlidingWindow", int debug=0): 
      SlidingWindow(max_capacity, id, debug) {
      // Assert that It obeys the STL forward iterator concept
      for(IteratorT it = initialData_begin; it != initialData_end; ++it) {
        append(*it);
      }
    }


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
      idxNext_ = (idxNext_ +1 ) %maxCapacity;
    }


      /** like append, but return the dropped value if it was dropped.
        :param T newValue - new value to append to the sliding window
        :param T* - a return pass-by-value with the removed element,
          if this function returns false, this value will remain unchanged.
        :return bool if some value has been dropped (and updated as 
          droppedValue) 
      */
      bool append(T newValue, T* droppedValue) {
        //only in this case we drop oldest; this happens always after
        //first maxCap steps ; must be checked before append()
        const bool isFull = (buffer_.size()==maxCapacity);
        if(isFull) {
          *droppedValue = buffer_[idxNext_];
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
        const bool sameSizes = (this->size() == r2.size()) && (this->maxCapacity == r2.maxCapacity);
        if(!sameSizes) return false; 
        const std::vector<T> v1 = this->getLinearizedData(); 
        const std::vector<T> v2 = r2.getLinearizedData();
        return sameSizes && std::equal(v1.cbegin(), v1.cend(), v2.cbegin()); //also content must be same
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
      T operator[](UInt index) const {
        NTA_ASSERT(index <= size());
        NTA_ASSERT(size() > 0);
        //get last updated position, "current"+index(offset)
        //avoid calling getLinearizeData() as it involves copy()
        if (size() == maxCapacity) {
          return buffer_[(idxNext_ + index) % maxCapacity];
        } else {
          return buffer_[index];
        }
      }

      CerealAdapter;
      template<class Archive>
      void save_ar(Archive & ar) const {
        ar(CEREAL_NVP(ID), 
           CEREAL_NVP(buffer_), 
           CEREAL_NVP(idxNext_));
      }
      template<class Archive>
      void load_ar(Archive & ar) {
        std::string name; // for debugging. ID should be already set from constructor.
        ar( name, buffer_, idxNext_);
        // Note: ID, maxCapacity, DEBUG are already set from constructor.
      }
}; 
} //end ns
#endif //header
