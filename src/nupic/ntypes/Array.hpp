/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

// ---
//
// Definitions for the Array class
//
// It is a sub-class of ArrayBase that owns its buffer.
// This object is a container for most data sets that get passed around between
// regions. This container can hold an array of any of the NTA_BasicType types.
// It does not use templates so the getBuffer() function which returns a void*
// pointer will have to cast to the appropreate type.
//
// ASSIGNMENT:
// If an Array object is assigned, it makes a new instance of itself but does
// NOT copy the buffer. Both instances of Array point to the same buffer.  Only
// when all instances of the original object are deleted will the buffer be
// deleted.  (uses smart pointers, std::shared_ptr())
//
// Array B = A;
//
// Returning an Array from a function will also make an assignment (the Array
// inside the function is deleted when it went out of scope).
//
// BUFFER CHANGES:
// The buffer in an Array can be modified so if any instance of the Array
// changes the buffer then all instances see the changes. The buffer is modified
// using the getBuffer() function.
//    NTA_BasicType type = A.getType();
//    size_t count = A.getCount();
//    void *ptr = A.getBuffer();    Then change the contents of the array
//    pointed to by ptr.
//
//    A.zeroBuffer();               This will fill the buffer with 0's.
//
// RELEASING BUFFER:
// Note that re-allocating the buffer or releasing the buffer will disconnect it
// from other Array object instances. Those instances will continue to see the
// old buffer. The old buffer will be deleted only when all instances have
// released the buffer or were deleted.
//    A.allocateBuffer(count);      A new buffer is created (old one released if
//    there was one)
//                                  Then use getBuffer() to fetch the pointer
//                                  and populate the buffer.
//
//    A.releaseBuffer();            The buffer is released for this instance
//    only.
//
//    delete A;                     The buffer is release for this instance but
//    other instances unaffected.
//
// FULL COPY OF BUFFER
// You can make a FULL or deep copy of the buffer either using the type 2
// constructor for Array or the copy() function.
//     Array B(A.getType(), A.getBuffer(), A.getCount());
// or
//     Array B = A.copy();
// This full copy will not be affected by changes to the original buffer.
//
// TYPE CONVERSION:
// You can convert a buffer using the convertInto() or as() functions
// The as() function will allocate a new buffer convert this buffer into another
// type. The convertInto() function trys to keep the same output buffer and just
// converts into the existing buffer. However, if the existing buffer is too
// small it will reallocate the buffer.
//
// SERIALIZATION
// Two serialization methods are supported.
// The simplest is a std::istream/std::ostream interface.
//
//      Array A;
//        ... populate A
//      std::ofstream &f = bundle.getOutputStream("Region");
//      f << A;      // serializes A
// or
//      std::ifstream &f = bundle.getInputStream("Region");
//      Array A(type);
//      f >> A;      // deserializes A
//
//  The second method is YAML serialization.
//  For example, see Network.cpp
//
//     Array A;
//        ...A gets populated
//     YAML::Emitter out;
//     out << YAML::BeginDoc;
//       ...
//     A.serialize(out);      // serializes Array A
//       ...
//     out << YAML::EndDoc;
//    OFStream f;
//    f.open(filename.c_str());
//    f << out.c_str();
//    f.close();
// or
//    const YAML::Node doc = YAML::LoadFile(filename);
//      ...
//    Array A(type);
//    A.deserialize(doc);      // deserializes Array A, populating buffer.
//      ...
// ---

#ifndef NTA_ARRAY_HPP
#define NTA_ARRAY_HPP

#include <cstring>
#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
class Array : public ArrayBase {
public:
  /**
   * default constructor.
   */
  Array() : ArrayBase(NTA_BasicType_Int32) {}

  /**
   * Initialize by copying in from a raw C-type buffer.  See ArrayBase
   */
  template <typename T>
  Array(NTA_BasicType type, T *buffer, size_t count)
      : ArrayBase(type, buffer, count) {}

  /**
   * constructor for empty Array with a known type.
   */
  Array(NTA_BasicType type) : ArrayBase(type) {}

  /**
   * constructor for Array object containing an SDR.
   */
  Array(SDR &sdr) : ArrayBase(NTA_BasicType_SDR, &sdr, 0) {}

  // copy constructor
  // by default, a copy constructor is a shallow copy which would result in two
  // Array objects pointing to the same buffer. However, this is stored in a
  // shared smart pointer so both Array objects must be deleted before the
  // buffer is deleted. So the default copy and assignment constructor is ok.
  //Array(const Array &other) : ArrayBase(other) 
  //Array& operator=(const Array& other)


  /////////////////////////////////////
  //   Copy tools

  /**
   * There are times when we do want a deep copy.  The copy() function
   * will make a full copy of the buffer and becomes the buffer owner.
   */
  Array copy() const {
    Array a(type_);
    if (count_ > 0) {
      a.allocateBuffer(count_);
      memcpy((char *)a.getBuffer(), (char *)getBuffer(),
             count_ * BasicType::getSize(type_));
    }
    return a;
  }

  /**
   * copies the buffer into the Array, setting a new type.
   */
  void copyFrom(NTA_BasicType type, void *buf, size_t size) {
    type_ = type;
    allocateBuffer(size);
    memcpy((char *)getBuffer(), (char *)buf,
           count_ * BasicType::getSize(type_));
  }

  /**
   * This will do a shallow copy into the Array in the argument.
   * This is for when we do not want to replace the Array object but
   * want it to become a shared buffer instance.
   * The buffer data is not copied, but becomes shared.
   * Same as an assignment.
   */
  void share(Array &a) {
    a.buffer_ = buffer_; // copies the shared_ptr
    a.count_ = count_;
    a.capacity_ = capacity_;
    a.type_ = type_;
  }

  /**
   * Convert to a vector; copies buffer,
   * example: vector<Int32> v = array.asVector<Int32>();
   */
  template <typename T> std::vector<T> asVector() const {
    NTA_CHECK(type_ == BasicType::getType<T>())
        << "Expected an Array with type of " << BasicType::getName<T>();
    std::vector<T> v((T *)getBuffer(), (T *)getBuffer() + count_);
    return v;
  }

  /**
   * from a vector; copies buffer into this object,
   *   example:   array.fromVector(v);
   */
  template <typename T> void fromVector(std::vector<T> &vect) {
    type_ = BasicType::getType<T>();
    allocateBuffer(vect.size());
    memcpy(getBuffer(), vect.data(), count_ * BasicType::getSize(type_));
  }

  /**
   * Type conversion
   * This will do a full copy, converting to target type,
   * and return the new Array.
   */
  Array get_as(NTA_BasicType type) const {
    Array a(type);
    a.allocateBuffer(count_);
    convertInto(a);
    return a;
  }

  /**
   * Type conversion
   * This will do a full copy into a shared buffer.
   * Note: this will attempt to reuse the same buffer
   *       so that shared instances will not be disconnected.
   *       However, if buffer is too small, it will be reallocated.
   */
  void convertInto(Array &a, size_t offset = 0) const {
    ArrayBase::convertInto(a, offset);
  }

  /**
   * Create a sparse array from the indexes of non-zero values of the local
   * array and return the new Array.
   */
  Array sparse() const {
    Array a(NTA_BasicType_Sparse);
    ArrayBase::toSparse(a, 0);
    return a;
  }

  /**
   * Copy a subset
   * This creates a new Array of the same type but contains a subset
   * range of values from this Array.
   */
  Array subset(size_t offset, size_t count) const {
    NTA_CHECK(getCount() <= count + offset) << "Requested subset out of range.";
    NTA_CHECK(type_ != NTA_BasicType_SDR)
        << "SDR support for a subset not supported";
    Array a(type_);
    a.allocateBuffer(count);
    memcpy(a.getBuffer(),
           (char *)getBuffer() + offset * BasicType::getSize(type_),
           count * BasicType::getSize(type_));
    return a;
  }

  void invariant() {
    if (!own_)
      NTA_THROW << "Array must own its buffer";
  }

private:
  // Hide base class method (invalid for Array)
  void setBuffer(void *buffer, size_t count) override {}
};
} // namespace nupic

#endif
