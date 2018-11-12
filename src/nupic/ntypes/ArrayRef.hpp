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
// Definitions for the ArrayRef class
//
// The ArrayRef is a sub-class of ArrayBase that doesn't own its buffer
// This is a immutable version of an Array object.
// The buffer cannot be modified because it is defined as const.
// It does not own its buffer so if the ArrayRef is deleted the buffer is not
// deleted. Method 3 is a special case where it will delete the buffer if
// no other references exists.
//
// It can be created and populated by any of the following methods:
// Method 1:  Constructor
//            ArrayRef(type, buffer, count);
//    The buffer argument is any existing array.
//    If the caller deletes the buffer, the ArrayRef becomes invalid.
//
// Method 2: Constructor and SetBuffer()
//            ArrayRef(type);               // creates an empty buffer
//            setBuffer(buffer, count);     // populates the buffer by assigning
//            the pointer.
//    The buffer argument is any existing array.
//    If the caller deletes the buffer, the ArrayRef becomes invalid.
//
// Method 3: Ref() function
//           ArrayRef B = A.ref();          // The ref() function on the Array
//           object
//    The buffer will NOT be deleted when the Array object is deleted.
//    However, this uses a smart pointer so it remains valid until the ArrayRef
//    object is deleted.  If there are no other references remaining, the buffer
//    will be deleted.
//
// ASSIGNMENT
// If an ArrayRef is assigned to another ArrayRef another instance is created
// and both reference the same buffer.
//     ArrayRef B = A;
// ---

#ifndef NTA_ARRAY_REF_HPP
#define NTA_ARRAY_REF_HPP

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {
class ArrayRef : public ArrayBase {
public:
  ArrayRef() : ArrayBase(NTA_BasicType_Int32) {}
  ArrayRef(NTA_BasicType type, void *buffer, size_t count) : ArrayBase(type) {
    setBuffer(buffer, count);
  }

  explicit ArrayRef(NTA_BasicType type) : ArrayBase(type) {}

  ArrayRef(const ArrayRef &other) : ArrayBase(other) {}

  const void *getBuffer() const { return buffer_.get(); }

  void invariant() {
    if (own_)
      NTA_THROW << "ArrayRef must not own its buffer";
  }

private:
  // Hide some base class methods (invalid for ArrayRef)
  void allocateBuffer(size_t count) override {}
  void zeroBuffer() override {}
  ArrayRef(NTA_BasicType type, std::shared_ptr<char> sharedBuffer, size_t count)
      : ArrayBase(type) {
    buffer_ = sharedBuffer;
    count_ = count;
    capacity_ = count;
    own_ = false;
  }
  friend class Array;
};
} // namespace nupic

#endif
