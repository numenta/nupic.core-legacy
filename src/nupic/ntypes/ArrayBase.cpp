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

/** @file
 * Implementation of the ArrayBase class
 */

#include <iostream> // for ostream
//#include <iomanip>  // for std::setprecision
#include <cstring>  // for memcpy, memcmp
#include <stdlib.h> // for size_t

#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic {

/**
 * Caller provides a buffer to use.
 * NuPIC always copies data into this buffer
 * Caller frees buffer when no longer needed.
 * ArrayBase() does not own buffer.
 *
 * Warning: Caller must insure that the buffer remains in scope
 *          until this object (and any instances of it) are out
 *          of scope.
 *          It is ok to use this pointer in another std::shared_ptr
 *          because this one will never delete the pointer.
 *
 * Note: for NTA_BasicType_SDR, the count variable is ignored. Its size is
 *       taken from the internal SDR object.
 */
template <typename T>
ArrayBase::ArrayBase(NTA_BasicType type, T *buffer, size_t count) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
  type_ = type;
  setBuffer((char *)buffer, count);
}

/**
 * Caller does not provide a buffer --
 * Nupic will either provide a buffer via setBuffer or
 * ask the ArrayBase to allocate a buffer via allocateBuffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type) {
  if (!BasicType::isValid(type)) {
    NTA_THROW << "Invalid NTA_BasicType " << type
              << " used in array constructor";
  }
  type_ = type;
  own_ = true;
  buffer_ = nullptr;
  count_ = 0;
  capacity_ = 0;
}

/**
 * The destructor will result in the shared_ptr being deleted.
 * If this is the last reference to the pointer, and this class owns the buffer,
 * the pointer will be deleted...making sure it will not leak.
 */
ArrayBase::~ArrayBase() {}

/**
 * Ask ArrayBase to allocate its buffer.  This class owns the buffer.
 * If there was already a buffer allocated, it will be released.
 * The buffer will be deleted when the last copy of this class has been deleted.
 */
void ArrayBase::allocateBuffer(size_t count) {
  // Note that you can allocate a buffer of size zero.
  // The C++ spec (5.3.4/7) requires such a new request to return
  // a non-NULL value which is safe to delete.  This allows us to
  // disambiguate uninitialized ArrayBases and ArrayBases initialized with
  // size zero.
  if (type_ == NTA_BasicType_SDR) {
    std::vector<UInt> dimension;
    dimension[0] = (UInt)count;
    allocateBuffer(dimension);
  } else {
    count_ = count;
    capacity_ = count_ * BasicType::getSize(type_);
    std::shared_ptr<char> sp(new char[capacity_],
                             std::default_delete<char[]>());
    buffer_ = sp;
    own_ = true;
  }
}

void ArrayBase::allocateBuffer(std::vector<UInt> dimensions) { // only for SDR
  NTA_CHECK(type_ == NTA_BasicType_SDR)
      << "Dimensions can only be set on the SDR payload";
  SDR *sdr = new SDR(dimensions);
  std::shared_ptr<char> sp((char *)(sdr));
  buffer_ = sp;
  own_ = true;
  capacity_ = 0; // not used
  count_ = 0;    // not used
}

/**
 * Will fill the buffer with 0's.
 */
void ArrayBase::zeroBuffer() {
  if (buffer_) {
    if (type_ == NTA_BasicType_SDR)
      ((SDR *)(buffer_.get()))->zero();
    else
      std::memset(buffer_.get(), 0, capacity_);
  }
}

/**
 * Internal function
 * Use the given pointer as the buffer.
 * The caller is responsible to delete the buffer.
 * This class will NOT own the buffer so when this class and all copies
 * of this class are deleted the buffer will NOT be deleted.
 * NOTE: A crash condition WILL exists if this class is used
 *       after the object pointed to has gone out of scope. No protections.
 * This allows external buffers to be carried in the Array structure.
 */
void ArrayBase::setBuffer(void *buffer, size_t count) {
  buffer_ = std::shared_ptr<char>((char *)buffer, nonDeleter());
  count_ = count;
  capacity_ = count * BasicType::getSize(type_);
  own_ = false;
}

void ArrayBase::releaseBuffer() {
  buffer_.reset();
  count_ = 0;
  capacity_ = 0;
}

void *ArrayBase::getBuffer() {
  if (type_ == NTA_BasicType_SDR)
    return getSDR()->getDense().data();
  return buffer_.get();
}

SDR *ArrayBase::getSDR() {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  return (SDR *)buffer_.get();
}
const SDR *ArrayBase::getSDR() const {
  NTA_CHECK(type_ == NTA_BasicType_SDR) << "Does not contain an SDR object";
  return (SDR *)buffer_.get();
}

/**
 * Actual size in bytes of the space allocated for the buffer.
 * Not valid for an SDR object.
 */
size_t ArrayBase::getBufferSize() {
  if (type_ == NTA_BasicType_SDR && buffer_ != nullptr) {
    return getSDR()->size;
  }
  return capacity_;
}

/**
 * number of elements of the given type in the buffer.
 */
size_t ArrayBase::getCount() const {
  if (type_ == NTA_BasicType_SDR && buffer_ != nullptr) {
    return ((SDR *)(buffer_.get()))->size;
  }
  return count_;
};

/**
 * max number of elements this buffer can hold.
 * We use this to determine if there is extra space in the buffer
 * to hold the new data so we can avoid having to re-allocate.
 */
size_t ArrayBase::getMaxElementsCount() const {
  if (type_ == NTA_BasicType_SDR)
    return getCount();
  return capacity_ / BasicType::getSize(type_);
};

/**
 * This can be used to truncate an array to a smaller size.
 * Not usable with an SDR.
 */
void ArrayBase::setCount(size_t count) {
  NTA_CHECK(type_ != NTA_BasicType_SDR) << "Operation not valid for SDR";
  NTA_ASSERT(count <= capacity_ / BasicType::getSize(type_))
      << "Cannot set the array count (" << count
      << ") greater than the capacity ("
      << (capacity_ / BasicType::getSize(type_)) << ").";
  count_ = count;
}

/**
 * Return the NTA_BasicType of the current contents.
 */
NTA_BasicType ArrayBase::getType() const { return type_; };

/**
 * Convert the buffer contents of the current ArrayBase into
 * the type of the incoming ArrayBase type. Applying an offset if specified.
 * If there is not enough room in the destination buffer a new one is created.
 * For Fan-In condition, be sure there is enough space in the buffer before
 * the first conversion to avoid loosing data during re-allocation.
 */
void ArrayBase::convertInto(ArrayBase &a, size_t offset) {
  if (offset + count_ > a.getMaxElementsCount()) {
    a.allocateBuffer(offset + count_);
  }
  if (type_ == NTA_BasicType_Sparse)
    fromSparse(a, offset, a.count_);
  else if (a.getType() == NTA_BasicType_Sparse)
    toSparse(a, offset);
  else {
    char *toPtr = (char *)a.getBuffer(); // type as char* so there is an element size
    if (offset)
      toPtr += (offset * BasicType::getSize(a.getType()));
    const void *fromPtr = getBuffer();
    BasicType::convertArray(toPtr, a.type_, fromPtr, type_, count_);
    a.count_ = offset + count_;
  }
}

bool ArrayBase::isInstance(const ArrayBase &a) {
  if (a.buffer_ == nullptr || buffer_ == nullptr)
    return false;
  return (buffer_ == a.buffer_);
}

template <typename T> static void NonZeroT(ArrayBase &a) {
  // populate the new array with indexes of non-zero values.
  a.allocateBuffer(
      count_); // allocating more space than we need, just to be sure.
  T *originalBuffer = (T *)buffer_.get();
  UInt32 j = 0u;
  UInt32 *Destptr = (UInt32 *)a.getBuffer();
  for (UInt32 i = 0u; i < count_; i++) {
    if (originalBuffer[i])
      Destptr[j++] = (UInt32)i;
  }
  a.setCount(j); // set the size.
}

// populate the given array a with the a sparse version of the current array.
void ArrayBase::toSparse(ArrayBase &a, UInt offset) {
  a.type_ = NTA_BasicType_Sparse;
  switch (type_) {
  case NTA_BasicType_Byte:
    NonZeroT<Byte>(a);
    break;
  case NTA_BasicType_Int16:
    NonZeroT<Int16>(a);
    break;
  case NTA_BasicType_UInt16:
    NonZeroT<UInt16>(a);
    break;
  case NTA_BasicType_Int32:
    NonZeroT<Int32>(a);
    break;
  case NTA_BasicType_UInt32:
    NonZeroT<UInt32>(a);
    break;
  case NTA_BasicType_Real32:
    NonZeroT<Real32>(a);
    break;
  case NTA_BasicType_Real64:
    NonZeroT<Real64>(a);
    break;
  case NTA_BasicType_Bool:
    NonZeroT<bool>(a);
    break;
  case NTA_BasicType_SDR:
    SDR_flatSparse_t &v = getSDR()->getFlatSparse();
    a.allocateBuffer(v.size());
    UInt32 *newBuffer = (UInt32 *)a.getBuffer();
    UInt32 *originalBuffer = v.data();
    for (size_t i = 0u; i < v.size(); i++) {
      newBuffer[i++] = originalBuffer[i];
    }
    break;
  case NTA_BasicType_Sparse:;
    break;
    a.allocateBuffer(count_);
    UInt32 *newBuffer = (UInt32 *)a.getBuffer();
    UInt32 *originalBuffer = (UInt32 *)getBuffer();
    for (size_t i = 0u; i < count_; i++) {
      newBuffer[i++] = originalBuffer[i];
    }
    break;
  default:
    NTA_THROW << "Unexpected source array type.";
  }
}

template <typename T>
static void DenseT(ArrayBase &a, UInt32 size, UInt32 *Fromptr, UInt32 count, UInt32 offset) {
  // populate the new array with indexes of non-zero values.
  a.allocateBuffer(size); // allocating more space than we need, just to be sure.
  a.zeroBuffer();
  T *newBuffer = (T *)a.buffer_.get() + offset;
  for (UInt32 i = 0u; i < count; i++) {
    if (Fromptr[i] < size)
      newBuffer[Fromptr[i]] = (T)1;
  }
}

//  populate the given Array a with a flat dense version in a's type.
//  The offset is provided for handling the Fan-in case. It is where this
//  buffer's contribution should be inserted.
// TODO: at the moment we don't have a way to retain full dense buffer size.
//       so for now the caller needs to make some assumptions and pass it in.
void ArrayBase::fromSparse(ArrayBase &a, UInt32 offset, UInt32 size) {
  NTA_CHECK(type_ == NTA_BasicType_Sparse)
      << "This buffer does not contain a sparse type.";
  UInt32 *value = (UInt32 *)getBuffer();
  switch (a.getType()) {
  case NTA_BasicType_Byte:
    DenseT<Byte>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_Int16:
    DenseT<Int16>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_UInt16:
    DenseT<UInt16>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_Int32:
    DenseT<Int32>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_UInt32:
    DenseT<UInt32>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_Real32:
    DenseT<Real32>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_Real64:
    DenseT<Real64>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_Bool:
    DenseT<bool>(a, size, value, count_, offset);
    break;
  case NTA_BasicType_SDR:
    SDR_flatSparse_t& flatSparse = a.getSDR()->getFlatSparse();
    for (size_t i = 0; i < count_; i++)
      flatSparse.push_back(value[i] + offset);
    // Note: before making this conversion, clear the destination SDR
    //       to avoid duplicates during a Fan-in.
    break;
  case NTA_BasicType_Sparse:
    a.allocateBuffer(count_);
    UInt32 *newBuffer = (UInt32 *)a.getBuffer();
    UInt32 *originalBuffer = (UInt32 *)getBuffer();
    for (size_t i = 0u; i < count_; i++) {
      newBuffer[i] = originalBuffer[i]+offset;
    }
    // Note: before making this conversion, clear the destination buffer
    //       to avoid duplicates during a Fan-in.
    break;
  default:
    NTA_THROW << "Unexpected source array type.";
  }
}

///////////////////////////////////////////////////////////////////////////////
//    Compare operators
///////////////////////////////////////////////////////////////////////////////
// Compare contents of two ArrayBase objects
// Note: An Array and an ArrayRef could be the same if type, count, and buffer
// contents are the same.
bool operator==(const ArrayBase &lhs, const ArrayBase &rhs) {
  if (lhs.getType() != rhs.getType() || lhs.getCount() != rhs.getCount())
    return false;
  if (lhs.getCount() == 0)
    return true;
  return (std::memcmp(lhs.getBuffer(), rhs.getBuffer(),
                      lhs.getCount() * BasicType::getSize(lhs.getType())) == 0);
}

////////////////////////////////////////////////////////////////////////////////
//         Stream Serialization (as binary)
////////////////////////////////////////////////////////////////////////////////
void ArrayBase::save(std::ostream &outStream) const {
  outStream << "[ " << count_ << " " << BasicType::getName(type_) << " ";
  if (type_ == NTA_BasicType_SDR) {
    const SDR *sdr = getSDR();
    sdr->save(outStream);
  } else {

    if (count_ > 0) {
      Size size = count_ * BasicType::getSize(type_);
      outStream.write((const char *)buffer_.get(), size);
    }
  }
  outStream << "]" << std::endl;
}
void ArrayBase::load(std::istream &inStream) {
  std::string tag;
  size_t count;

  NTA_CHECK(inStream.get() == '[')
      << "Binary load of Array, expected starting '['.";
  inStream >> count;
  inStream >> tag;
  type_ = BasicType::parse(tag);
  if (type_ == NTA_BasicType_SDR) {
    SDR *sdr = new SDR();
    sdr->load(inStream);
  } else {
    allocateBuffer(count);
    inStream.ignore(1);
    inStream.read(buffer_.get(), capacity_);
  }
  NTA_CHECK(inStream.get() == ']')
      << "Binary load of Array, expected ending ']'.";
  inStream.ignore(1); // skip over the endl
}

////////////////////////////////////////////////////////////////////////////////
//         Stream Serialization  (as Ascii text character strings)
//              [ type count ( item item item ...) ... ]
////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void _templatedStreamBuffer(std::ostream &outStream, const void *inbuf,
                                   size_t numElements) {
  outStream << "( ";

  // Stream the elements
  auto it = (const T *)inbuf;
  auto const end = it + numElements;
  if (it < end) {
    for (; it < end; ++it) {
      outStream << *it << " ";
    }
  }
  outStream << ") ";
}

std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a) {
  auto const inbuf = a.getBuffer();
  auto const numElements = a.getCount();
  auto const elementType = a.getType();

    outStream << "[ " << BasicType::getName(elementType) << " " << numElements << " ";

    switch (elementType)
    {
    case NTA_BasicType_Byte:    _templatedStreamBuffer<Byte>(outStream,   inbuf, numElements);  break;
    case NTA_BasicType_Int16:   _templatedStreamBuffer<Int16>(outStream,  inbuf, numElements);  break;
    case NTA_BasicType_UInt16:  _templatedStreamBuffer<UInt16>(outStream, inbuf, numElements);  break;
    case NTA_BasicType_Int32:   _templatedStreamBuffer<Int32>(outStream,  inbuf, numElements);  break;
    case NTA_BasicType_UInt32:  _templatedStreamBuffer<UInt32>(outStream, inbuf, numElements);  break;
    case NTA_BasicType_Int64:   _templatedStreamBuffer<Int64>(outStream,  inbuf, numElements);  break;
    case NTA_BasicType_UInt64:  _templatedStreamBuffer<UInt64>(outStream, inbuf, numElements);  break;
    case NTA_BasicType_Real32:  _templatedStreamBuffer<Real32>(outStream, inbuf, numElements);  break;
    case NTA_BasicType_Real64:  _templatedStreamBuffer<Real64>(outStream, inbuf, numElements);  break;
    case NTA_BasicType_Bool:    _templatedStreamBuffer<bool>(outStream,   inbuf, numElements);  break;
		//TODO: handle NTA_BasicType_SDR and NTA_BasicType_Sparse
    default:
      NTA_THROW << "Unexpected Element Type: " << elementType;
      break;
    }
    outStream << " ] ";

  return outStream;
}


  template <typename T>
  static void _templatedStreamBuffer(std::istream &inStream, void *buf, size_t numElements) {
    std::string v;
    inStream >> v;
    NTA_CHECK (v == "(") << "deserialize Array buffer...expected an opening '(' but not found.";

    // Stream the elements
    auto it = (T *)buf;
    auto const end = it + numElements;
    if (it < end) {
      for (; it < end; ++it) {
        inStream >> *it;
      }
    }
    inStream >> v;
    NTA_CHECK (v == ")") << "deserialize Array buffer...expected a closing ')' but not found.";
  }



  std::istream &operator>>(std::istream &inStream, ArrayBase &a) {
    std::string v;
    size_t numElements;

    inStream >> v;
    NTA_CHECK(v == "[")  << "deserialize Array object...expected an opening '[' but not found.";

    inStream >> v;
    NTA_BasicType elementType = BasicType::parse(v);
    inStream >> numElements;
    if (a.own_) {
      // An Array, the Array owns its buffer.
      a.type_ = elementType;
      a.allocateBuffer(numElements);
    } else {
      // An ArrayRef, the ArrayRef does not own the buffer
      // but we can overwrite the buffer if there is room.
      size_t neededSize = numElements * BasicType::getSize(elementType);
      NTA_CHECK(a.capacity_ >= neededSize) << "deserialize into an ArrayRef object...Not enough space in buffer.";
      a.count_ = numElements;
      a.type_ = elementType;
    }
    auto inbuf = a.buffer_.get();

    switch (elementType) {
    case NTA_BasicType_Byte:   _templatedStreamBuffer<Byte>(inStream, inbuf,   numElements); break;
    case NTA_BasicType_Int16:  _templatedStreamBuffer<Int16>(inStream, inbuf,  numElements); break;
    case NTA_BasicType_UInt16: _templatedStreamBuffer<UInt16>(inStream, inbuf, numElements); break;
    case NTA_BasicType_Int32:  _templatedStreamBuffer<Int32>(inStream, inbuf,  numElements); break;
    case NTA_BasicType_UInt32: _templatedStreamBuffer<UInt32>(inStream, inbuf, numElements); break;
    case NTA_BasicType_Int64:  _templatedStreamBuffer<Int64>(inStream, inbuf,  numElements); break;
    case NTA_BasicType_UInt64: _templatedStreamBuffer<UInt64>(inStream, inbuf, numElements); break;
    case NTA_BasicType_Real32: _templatedStreamBuffer<Real32>(inStream, inbuf, numElements); break;
    case NTA_BasicType_Real64: _templatedStreamBuffer<Real64>(inStream, inbuf, numElements); break;
    case NTA_BasicType_Bool:   _templatedStreamBuffer<bool>(inStream, inbuf,   numElements); break;
		//TODO: handle NTA_BasicType_SDR and NTA_BasicType_Sparse
    default:  NTA_THROW << "Unexpected Element Type: " << elementType; break;
    }
    inStream >> v;
    NTA_CHECK(v == "]") << "deserialize Array buffer...expected a closing ']' but not found.";
    inStream.ignore(1);

  return inStream;
}

} // namespace nupic
