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
#include <stdlib.h> // for size_t
#include <cstring>   // for memcpy, memcmp

#include <nupic/types/Types.hpp>
#include <nupic/types/BasicType.hpp>
#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/utils/Log.hpp>

namespace nupic
{



/**
 * Caller provides a buffer to use.
 * NuPIC always copies data into this buffer
 * Caller frees buffer when no longer needed.
 * ArrayBase() does not own buffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type, void* buffer, size_t count)
{
  if(!BasicType::isValid(type))
  {
    NTA_THROW << "Invalid NTA_BasicType " << type << " used in array constructor";
  }
  type_ = type;
  setBuffer(buffer, count);
}

/**
 * Caller does not provide a buffer --
 * Nupic will either provide a buffer via setBuffer or
 * ask the ArrayBase to allocate a buffer via allocateBuffer.
 */
ArrayBase::ArrayBase(NTA_BasicType type) 
{
  if(!BasicType::isValid(type))
  {
    NTA_THROW << "Invalid NTA_BasicType " << type << " used in array constructor";
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
ArrayBase::~ArrayBase()
{
}

/**
 * Ask ArrayBase to allocate its buffer.  This class owns the buffer.
 * If there was already a buffer allocated, it will be released.
 * The buffer will be deleted when the last copy of this class has been deleted.
 */
void
ArrayBase::allocateBuffer(size_t count)
{
  //Note that you can allocate a buffer of size zero.
  //The C++ spec (5.3.4/7) requires such a new request to return
  //a non-NULL value which is safe to delete.  This allows us to
  //disambiguate uninitialized ArrayBases and ArrayBases initialized with
  //size zero.
  count_ = count;
  capacity_ = count_ * BasicType::getSize(type_);
  std::shared_ptr<char> sp(new char[capacity_], std::default_delete<char[]>());
  buffer_ = sp;
  own_ = true;
}


/**
 * Will fill the buffer with 0's.
 */
void
ArrayBase::zeroBuffer()
{
  std::memset(buffer_.get(), 0, capacity_);
}

/**
 * Use the given pointer as the buffer.  
 * The caller is responsible to delete the buffer.
 * This class will NOT own the buffer so when this class and all copies
 * of this class are deleted the buffer will NOT be deleted.
 * NOTE: A crash condition WILL exists if this class is used
 *       after the caller has deleted the pointer. No protections.
 */
void
ArrayBase::setBuffer(void *buffer, size_t count)
{
  buffer_ = std::shared_ptr<char>((char*)buffer, nonDeleter());
  count_ = count;
  capacity_ = count * BasicType::getSize(type_);
  own_ = false;
}

void
ArrayBase::releaseBuffer()
{
  buffer_.reset();
  count_ = 0;
  capacity_ = 0;
}

void*
ArrayBase::getBuffer() const
{
  return buffer_.get();
}

size_t ArrayBase::getBufferSize() const { return capacity_; }


// number of elements of given type in the buffer
size_t
ArrayBase::getCount() const
{
  return count_;
};

// max number of elements this buffer can hold
size_t ArrayBase::getMaxElementsCount() const {
  return capacity_ / BasicType::getSize(type_);
};

void ArrayBase::setCount(size_t count)
{ 
  NTA_ASSERT(count <= capacity_/BasicType::getSize(type_))  
		<< "Cannot set the array count (" << count << ") greater than the capacity ("
		<< (capacity_/BasicType::getSize(type_)) << ").";
  count_ = count; 
}


NTA_BasicType
ArrayBase::getType() const
{
  return type_;
};


void 
ArrayBase::convertInto(ArrayBase &a, size_t offset) const { 
  if (offset + count_ > a.getMaxElementsCount()) {
    a.allocateBuffer(offset + count_);
  }
  char *toPtr = (char *)a.getBuffer();  // type as char* so there is an element size
  if (offset)
    toPtr += (offset * BasicType::getSize(a.getType()));
  const void *fromPtr = getBuffer();
  BasicType::convertArray(toPtr, a.type_, fromPtr, type_, count_);
  a.count_ = offset + count_;
}


bool ArrayBase::isInstance(const ArrayBase &a) { 
  if (a.buffer_ == nullptr || buffer_ == nullptr)  return false;
  return (buffer_ == a.buffer_);
}

// populate the given array with the NZ of the current array.
void ArrayBase::NonZero(ArrayBase& a) const {
      switch(type_) 
      {
      case NTA_BasicType_Byte:   ArrayBase::NonZeroT<Byte>(a);   break;
      case NTA_BasicType_Int16:  ArrayBase::NonZeroT<Int16>(a);  break;
      case NTA_BasicType_UInt16: ArrayBase::NonZeroT<UInt16>(a); break;
      case NTA_BasicType_Int32:  ArrayBase::NonZeroT<Int32>(a);  break;
      case NTA_BasicType_UInt32: ArrayBase::NonZeroT<UInt32>(a); break;
      case NTA_BasicType_Real32: ArrayBase::NonZeroT<Real32>(a); break;
      case NTA_BasicType_Real64: ArrayBase::NonZeroT<Real64>(a); break;
      default:
        NTA_THROW << "Unexpected source array type.";
      }
}


template <typename T>
void ArrayBase::NonZeroT(ArrayBase& a) const
{
  NTA_ASSERT(a.getType() == NTA_BasicType_UInt32) 
    << "Expected UInt32 type for NonZero() destination array";
  T *originalBuffer = (T *)buffer_.get();
  // find the number of elements for the NZ array
  size_t nonZeroLen = 0;
  for (size_t i = 0; i < count_; i++) {
    if (originalBuffer[i])
      nonZeroLen++;
  }
  // populate the new array with indexes of non-zero values.
  a.allocateBuffer(nonZeroLen);
  UInt32 *ptr = (UInt32 *)a.getBuffer();
  for (size_t i = 0; i < count_; i++) {
    if (originalBuffer[i])
      *ptr++ = (UInt32)i;
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
//         Binary Serialization
////////////////////////////////////////////////////////////////////////////////
void ArrayBase::binarySave(std::ostream &outStream) const
{
    outStream << "[ " << count_ << " " << BasicType::getName(type_) << " ";
    if (count_ > 0) {
      Size size = count_ * BasicType::getSize(type_);
      outStream.write((const char*)buffer_.get(), size);
    }
    outStream << "]" << std::endl;

}
void ArrayBase::binaryLoad(std::istream &inStream) { 
  std::string tag;
  size_t count;

  NTA_CHECK(inStream.get() == '[') << "Binary load of Array, expected starting '['.";
  inStream >> count;
  inStream >> tag;
  type_ = BasicType::parse(tag);
  allocateBuffer(count);
  inStream.ignore(1);
  inStream.read(buffer_.get(), capacity_);
  NTA_CHECK(inStream.get() == ']') << "Binary load of Array, expected ending ']'.";
  inStream.ignore(1); // skip over the endl
}


////////////////////////////////////////////////////////////////////////////////
//         Stream Serialization
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

  std::ostream& operator<<(std::ostream& outStream, const ArrayBase& a)
  {
    auto const inbuf = a.getBuffer();
    auto const numElements = a.getCount();
    auto const elementType = a.getType();

    outStream << " [ " << BasicType::getName(elementType) << " " << numElements << " ";

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
    default:
      NTA_THROW << "Unexpected Element Type: " << elementType;
      break;
    }
    outStream << " ]";

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
    default:  NTA_THROW << "Unexpected Element Type: " << elementType; break;
    }
    inStream >> v;
    NTA_CHECK(v == "]") << "deserialize Array buffer...expected a closing ']' but not found.";

    return inStream;
  }

  ////////////////////////////////////////////////////////////////////////////////
  //         YAML Serialization
  ////////////////////////////////////////////////////////////////////////////////

  void ArrayBase::serialize(YAML::Emitter& out) const
  {
    out << YAML::BeginMap;
    out << YAML::Key << "type" << YAML::Value  << BasicType::getName(type_);
    out << YAML::Key << "count" << YAML::Value << count_;
    out << YAML::Key << "buffer" << YAML::Value << YAML::BeginSeq;
    switch (type_) {
    case NTA_BasicType_Byte:
      for (size_t i = 0; i < count_; i++) {
        out << ((unsigned char*)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Int16:
      for (size_t i = 0; i < count_; i++) {
        out << ((Int16 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_UInt16:
      for (size_t i = 0; i < count_; i++) {
        out << ((UInt16 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Int32:
      for (size_t i = 0; i < count_; i++) {
        out << ((Int32 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_UInt32:
      for (size_t i = 0; i < count_; i++) {
        out << ((Int16 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Int64:
      for (size_t i = 0; i < count_; i++) {
        out << ((Int64 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_UInt64:
      for (size_t i = 0; i < count_; i++) {
        out << ((UInt64 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Real32:
      for (size_t i = 0; i < count_; i++) {
        out << ((Real32 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Real64:
      for (size_t i = 0; i < count_; i++) {
        out << ((Real64 *)buffer_.get())[i];
      }
      break;
    case NTA_BasicType_Bool:
      for (size_t i = 0; i < count_; i++) {
        out << ((bool *)buffer_.get())[i];
      }
      break;
    default:
      NTA_THROW << "Serializing, Unexpected Data Type in Array: " << type_;
      break;
    }
    out << YAML::EndSeq;
    out << YAML::EndMap;
  }


  void ArrayBase::deserialize(const YAML::Node &doc)
  {
    NTA_CHECK(doc.Type() == YAML::NodeType::Map)
        << "Invalid deserializing of Array -- expecting a map";
    NTA_CHECK(doc.size() == 3) << "Invalid deserializing of Array -- contains "
                               << doc.size() << " elements, expected 3.";
    YAML::Node node;

    // 1. type
    node = doc["type"];
    NTA_CHECK(node.IsScalar()) << "Invalid deserializing of Array-- does not have a 'type' field.";
    std::string linkType = node.as<std::string>();
    type_ = BasicType::parse(linkType);

    // 2. count
    node = doc["count"];
    NTA_CHECK(node.IsScalar()) << "Invalid deserializing of Array-- does not have a 'count' field.";
    size_t count = node.as<size_t>();
    allocateBuffer(count);

    // 3. buffer
    node = doc["buffer"];
    NTA_CHECK(node.IsSequence())
        << "Invalid deserializing of Array-- does not have a 'buffer' field.";
    size_t i = 0;
    for (const auto &dataValiter : node) {
      NTA_CHECK(dataValiter.IsScalar()) << "Invalid deserializing of Array-- missing element " << i << ".";
      NTA_CHECK(i < count_)  << "Invalid deserializing of Array-- has too many data elements.";
      char *ptr = buffer_.get();
      switch (type_) {
      case NTA_BasicType_Byte:
        ((unsigned char*)ptr)[i++] = dataValiter.as<unsigned char>();
        break;
      case NTA_BasicType_Int16:
        ((Int16 *)ptr)[i++] = dataValiter.as<Int16>();
        break;
      case NTA_BasicType_Int32:
        ((Int32 *)ptr)[i++] = dataValiter.as<Int32>();
        break;
      case NTA_BasicType_Int64:
        ((Int64 *)ptr)[i++] = dataValiter.as<Int64>();
        break;
      case NTA_BasicType_UInt32:
        ((UInt32 *)ptr)[i++] = dataValiter.as<UInt32>();
        break;
      case NTA_BasicType_UInt64:
        ((UInt64 *)ptr)[i++] = dataValiter.as<UInt64>();
        break;
      case NTA_BasicType_Real32:
        ((Real32 *)ptr)[i++] = dataValiter.as<Real32>();
        break;
      case NTA_BasicType_Real64:
        ((Real64 *)ptr)[i++] = dataValiter.as<Real64>();
        break;
      case NTA_BasicType_Bool:
        ((bool *)ptr)[i++] = dataValiter.as<bool>();
        break;
      default:
        NTA_THROW << "Unexpected data type in Array deserialization.";
      } // switch
    }   // for
  }


} // namespace

