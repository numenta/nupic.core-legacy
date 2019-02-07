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
 * Definitions for the ArrayBase class
  *
  * An ArrayBase object contains a memory buffer that is used for
  * implementing zero-copy and one-copy operations in NuPIC.
  * An ArrayBase contains:
  * - a pointer to a buffer (held in a shared_ptr)
  * - a length
  * - a capacity   (useful if buffer is larger than data in buffer)
  * - a type
  * - a flag indicating whether or not the object owns the buffer.
  * Note: if buffer is not owned, shared_ptr will not delete it. 
  *       But it also does not provide any protections either.
  */

#ifndef NTA_ARRAY_BASE_HPP
#define NTA_ARRAY_BASE_HPP

#include <iostream> // for ostream, istream
#include <stdlib.h> // for size_t
#include <string>
#include <memory>	// for shared_ptr

#include <nupic/types/Types.hpp>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Serializable.hpp>





namespace nupic
{

  /**
   * An ArrayBase is used for passing arrays of data back and forth between
   * a client application and NuPIC, minimizing copying. It facilitates
   * both zero-copy and one-copy operations.
   */
  class ArrayBase : public Serializable
  {
  public:
    /**
     * Caller provides a buffer to use.
     * NuPIC always copies data into this buffer
     * Caller frees buffer when no longer needed.
     * For NTA_BasicType_SDR, use ArrayBase(SDR&) so dimensions are set.
     */
    ArrayBase(NTA_BasicType type, void *buffer, size_t count);


    /**
     * Create an ArrayBase containing a copy of an SDR.
     */
    ArrayBase(const SDR &sdr);

    /**
     * Caller does not provide a buffer --
     * Nupic will either provide a buffer via setBuffer or
     * ask the ArrayBase to allocate a buffer via allocateBuffer.
     */
    ArrayBase(NTA_BasicType type);


    /**
     * It is ok to use the default copy and assign constructors.
     */
    //ArrayBase(const ArrayBase& other) {
    //  type_ = other.type_;
    //  buffer_ = other.buffer_;
    //  count_ = other.count_;
    //  capacity_ = other.capacity_;
    //  own_ = other.own_;
    //}

    /**
     * The destructor ensures the array doesn't leak its buffer (if
     * it owns it).
     */
    virtual ~ArrayBase();

    /**
     * Ask ArrayBase to allocate its buffer
     * NOTE: for NTA_BasicType_Sparse this sets the size of the dense buffer is describes.
     */
    virtual void allocateBuffer(size_t count);
    virtual void allocateBuffer(const std::vector<UInt> dimensions);  // only for SDR

    /**
     * Ask ArrayBase to zero fill its buffer
     */
    virtual void zeroBuffer();

    /** 
     * resets the shared_ptr. The Array object is now empty.
     */
    virtual void releaseBuffer();

    /**
     * Returns a pointer to the beginning of the buffer.
     * For SDR, this returns a pointer to getDense().data();
     */
    void* getBuffer();
    const void* getBuffer() const;

    /**
     * Returns a pointer to the underlining SDR.
     * If it is not an SDR type, throws exception.
     */
    SDR* getSDR();
    const SDR* getSDR() const;

    /**
     * number of elements of given type in the buffer
     */
    size_t getCount() const;

    /**
     * max number of elements this buffer can hold (capacity)
     */
	  size_t getMaxElementsCount() const;

	  /**
     * Returns the allocated buffer size in bytes independent of array length
     */
    size_t getBufferSize() const;

    /**
     * Returns true if the buffer is allocated a size > 0.  A capacity == 0 could
     * indicate that data is not provided or that a buffer is unused. A zero
     * length buffer is a valid condition.
     */
    bool has_buffer() const;

    /**
     * Put an external buffer into the ArrayBase which is not owned by this class.
     * This allows the Array to transport .py numpy buffers without copy.
     * This allows wrapping an existing SDR without copying it.
     * Caller must ensure that the pointer remains valid over the life of this instance.
     * ArrayBase will NOT free the pointer when this instance goes out of scope.
     */
    virtual void setBuffer(void *buffer, size_t count);
    virtual void setBuffer(SDR &sdr);

    /**
     * Set the number of elements.  This is used to truncate the array.
     * For NTA_BasicTypeSparse, this sets the current length of the sparse array.
     */
    void setCount(size_t count);

    /**
     * Return the type of data contained in the ArrayBase object.
     */
    NTA_BasicType getType() const;

    /**
     * Determines if the argument contains a pointer to the same Shared_ptr.
     */
    bool isInstance(const ArrayBase &a) const;

    /**
     * Call this to refresh the cache in the SDR after making a lot of changes 
     * to the dense buffer.  Call this just before doing anything else with the SDR.
     */
    void inline RefreshCache() {
      if (type_ == NTA_BasicType_SDR) {
        SDR *sdr = getSDR();
        sdr->setDense(sdr->getDense());
      }
    }
    /**
    * serialization and deserialization for an Array and ArrayBase
    */
    // binary representation
    void save(std::ostream &outStream) const override;
    void load(std::istream &inStream) override;

    // ascii text representation
    //    [ type count ( item item item ...) ... ]
    friend std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a);
    friend std::istream &operator>>(std::istream &inStream, ArrayBase &a);

  protected:
    // buffer_ is typed so that we can use new/delete
    // cast to/from void* as necessary
    std::shared_ptr<char> buffer_;
    size_t count_;      // number of elements in the buffer
    size_t capacity_;   // size of the allocated buffer in bytes
    NTA_BasicType type_;// type of data in this buffer
    bool own_;

    // Buffer array conversion routines
    void convertInto(ArrayBase &a, size_t offset=0, size_t maxsize=0) const;


  private:

  };

  // If this class does NOT own the buffer we instantiate the shared_ptr
  // with a version that uses this class as the deleter.  This results
  // in the buffer not being deleted when the last instance of this class
  // is deleted. The Caller is responsible for deleting the buffer.
  struct nonDeleter {
    void operator()(char *p) const {
    }
  };
  ///////////////////////////////////////////////////////////
  // for stream serialization on an Array
  //    [ type count ( item item item ) ]
  // for inStream the Array object must already exist and initialized with a type.
  // The buffer will be allocated and populated with this class as owner.
  std::ostream &operator<<(std::ostream &outStream, const ArrayBase &a);
  std::istream &operator>>(std::istream &inStream, ArrayBase &a);

  // Compare contents of two ArrayBase objects
  // Note: An Array and an ArrayRef could be the same if type, count, and buffer
  // contents are the same.
  bool operator==(const ArrayBase &lhs,const  ArrayBase &rhs);
  inline bool operator!=(const ArrayBase &lhs, const ArrayBase &rhs) {return !(lhs == rhs);}

} // namespace


#endif

