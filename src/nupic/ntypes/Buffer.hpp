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
*/

#ifndef NTA_BUFFER_HPP
#define NTA_BUFFER_HPP

#include <boost/shared_array.hpp>
#include <nupic/ntypes/MemStream.hpp>
#include <nupic/ntypes/ObjectModel.h>
#include <nupic/ntypes/ObjectModel.hpp>
#include <nupic/types/Types.hpp>

#include <string>
#include <vector>

namespace nupic
{
  typedef std::vector<nupic::IReadBuffer *> ReadBufferVec;

  /**
   * ReadBuffer is a class that stores arbitrary binary data in memory. 
   * It has a very simple interface that allows linear writing.
   * You can reset it to the beginning but no random seeking.
   * Very simple. It implements the IReadBuffer interface and
   * NTA_ReadBuffer interface.
   *
   * @b Responsibility:
   * Provide efficient write access of arbitrary binary data
   * from the buffer. The interface is simple enough so it can be 
   * easilly ported to C (so no streams)
   *
   * @b Rationale:
   * Several methods of the plugin API require an arbitrary binary 
   * data store. This is it. The interface is intentionally simple
   * so it can used for the C plugin API.
   *
   * @b Resource/Ownerships:
   *  A vector of bytes that represent the binary data
   *  an IMemeStream to internally stream the data. 
   * 
   * @b Invariants:
   * index_ must be in the range [0, element count). 
   * When the buffer is empty it should be 0.
   * 
   * @b Notes:
   *  see IReadBuffer documentation in nupic/plugin/object_model.hpp for
   * further details
   */
  class ReadBuffer : 
    public IReadBuffer,
    public NTA_ReadBuffer
  {  
  public:
    ReadBuffer(const Byte * value, Size size, bool copy=true);
    ReadBuffer(const ReadBuffer &);
    ReadBuffer & operator=(const ReadBuffer &);
    void assign(const ReadBuffer &);
    void reset() const override;
    Size getSize() const override;
    const Byte * getData() const override;
    
    Int32 read(Byte & value) const override;
    Int32 read(Byte * value, Size & size) const override;
    Int32 read(Int32 & value) const override;
    Int32 read(Int32 * value, Size size) const override;
    Int32 read(UInt32 & value) const override;
    Int32 read(UInt32 * value, Size size) const override;
    Int32 read(Int64 & value) const override;
    Int32 read(Int64 * value, Size size) const override;
    Int32 read(UInt64 & value) const override;
    Int32 read(UInt64 * value, Size size) const override;
    Int32 read(Real32 & value) const override;
    Int32 read(Real32 * value, Size size) const override;
    Int32 read(Real64 & value) const override;
    Int32 read(Real64 * value, Size size) const override;
    Int32 read(bool & value) const override;
    Int32 read(bool * value, Size size) const override;
    Int32 readString(
        NTA_Byte * &value, 
        NTA_UInt32 &size,
        NTA_Byte *(*fAlloc)(NTA_UInt32 size)=nullptr,
        void (*fDealloc)(NTA_Byte *)=nullptr
      ) const override;
    
    template <typename T>
    Int32 readT(T & value)  const
    {
      ReadBuffer * r = const_cast<ReadBuffer *>(this);
      if (memStream_.eof())
        return 1;
        
      try
      {
        r->memStream_ >> value;
        return 0;
      }
      catch (...)
      {
        if (memStream_.eof())
          return 1;
        else
          return -1;
      }
    }
    
    template <typename T>
    Int32 readT(T * value, Size size)  const
    {
      ReadBuffer * r = const_cast<ReadBuffer *>(this);
      try
      {
        for (Size i = 0; i < size; ++i)
          r->read(value[i]);
        return 0;
      }
      catch (...)
      {
        return -1;
      }
    }
  private:
    boost::shared_array<Byte> bytes_;
    mutable IMemStream memStream_;
  };

  class ReadBufferIterator :
    public IReadBufferIterator,
    public NTA_ReadBufferIterator
  {  
  public:
    ReadBufferIterator(ReadBufferVec & rbv);
    const IReadBuffer * next() override;
    void reset() override;
  private:
    ReadBufferVec & readBufferVec_;
    Size index_;
  };

  /**
   * WriteBuffer is a class that stores arbitrary binary data in memory. 
   * It has a very simple interface that allows linear writing.
   * You can get the entire buffer using getData().
   * Very simple. It implements the IWriteBuffer interface and
   * NTA_WriteBuffer interface.
   *
   * @b Responsibility:
   * Provide efficient write access of arbitrary binary data
   * to the buffer. The interface is simple enough so it can be 
   * easilly ported to C (so no streams)
   *
   * @b Rationale:
   * Several methods of the plugin API require an arbitrary binary 
   * data store. This is it. The interface is intentionally simple
   * so it can used for the C plugin API.
   *
   * @b Resource/Ownerships:
   *  The OMemeStream private base class manages the actual data
   * 
   * @b Invariants:
   * index_ must be in the range [0, element count). 
   * When the buffer is empty it should be 0.
   * 
   * @b Notes:
   *  see IWriteBuffer documentation in nupic/plugin/object_model.hpp for
   * further details
   */
  class WriteBuffer : 
    public IWriteBuffer,
    public NTA_WriteBuffer,
    private OMemStream
  {  
  public:
    WriteBuffer();
    virtual ~WriteBuffer();
    Int32 write(Byte value) override;
    Int32 write(const Byte * value, Size size) override;
    Int32 write(Int32 value) override;
    Int32 write(const Int32 * value, Size size) override;
    Int32 write(UInt32 value) override;
    Int32 write(const UInt32 * value, Size size) override;
    Int32 write(Int64 value) override;
    Int32 write(const Int64 * value, Size size) override;
    Int32 write(UInt64 value) override;
    Int32 write(const UInt64 * value, Size size) override;
    Int32 write(Real32 value) override;  
    Int32 write(const Real32 * value, Size size) override;
    Int32 write(Real64 value) override;  
    Int32 write(const Real64 * value, Size size) override;
    Int32 write(bool value) override;
    Int32 write(const bool * value, Size size) override;
    Int32 writeString(const Byte * value, Size size) override;
    
    Size getSize() override;
    const Byte * getData() override;
  
    template <typename T>
    Int32 writeT(T value, const char *sep=" ")
    {
      try
      {
        if (sep && (getSize() > 0))
          *this << ' ';
        *this << value;
        return 0;
      }
      catch (...)
      {
        return -1;
      }
    }
  
    template <typename T>
    Int32 writeT(const T * value, Size size)
    {
      try
      {
        for (Size i = 0; i < size; ++i)
        {
          const T & val = value[i];
          write(val);
        }
        return 0;
      }
      catch (...)
      {
        return -1;
      }
    }
  };
}

#endif // NTA_BUFFER_HPP
