/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
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

/** @file
 * Array utilities
 */

#ifndef NTA_ARRAY_UTILS_HPP
#define NTA_ARRAY_UTILS_HPP

#include <string>
#include <stdlib.h> // for size_t

#include <nupic/ntypes/Array.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp>


namespace nupic
{

  class ArrayUtils
  {
  public:
    /**
     * Dump an NTA Array to string for debugging.
     *
     * NOTE: presently, only numeric element types are supported.
     *
     * @param array         source array
     *
     * @returns Comma-separated values
     */
    static std::string arrayToString(const Array& array)
    {
      return bufferOfBasicTypeToString(array.getBuffer(),
                                       array.getCount(),
                                       array.getType());
    }


    /**
     * Dump a buffer of NTA_BasicType to string for debugging.
     *
     * NOTE: presently, only numeric element types are supported.
     *
     * @param inbuf       input buffer
     * @param numElements number of elements to use from the beginning of buffer
     * @param elementType type of the elements
     *
     * @returns Comma-separated values
     */
    static std::string bufferOfBasicTypeToString(const void* inbuf,
                                                 size_t numElements,
                                                 NTA_BasicType elementType)
    {
      switch (elementType)
      {
      case NTA_BasicType_Byte:
        return _templatedDumpBufferToString<NTA_Byte>(inbuf, numElements);
        break;
      case NTA_BasicType_Int16:
        return _templatedDumpBufferToString<NTA_Int16>(inbuf, numElements);
        break;
      case NTA_BasicType_UInt16:
        return _templatedDumpBufferToString<NTA_UInt16>(inbuf, numElements);
        break;
      case NTA_BasicType_Int32:
        return _templatedDumpBufferToString<NTA_Int32>(inbuf, numElements);
        break;
      case NTA_BasicType_UInt32:
        return _templatedDumpBufferToString<NTA_UInt32>(inbuf, numElements);
        break;
      case NTA_BasicType_Int64:
        return _templatedDumpBufferToString<NTA_Int64>(inbuf, numElements);
        break;
      case NTA_BasicType_UInt64:
        return _templatedDumpBufferToString<NTA_UInt64>(inbuf, numElements);
        break;
      case NTA_BasicType_Real32:
        return _templatedDumpBufferToString<NTA_Real32>(inbuf, numElements);
        break;
      case NTA_BasicType_Real64:
        return _templatedDumpBufferToString<NTA_Real64>(inbuf, numElements);
        break;
      default:
        NTA_THROW << "Unexpected Element Type: " << elementType;
        break;
      }
    }

  private:
    /**
     * Element type-specific templated function dumping elements to string.
     *
     * @param inbuf       input buffer
     * @param numElements number of elements to use from the beginning of buffer
     *
     * @returns std::string of comma-separated values
     */
    template <typename SourceElementT>
    static std::string _templatedDumpBufferToString(const void* inbuf,
                                                    size_t numElements)
    {
      auto srcData = (SourceElementT*)inbuf;
      std::stringstream ss;

      for (size_t i=0; i < numElements; ++i)
      {
        ss << *srcData++;
        if (i < (numElements - 1))
        {
          ss << ", ";
        }
      }

      return ss.str();
    }
  };




} // namespace nupic


#endif // NTA_ARRAY_UTILS_HPP
