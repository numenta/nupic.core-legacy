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
 * Interface for the Array Capnproto utilities
 */

#ifndef NTA_ARRAY_PROTO_UTILS_HPP
#define NTA_ARRAY_PROTO_UTILS_HPP

#include <nupic/ntypes/Array.hpp>
#include <nupic/proto/LinkProto.capnp.h>
#include <nupic/types/Types.hpp>
#include <stdlib.h> // for size_t


namespace nupic
{

  class ArrayProtoUtils
  {
  public:
    /**
     * Serialise NTA Array to ArrayProto
     *
     * @param array         source array
     * @param arrayBuilder  destination capnproto array builder
     */
    static void copyArrayToArrayProto(const Array& array,
                                      ArrayProto::Builder arrayBuilder);

    /**
     * De-serialize ArrayProto into NTA Array
     *
     * @param arrayReader
     *                source capnproto array reader
     * @param array
     *                destination array. NOTE: the array's buffer must be
     *                preallocated and of size that matches the source data.
     *
     */
    static void copyArrayProtoToArray(const ArrayProto::Reader arrayReader,
                                      Array& array);

  private:
    template <typename SourceT, typename ArrayBuilderT>
    static void _templatedCopyArrayToArrayProto(SourceT src,
                                                ArrayBuilderT builder,
                                                size_t elementCount)
    {
      for (size_t i=0; i < elementCount; ++i)
      {
        builder.set(i, src[i]);
      }
    }

    template <typename DestDataT, typename ArrayReaderT>
    static void _templatedCopyArrayProtoToArray(ArrayReaderT reader,
                                                Array & dest,
                                                NTA_BasicType arrayType)
    {
      NTA_CHECK(reader.size() == dest.getCount());
      NTA_CHECK(dest.getType() == arrayType);

      auto destData = (DestDataT*)dest.getBuffer();

      for (auto entry: reader)
      {
        *destData++ = entry;
      }
    }
  };

} // namespace nupic


#endif // NTA_ARRAY_PROTO_UTILS_HPP
