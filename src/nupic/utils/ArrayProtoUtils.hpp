/*
 * Copyright 2017 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Interface for the Array Capnproto utilities
 */

#ifndef NTA_ARRAY_PROTO_UTILS_HPP
#define NTA_ARRAY_PROTO_UTILS_HPP

#include <nupic/ntypes/Array.hpp>
#include <nupic/proto/LinkProto.capnp.h>
#include <nupic/types/BasicType.hpp>
#include <nupic/types/Types.hpp>
#include <stdlib.h> // for size_t

namespace nupic {

class ArrayProtoUtils {
public:
  /**
   * Serialise NTA Array to ArrayProto
   *
   * @param array         source array
   * @param arrayBuilder  destination capnproto array builder
   */
  static void copyArrayToArrayProto(const Array &array,
                                    ArrayProto::Builder arrayBuilder);

  /**
   * De-serialize ArrayProto into NTA Array
   *
   * @param arrayReader
   *                source capnproto array reader
   * @param array
   *                destination array. NOTE: the array's buffer must be
   *                preallocated and of size that matches the source data.
   * @param allocArrayBuffer
   *          If allocArrayBuffer is false, the Array is assumed to have its
   *          buffer preinitialized with the count of elements of type
   *          DestElementT matching the count in reader; if allocArrayBuffer
   *          is True, the Array's buffer will be released and replaced by a
   *          new buffer of the appropriate size.
   */
  static void copyArrayProtoToArray(const ArrayProto::Reader arrayReader,
                                    Array &array, bool allocArrayBuffer);

  /**
   * Return the NTA_BasicType corresponding to the given ArrayProto reader.
   *
   * @param arrayReader
   *                capnproto array reader
   */
  static NTA_BasicType
  getArrayTypeFromArrayProtoReader(const ArrayProto::Reader arrayReader);

private:
  /**
   * Element type-specific templated function for copying an NTA Array to
   * capnproto ArrayProto builder.
   *
   * @param src
   *          Source Array with elements of type SourceElementT
   * @param builder
   *          Destination type-specific array union element builder of
   *          capnproto ArrayProto.
   */
  template <typename SourceElementT, typename ArrayBuilderT>
  static void _templatedCopyArrayToArrayProto(const Array &src,
                                              ArrayBuilderT builder) {
    NTA_CHECK(BasicType::getSize(src.getType()) == sizeof(SourceElementT));
    NTA_CHECK(builder.size() == src.getCount());

    auto srcData = (SourceElementT *)src.getBuffer();

    for (size_t i = 0; i < src.getCount(); ++i) {
      builder.set(i, srcData[i]);
    }
  }

  /**
   * Element type-specific templated function for copying an NTA Array to
   * capnproto ArrayProto builder.
   *
   * @param reader
   *          Destination type-specific array union element reader of
   *          capnproto ArrayProto.
   * @param dest
   *          Destination Array of type arrayType.
   * @param arrayType
   *          NTA_BasicType of Array elements.
   * @param allocArrayBuffer
   *          If allocArrayBuffer is false, the Array is assumed to have its
   *          buffer preinitialized with the count of elements of type
   *          DestElementT matching the count in reader; if allocArrayBuffer
   *          is True, the Array's buffer will be released and replaced by a
   *          new buffer of the appropriate size.
   */
  template <typename DestElementT, typename ArrayReaderT>
  static void _templatedCopyArrayProtoToArray(ArrayReaderT reader, Array &dest,
                                              NTA_BasicType arrayType,
                                              bool allocArrayBuffer) {
    NTA_CHECK(dest.getType() == arrayType);
    NTA_CHECK(BasicType::getSize(arrayType) == sizeof(DestElementT));

    if (allocArrayBuffer) {
      dest.releaseBuffer();
      dest.allocateBuffer(reader.size());
    }

    NTA_CHECK(reader.size() == dest.getCount());

    auto destData = (DestElementT *)dest.getBuffer();

    for (auto entry : reader) {
      *destData++ = entry;
    }
  }
};

} // namespace nupic

#endif // NTA_ARRAY_PROTO_UTILS_HPP
