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
 * Implementation of the Array Capnproto utilities
 */

#include <nupic/ntypes/Array.hpp>
#include <nupic/proto/LinkProto.capnp.h>
#include <nupic/types/Types.hpp>
#include <nupic/utils/ArrayProtoUtils.hpp>
#include <nupic/utils/Log.hpp>
#include <stdlib.h> // for size_t

using namespace nupic;

void ArrayProtoUtils::copyArrayToArrayProto(const Array &array,
                                            ArrayProto::Builder arrayBuilder) {
  const size_t elementCount = array.getCount();
  const auto arrayType = array.getType();

  switch (arrayType) {
  case NTA_BasicType_Byte:
    _templatedCopyArrayToArrayProto<NTA_Byte>(
        array, arrayBuilder.initByteArray(elementCount));
    break;
  case NTA_BasicType_Int16:
    _templatedCopyArrayToArrayProto<NTA_Int16>(
        array, arrayBuilder.initInt16Array(elementCount));
    break;
  case NTA_BasicType_UInt16:
    _templatedCopyArrayToArrayProto<NTA_UInt16>(
        array, arrayBuilder.initUint16Array(elementCount));
    break;
  case NTA_BasicType_Int32:
    _templatedCopyArrayToArrayProto<NTA_Int32>(
        array, arrayBuilder.initInt32Array(elementCount));
    break;
  case NTA_BasicType_UInt32:
    _templatedCopyArrayToArrayProto<NTA_UInt32>(
        array, arrayBuilder.initUint32Array(elementCount));
    break;
  case NTA_BasicType_Int64:
    _templatedCopyArrayToArrayProto<NTA_Int64>(
        array, arrayBuilder.initInt64Array(elementCount));
    break;
  case NTA_BasicType_UInt64:
    _templatedCopyArrayToArrayProto<NTA_UInt64>(
        array, arrayBuilder.initUint64Array(elementCount));
    break;
  case NTA_BasicType_Real32:
    _templatedCopyArrayToArrayProto<NTA_Real32>(
        array, arrayBuilder.initReal32Array(elementCount));
    break;
  case NTA_BasicType_Real64:
    _templatedCopyArrayToArrayProto<NTA_Real64>(
        array, arrayBuilder.initReal64Array(elementCount));
    break;
  default:
    NTA_THROW << "Unexpected Array Type: " << arrayType;
    break;
  }
}

void ArrayProtoUtils::copyArrayProtoToArray(
    const ArrayProto::Reader arrayReader, Array &array, bool allocArrayBuffer) {
  auto unionSelection = arrayReader.which();

  switch (unionSelection) {
  case ArrayProto::BYTE_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Byte>(arrayReader.getByteArray(), array,
                                              NTA_BasicType_Byte,
                                              allocArrayBuffer);
    break;
  case ArrayProto::INT16_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Int16>(arrayReader.getInt16Array(),
                                               array, NTA_BasicType_Int16,
                                               allocArrayBuffer);
    break;
  case ArrayProto::UINT16_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_UInt16>(arrayReader.getUint16Array(),
                                                array, NTA_BasicType_UInt16,
                                                allocArrayBuffer);
    break;
  case ArrayProto::INT32_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Int32>(arrayReader.getInt32Array(),
                                               array, NTA_BasicType_Int32,
                                               allocArrayBuffer);
    break;
  case ArrayProto::UINT32_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_UInt32>(arrayReader.getUint32Array(),
                                                array, NTA_BasicType_UInt32,
                                                allocArrayBuffer);
    break;
  case ArrayProto::INT64_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Int64>(arrayReader.getInt64Array(),
                                               array, NTA_BasicType_Int64,
                                               allocArrayBuffer);
    break;
  case ArrayProto::UINT64_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_UInt64>(arrayReader.getUint64Array(),
                                                array, NTA_BasicType_UInt64,
                                                allocArrayBuffer);
    break;
  case ArrayProto::REAL32_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Real32>(arrayReader.getReal32Array(),
                                                array, NTA_BasicType_Real32,
                                                allocArrayBuffer);
    break;
  case ArrayProto::REAL64_ARRAY:
    _templatedCopyArrayProtoToArray<NTA_Real64>(arrayReader.getReal64Array(),
                                                array, NTA_BasicType_Real64,
                                                allocArrayBuffer);
    break;
  default:
    NTA_THROW << "Unexpected ArrayProto union member" << (int)unionSelection;
    break;
  }
}

NTA_BasicType ArrayProtoUtils::getArrayTypeFromArrayProtoReader(
    const ArrayProto::Reader arrayReader) {
  auto unionSelection = arrayReader.which();

  switch (unionSelection) {
  case ArrayProto::BYTE_ARRAY:
    return NTA_BasicType_Byte;
    break;
  case ArrayProto::INT16_ARRAY:
    return NTA_BasicType_Int16;
    break;
  case ArrayProto::UINT16_ARRAY:
    return NTA_BasicType_UInt16;
    break;
  case ArrayProto::INT32_ARRAY:
    return NTA_BasicType_Int32;
    break;
  case ArrayProto::UINT32_ARRAY:
    return NTA_BasicType_UInt32;
    break;
  case ArrayProto::INT64_ARRAY:
    return NTA_BasicType_Int64;
    break;
  case ArrayProto::UINT64_ARRAY:
    return NTA_BasicType_UInt64;
    break;
  case ArrayProto::REAL32_ARRAY:
    return NTA_BasicType_Real32;
    break;
  case ArrayProto::REAL64_ARRAY:
    return NTA_BasicType_Real64;
    break;
  default:
    NTA_THROW << "Unexpected ArrayProto union member" << (int)unionSelection;
    break;
  }
}