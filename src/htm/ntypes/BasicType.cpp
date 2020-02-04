/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

#include <limits>
#include <cerrno>
#include <cstring> // std::strerror(errno)

#include <htm/ntypes/BasicType.hpp>

#include <htm/types/Exception.hpp>
#include <htm/utils/Log.hpp>
#include <htm/types/Sdr.hpp>


using namespace htm;

bool BasicType::isValid(NTA_BasicType t) {
  return (t >= 0) && (t < NTA_BasicType_Last);
}

const char *BasicType::getName(NTA_BasicType t) {
  static const char *names[] = {
      "Byte",   "Int16",  "UInt16", "Int32",  "UInt32", "Int64",
      "UInt64", "Real32", "Real64", "Handle", "Bool", "SDR", "String"
  };

  if (!isValid(t))
    throw Exception(__FILE__, __LINE__,
                    "BasicType::getName -- Basic type is not valid");

  return names[t];
}

// gcc 4.2 requires (incorrectly) these to be defined inside a namespace
namespace htm {
// getName<T>
template <> const char *BasicType::getName<Byte>() {
  return getName(NTA_BasicType_Byte);
}

template <> const char *BasicType::getName<Int16>() {
  return getName(NTA_BasicType_Int16);
}

template <> const char *BasicType::getName<UInt16>() {
  return getName(NTA_BasicType_UInt16);
}

template <> const char *BasicType::getName<Int32>() {
  return getName(NTA_BasicType_Int32);
}

template <> const char *BasicType::getName<UInt32>() {
  return getName(NTA_BasicType_UInt32);
}

template <> const char *BasicType::getName<Int64>() {
  return getName(NTA_BasicType_Int64);
}

template <> const char *BasicType::getName<UInt64>() {
  return getName(NTA_BasicType_UInt64);
}

template <> const char *BasicType::getName<Real32>() {
  return getName(NTA_BasicType_Real32);
}

template <> const char *BasicType::getName<Real64>() {
  return getName(NTA_BasicType_Real64);
}

template <> const char *BasicType::getName<Handle>() {
  return getName(NTA_BasicType_Handle);
}

template <> const char *BasicType::getName<bool>() {
  return getName(NTA_BasicType_Bool);
}
template <> const char *BasicType::getName<SDR>() {
  return getName(NTA_BasicType_SDR);
}

// getType<T>()
template <> NTA_BasicType BasicType::getType<Byte>() {
  return NTA_BasicType_Byte;
}

template <> NTA_BasicType BasicType::getType<Int16>() {
  return NTA_BasicType_Int16;
}

template <> NTA_BasicType BasicType::getType<UInt16>() {
  return NTA_BasicType_UInt16;
}

template <> NTA_BasicType BasicType::getType<Int32>() {
  return NTA_BasicType_Int32;
}

template <> NTA_BasicType BasicType::getType<UInt32>() {
  return NTA_BasicType_UInt32;
}

template <> NTA_BasicType BasicType::getType<Int64>() {
  return NTA_BasicType_Int64;
}

template <> NTA_BasicType BasicType::getType<UInt64>() {
  return NTA_BasicType_UInt64;
}

template <> NTA_BasicType BasicType::getType<Real32>() {
  return NTA_BasicType_Real32;
}

template <> NTA_BasicType BasicType::getType<Real64>() {
  return NTA_BasicType_Real64;
}

template <> NTA_BasicType BasicType::getType<Handle>() {
  return NTA_BasicType_Handle;
}

template <> NTA_BasicType BasicType::getType<bool>() {
  return NTA_BasicType_Bool;
}
template <> NTA_BasicType BasicType::getType<SDR>() {
  return NTA_BasicType_SDR;
}
template <> NTA_BasicType BasicType::getType<std::string>() { return NTA_BasicType_Str; }
} // namespace htm

// Return the size in bytes of a basic type (element of the array)
size_t BasicType::getSize(NTA_BasicType t) {
  static size_t basicTypeSizes[] = {
      sizeof(Byte),   sizeof(Int16),  sizeof(UInt16), sizeof(Int32),
      sizeof(UInt32), sizeof(Int64),  sizeof(UInt64), sizeof(Real32), 
      sizeof(Real64), sizeof(Handle), sizeof(bool),  sizeof(char), sizeof(std::string)
  };

  if (!isValid(t))
    throw Exception(__FILE__, __LINE__,
                    "BasicType::getSize -- basic type is not valid");
  return basicTypeSizes[t];
}

NTA_BasicType BasicType::parse(const std::string &s) {
  if (s == std::string("Byte") || s == std::string("str"))
    return NTA_BasicType_Byte;
  else if (s == std::string("Int16"))
    return NTA_BasicType_Int16;
  else if (s == std::string("UInt16"))
    return NTA_BasicType_UInt16;
  else if (s == std::string("Int32") || s == std::string("int"))
    return NTA_BasicType_Int32;
  else if (s == std::string("UInt32") || s == std::string("uint"))
    return NTA_BasicType_UInt32;
  else if (s == std::string("Int64"))
    return NTA_BasicType_Int64;
  else if (s == std::string("UInt64"))
    return NTA_BasicType_UInt64;
  else if (s == std::string("Real32") || s == std::string("float"))
    return NTA_BasicType_Real32;
  else if (s == std::string("Real64"))
    return NTA_BasicType_Real64;
  else if (s == std::string("Real"))
    return NTA_BasicType_Real;
  else if (s == std::string("Handle"))
    return NTA_BasicType_Handle;
  else if (s == std::string("Bool") || s == std::string("bool"))
    return NTA_BasicType_Bool;
  else if (s == std::string("SDR"))
    return NTA_BasicType_SDR;
  else if (s == std::string("String") || s == std::string("std::string"))
    return NTA_BasicType_Str;
  else if (s == std::string("Last"))
    return NTA_BasicType_Last;  // Means none-of-the-above.
  else
    throw Exception(__FILE__, __LINE__,
                    std::string("Invalid basic type name: ") + s);
}

/**
* target is bool (0 or anything else)
* target is same type as source.
* target is larger type then source and same sign.
* No range checks needed.
*/
template <typename T, typename F>
static void cpyarray(void *toPtr, const void *fromPtr, size_t count) {
  T *ptr1 = static_cast<T *>(toPtr);
  const F *ptr2 = reinterpret_cast<const F *>(fromPtr);
  for (size_t i = 0; i < count; i++) {
    *ptr1++ = static_cast<T>(*ptr2++);
  }
}

/**
 * source type larger than source or sign different.
 * Range checks needed
 */
template <typename T, typename F>
static void cpyarray(void *toPtr, const void *fromPtr, size_t count, F minVal, F maxVal) {
  T *ptr1 = static_cast<T *>(toPtr);
  const F *ptr2 = reinterpret_cast<const F *>(fromPtr);
  for (size_t i = 0; i < count; i++) {
    NTA_CHECK(*ptr2 >= minVal && *ptr2 <= maxVal)
          << "Value Out of range. Value: " << *ptr2 << " ";
    *ptr1++ = static_cast<T>(*ptr2++);
  }
}

template <typename T>
static void cpyIntoSDR(Byte *toPtr, const T *fromPtr, size_t count) {
  const T zero = static_cast<T>(0);
  for(size_t i = 0u; i < count; i++)
    toPtr[i] = fromPtr[i] != zero; // 1 or 0
}
template <typename T> 
static void cpyIntoStr(std::string *toPtr, const T *fromPtr, size_t count) {
  for (size_t i = 0u; i < count; i++)
    toPtr[i] = std::to_string(fromPtr[i]);
}


// Create a set of static copyFromStr() functions to convert from an array of strings to 
// an array of type T.
#define NTA_CONVERT_FROM_STR(T, I)                                                                        \
static void cpyFromStr(T *toPtr, const std::string *fromPtr, size_t count) {                              \
  errno = 0;                                                                                              \
  char *end;                                                                                              \
  for (size_t i = 0u; i < count; i++) {                                                                   \
    toPtr[i] = (T)I;                                                                                        \
    if (errno)  NTA_THROW << "In '" << fromPtr[i] << "' numeric conversion error: " << std::strerror(errno);\
    if (*end != '\0') NTA_THROW << "In '" << fromPtr[i] << "' numeric conversion error: invalid char.";     \
  }                                                                                                       \
}
NTA_CONVERT_FROM_STR(Byte, std::strtol(fromPtr[i].c_str(), &end, 0) );
NTA_CONVERT_FROM_STR(int16_t, std::strtol(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(uint16_t, std::strtoul(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(int32_t, std::strtol(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(uint32_t, std::strtoul(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(int64_t, std::strtoll(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(uint64_t, std::strtoull(fromPtr[i].c_str(), &end, 0));
NTA_CONVERT_FROM_STR(float, std::strtof(fromPtr[i].c_str(), &end));
NTA_CONVERT_FROM_STR(double, std::strtod(fromPtr[i].c_str(), &end));
static void cpyFromStr(bool *toPtr, const std::string *fromPtr, size_t count) {
  for (size_t i = 0u; i < count; i++) {
    std::string val = fromPtr[i];
    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
    if (val == "true" || val == "on" || val == "1")
      toPtr[i] = true;
    else if (val == "false" || val == "off" || val == "0")
      toPtr[i] = false;
    else NTA_THROW << "Invalid value for a boolean. " << val;
  }
}




void BasicType::convertArray(void *ptr1, NTA_BasicType toType, const void *ptr2,
                             NTA_BasicType fromType, size_t count) {
  if (ptr2 == nullptr || count == 0)
    return;
  NTA_CHECK(ptr1 != nullptr);
  try {
    switch (fromType) {
    case NTA_BasicType_Byte: // char.  This might be signed or unsigned.
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Byte>(ptr1, ptr2, count,  0, std::numeric_limits<Byte>::max());
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Byte>(ptr1, ptr2, count, 0, std::numeric_limits<Byte>::max());
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Byte>(ptr1, ptr2, count, 0, std::numeric_limits<Byte>::max());
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR((Byte*)ptr1, (Byte*)ptr2, count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr((std::string*)ptr1, (Byte*)ptr2, count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Int16:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Int16>(ptr1, ptr2, count, static_cast<Int16>(std::numeric_limits<Byte>::min()), 
                                                 static_cast<Int16>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Int16>(ptr1, ptr2, count, 0, static_cast<Int16>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Int16>(ptr1, ptr2, count, 0, static_cast<Int16>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Int16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Int16*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Int16 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;

    case NTA_BasicType_UInt16:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, UInt16>(ptr1, ptr2, count, 0, static_cast<UInt16>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, UInt16>(ptr1, ptr2, count, 0, static_cast<UInt16>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, UInt16>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const UInt16*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const UInt16 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Int32:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Int32>(ptr1, ptr2, count, static_cast<Int32>(std::numeric_limits<Byte>::min()), 
                                                 static_cast<Int32>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Int32>(ptr1, ptr2, count, static_cast<Int32>(std::numeric_limits<Int16>::min()),
                                                  static_cast<Int32>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Int32>(ptr1, ptr2, count, 0, static_cast<Int32>(std::numeric_limits<UInt16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Int32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Int32>(ptr1, ptr2, count, 0, static_cast<Int32>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Int32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Int32>(ptr1, ptr2, count, 0, static_cast<Int32>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Int32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Int32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Int32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Int32*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Int32 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_UInt32:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, UInt32>(ptr1, ptr2, count, 0, static_cast<UInt32>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, UInt32>(ptr1, ptr2, count, 0, static_cast<UInt32>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, UInt32>(ptr1, ptr2, count, 0, static_cast<UInt32>(std::numeric_limits<UInt16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, UInt32>(ptr1, ptr2, count, 0, static_cast<UInt32>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, UInt32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, UInt32>(ptr1, ptr2, count, 0, static_cast<UInt32>(std::numeric_limits<UInt32>::max()));
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, UInt32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, UInt32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, UInt32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, UInt32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const UInt32*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const UInt32 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Int64:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Int64>(ptr1, ptr2, count, static_cast<Int64>(std::numeric_limits<Byte>::min()), 
                                                 static_cast<Int64>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Int64>(ptr1, ptr2, count, static_cast<Int64>(std::numeric_limits<Int16>::min()), 
                                                  static_cast<Int64>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Int64>(ptr1, ptr2, count, 0, static_cast<Int64>(std::numeric_limits<UInt16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Int64>(ptr1, ptr2, count, static_cast<Int64>(std::numeric_limits<Int32>::min()), 
                                                  static_cast<Int64>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Int64>(ptr1, ptr2, count, 0, static_cast<Int64>(std::numeric_limits<UInt32>::max()));
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Int64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Int64>(ptr1, ptr2, count, 0, static_cast<Int64>(std::numeric_limits<Int64>::max()));
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Int64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Int64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Int64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Int64*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Int64 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;

    case NTA_BasicType_UInt64:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<UInt16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<UInt32>::max()));
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, UInt64>(ptr1, ptr2, count, 0, static_cast<UInt64>(std::numeric_limits<Int64>::max()));
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, UInt64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, UInt64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, UInt64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, UInt64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const UInt64*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const UInt64 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Real32:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Real32>(ptr1, ptr2, count, static_cast<Real32>(std::numeric_limits<Byte>::min()), 
                                                  static_cast<Real32>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Real32>(ptr1, ptr2, count, static_cast<Real32>(std::numeric_limits<Int16>::min()), 
                                                   static_cast<Real32>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Real32>(ptr1, ptr2, count, 0.0f, static_cast<Real32>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Real32>(ptr1, ptr2, count, static_cast<Real32>(std::numeric_limits<Int32>::min()), 
                                                   static_cast<Real32>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Real32>(ptr1, ptr2, count, 0.0f, static_cast<Real32>(std::numeric_limits<UInt32>::max()));
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Real32>(ptr1, ptr2, count, static_cast<Real32>(std::numeric_limits<Int64>::min()), 
                                                   static_cast<Real32>(std::numeric_limits<Int64>::max()));
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Real32>(ptr1, ptr2, count, 0.0f, static_cast<Real32>(std::numeric_limits<UInt64>::max()));
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Real32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Real32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Real32>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Real32*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Real32 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Real64:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Real64>(ptr1, ptr2, count, static_cast<Real64>(std::numeric_limits<Byte>::min()), 
                                                  static_cast<Real64>(std::numeric_limits<Byte>::max()));
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Real64>(ptr1, ptr2, count, static_cast<Real64>(std::numeric_limits<Int16>::min()), 
                                                   static_cast<Real64>(std::numeric_limits<Int16>::max()));
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Real64>(ptr1, ptr2, count, static_cast<Real64>(std::numeric_limits<Byte>::min()), 
                                                    static_cast<Real64>(std::numeric_limits<UInt16>::max()));
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Real64>(ptr1, ptr2, count, static_cast<Real64>(std::numeric_limits<Int32>::min()), 
                                                   static_cast<Real64>(std::numeric_limits<Int32>::max()));
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Real64>(ptr1, ptr2, count, 0.0, static_cast<Real64>(std::numeric_limits<UInt32>::max()));
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Real64>(ptr1, ptr2, count, static_cast<Real64>(std::numeric_limits<Int64>::min()), 
                                                   static_cast<Real64>(std::numeric_limits<Int64>::max()));
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Real64>(ptr1, ptr2, count, 0.0, static_cast<Real64>(std::numeric_limits<UInt64>::max()));
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Real64>(ptr1, ptr2, count, static_cast<Real64>(-std::numeric_limits<Real32>::max()), 
                                                    static_cast<Real64>(std::numeric_limits<Real32>::max()));
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Real64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Real64>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Real64*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Real64 *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Bool:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, bool>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const bool*>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const bool *>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_SDR:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyarray<Byte, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int16:
        cpyarray<Int16, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt16:
        cpyarray<UInt16, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int32:
        cpyarray<Int32, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt32:
        cpyarray<UInt32, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Int64:
        cpyarray<Int64, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_UInt64:
        cpyarray<UInt64, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real32:
        cpyarray<Real32, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Real64:
        cpyarray<Real64, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_Bool:
        cpyarray<bool, Byte>(ptr1, ptr2, count);
        break;
      case NTA_BasicType_SDR:
        cpyIntoSDR(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const Byte*>(ptr2), count);
        break;
      case NTA_BasicType_Str:  // array of bytes in SDR becomes an array of strings.
        cpyIntoStr(reinterpret_cast<std::string *>(ptr1), reinterpret_cast<const Byte*>(ptr2), count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
	      break;
      }
      break;
    case NTA_BasicType_Str:
      switch (toType) {
      case NTA_BasicType_Byte:
        cpyFromStr(reinterpret_cast<Byte *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Int16:
        cpyFromStr(reinterpret_cast<Int16 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_UInt16:
        cpyFromStr(reinterpret_cast<UInt16 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Int32:
        cpyFromStr(reinterpret_cast<Int32 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_UInt32:
        cpyFromStr(reinterpret_cast<UInt32 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Int64:
        cpyFromStr(reinterpret_cast<Int64 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_UInt64:
        cpyFromStr(reinterpret_cast<UInt64 *>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Real32:
        cpyFromStr(reinterpret_cast<Real32*>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Real64:
        cpyFromStr(reinterpret_cast<Real64*>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Bool:
        cpyFromStr(reinterpret_cast<bool*>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_SDR:  // expecting an array of strings, each a string representation of a Byte.
        cpyFromStr(reinterpret_cast<Byte*>(ptr1), reinterpret_cast<const std::string *>(ptr2), count);
        break;
      case NTA_BasicType_Str:
        cpyarray<std::string, std::string>(ptr1, ptr2, count);
        break;
      default:
        NTA_THROW << "Could not perform array type conversion.";
        break;
      }
      break;

    default:
      break;
    }
  } catch (htm::Exception &e) {
    NTA_THROW << "Error Converting Array from " << BasicType::getName(fromType)
              << " to " << BasicType::getName(toType) << " " << e.getMessage();
  } catch (std::exception &e) {
    NTA_THROW << "Error Converting Array from " << BasicType::getName(fromType)
              << " to " << BasicType::getName(toType) << " " << e.what();
  }
}
