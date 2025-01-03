/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file
 * Basic C++ type definitions used throughout `nupic.core` and rely on `Types.h`
 */

#ifndef NTA_TYPES_HPP
#define NTA_TYPES_HPP

#include <nupic/types/Types.h>

//----------------------------------------------------------------------

namespace nupic {
/**
 * @name Basic types
 *
 * @{
 */

/**
 * Represents a 8-bit byte.
 */
typedef NTA_Byte Byte;

/**
 * Represents a 16-bit signed integer.
 */
typedef NTA_Int16 Int16;

/**
 * Represents a 16-bit unsigned integer.
 */
typedef NTA_UInt16 UInt16;

/**
 * Represents a 32-bit signed integer.
 */
typedef NTA_Int32 Int32;

/**
 * Represents a 32-bit unsigned integer.
 */
typedef NTA_UInt32 UInt32;

/**
 * Represents a 64-bit signed integer.
 */
typedef NTA_Int64 Int64;

/**
 * Represents a 64-bit unsigned integer.
 */
typedef NTA_UInt64 UInt64;

/**
 * Represents a 32-bit real number(a floating-point number).
 */
typedef NTA_Real32 Real32;

/**
 * Represents a 64-bit real number(a floating-point number).
 */
typedef NTA_Real64 Real64;

/**
 * Represents an opaque handle/pointer, same as `void *`
 */
typedef NTA_Handle Handle;

/**
 * Represents an opaque pointer, same as `uintptr_t`
 */
typedef NTA_UIntPtr UIntPtr;

/**
 * @}
 */

/**
 * @name Flexible types
 *
 * The following are flexible types depending on `NTA_DOUBLE_PRECISION` and
 * `NTA_BIG_INTEGER`.
 *
 * @{
 *
 */

/**
 * Represents a real number(a floating-point number).
 *
 * Same as nupic::Real64 if `NTA_DOUBLE_PRECISION` is defined, nupic::Real32
 * otherwise.
 */
typedef NTA_Real Real;

/**
 * Represents a signed integer.
 *
 * Same as nupic::Int64 if `NTA_BIG_INTEGER` is defined, nupic::Int32 otherwise.
 */
typedef NTA_Int Int;

/**
 * Represents a unsigned integer.
 *
 * Same as nupic::UInt64 if `NTA_BIG_INTEGER` is defined, nupic::UInt32
 * otherwise.
 */
typedef NTA_UInt UInt;

/**
 * Represents lengths of arrays, strings and so on.
 */
typedef NTA_Size Size;

/**
 * @}
 */

/**
 * This enum represents the documented logging level of the debug logger.
 *
 * Use it like `LDEBUG(nupic::LogLevel_XXX)`.
 */
enum LogLevel {
  /**
   * Log level: None.
   */
  LogLevel_None = NTA_LogLevel_None,
  /**
   * Log level: Minimal.
   */
  LogLevel_Minimal,
  /**
   * Log level: Normal.
   */
  LogLevel_Normal,
  /**
   * Log level: Verbose.
   */
  LogLevel_Verbose,
};

} // end namespace nupic

#ifdef SWIG
#undef NTA_INTERNAL
#endif // SWIG

#endif // NTA_TYPES_HPP
