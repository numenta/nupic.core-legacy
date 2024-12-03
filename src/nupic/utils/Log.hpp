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

/**
 * @file
 * Definition of C++ macros for logging.
 */

#ifndef NTA_LOG2_HPP
#define NTA_LOG2_HPP

#include <nupic/utils/LogItem.hpp>
#include <nupic/utils/LoggingException.hpp>

#define NTA_DEBUG                                                              \
  nupic::LogItem(__FILE__, __LINE__, nupic::LogItem::debug).stream()

// Can be used in Loggable classes
#define NTA_LDEBUG(level)                                                      \
  if (logLevel_ < (level)) {                                                   \
  } else                                                                       \
    nupic::LogItem(__FILE__, __LINE__, nupic::LogItem::debug).stream()

// For informational messages that report status but do not indicate that
// anything is wrong
#define NTA_INFO                                                               \
  nupic::LogItem(__FILE__, __LINE__, nupic::LogItem::info).stream()

// For messages that indicate a recoverable error or something else that it may
// be important for the end user to know about.
#define NTA_WARN                                                               \
  nupic::LogItem(__FILE__, __LINE__, nupic::LogItem::warn).stream()

// To throw an exception and make sure the exception message is logged
// appropriately
#define NTA_THROW throw nupic::LoggingException(__FILE__, __LINE__)

// The difference between CHECK and ASSERT is that ASSERT is for
// performance critical code and can be disabled in a release
// build. Both throw an exception on error.

#define NTA_CHECK(condition)                                                   \
  if (condition) {                                                             \
  } else                                                                       \
    NTA_THROW << "CHECK FAILED: \"" << #condition << "\" "

#ifdef NTA_ASSERTIONS_ON

#define NTA_ASSERT(condition)                                                  \
  if (condition) {                                                             \
  } else                                                                       \
    NTA_THROW << "ASSERTION FAILED: \"" << #condition << "\" "

#else

// NTA_ASSERT macro does nothing.
// The second line should never be executed, or even compiled, but we
// need something that is syntactically compatible with NTA_ASSERT
#define NTA_ASSERT(condition)                                                  \
  if (1) {                                                                     \
  } else                                                                       \
    nupic::LogItem(__FILE__, __LINE__, nupic::LogItem::debug).stream()

#endif // NTA_ASSERTIONS_ON

#endif // NTA_LOG2_HPP
