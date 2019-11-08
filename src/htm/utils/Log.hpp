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

/**
 * @file
 * Definition of C++ macros for logging.
 */

#ifndef NTA_LOG2_HPP
#define NTA_LOG2_HPP

#include <iostream>
#include <htm/types/Exception.hpp>

namespace htm {
enum class LogLevel { LogLevel_None = 0, LogLevel_Minimal=1, LogLevel_Normal=2, LogLevel_Verbose=3 };
// change this in your class to set log level using Network.setLogLevel(level);
extern thread_local LogLevel NTA_LOG_LEVEL; 

//this code intentionally uses "if() dosomething" instead of "if() { dosomething }" 
// as the macro expects another "<< "my clever message";
// so it eventually becomes: `if() std::cout << "DEBUG:\t" << "users message";`
//
//Expected usage: 
//<your class>:
//Network::setLogLevel(LogLevel::LogLevel_Verbose);
//NTA_WARN << "Hello World!" << std::endl; //shows
//NTA_DEBUG << "more details how cool this is"; //not showing under "Normal" log level
//NTA_ERR << "You'll always see this, HAHA!";
//NTA_THROW << "crashing for a good cause";

#define NTA_DEBUG                                                \
  if (NTA_LOG_LEVEL >= LogLevel::LogLevel_Verbose)               \
    std::cout << "DEBUG:\t" << Path::getBasename(__FILE__) << ":" << __LINE__ << ": " 

// For informational messages that report status but do not indicate that
// anything is wrong
#define NTA_INFO                                                               \
  if (NTA_LOG_LEVEL >= LogLevel::LogLevel_Normal)                              \
  std::cout << "INFO:\t" << Path::getBasename(__FILE__) << ":" << __LINE__ << ": "

// For messages that indicate a recoverable error or something else that it may
// be important for the end user to know about.
#define NTA_WARN                                                               \
  if (NTA_LOG_LEVEL >= LogLevel::LogLevel_Normal)                              \
  std::cout << "WARN:\t" << Path::getBasename(__FILE__) << ":" << __LINE__ << ": "

// To throw an exception and make sure the exception message is logged
// appropriately
#define NTA_THROW throw htm::Exception(__FILE__, __LINE__)

// The difference between CHECK and ASSERT is that ASSERT is for
// performance critical code and can be disabled in a release
// build. Both throw an exception on error (if NTA_ASSERTIONS_ON is set).

#define NTA_CHECK(condition)                                                   \
  if (not (condition) )                                                        \
    NTA_THROW << "CHECK FAILED: \"" << #condition << "\" "

#ifdef NTA_ASSERTIONS_ON
// With NTA_ASSERTIONS_ON, NTA_ASSERT macro throws exception if condition is false.
// NTA_ASSERTIONS_ON should be set ON only in debug mode.
#define NTA_ASSERT(condition) NTA_CHECK(condition)

#else
// Without NTA_ASSERTIONS_ON, NTA_ASSERT macro does nothing.
// The second line (with `if(false)`) should never be executed, or even compiled, but we
// need something that is syntactically compatible with NTA_ASSERT << "msg";
#define NTA_ASSERT(condition)                                                  \
  if (false) std::cerr << "This line should never happen"

#endif // NTA_ASSERTIONS_ON
}
#endif // NTA_LOG2_HPP
