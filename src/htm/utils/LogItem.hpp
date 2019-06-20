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

/** @file
 * LogItem interface
 */

#ifndef NTA_LOG_ITEM_HPP
#define NTA_LOG_ITEM_HPP

#include <iostream>
#include <sstream>

namespace htm {

  /**
   * The type of log message.
   */
  typedef enum  { LogType_debug = 1, LogType_info, LogType_warn, LogType_error } LogType;

  /**
   * This enum represents the documented logging level of the debug logger.
   *
   * Use it like `LDEBUG(htm::LogLevel_XXX)`.
   */
  typedef enum { LogLevel_None = 0, LogLevel_Minimal, LogLevel_Normal, LogLevel_Verbose } LogLevel;

/**
 * @b Description
 * A LogItem represents a single log entry. It contains a stream that
 * accumulates a log message, and its destructor calls the logger.
 *
 * A LogItem contains an internal stream
 * which is used for building up an application message using
 * << operators.
 *
 */

class LogItem {
public:


  /**
   * Record information to be logged
   */
  LogItem(const char *filename, int line, htm::LogType type);

  /**
   * Destructor performs the logging
   */
  virtual ~LogItem();

  /*
   * Return the underlying stream object. Caller will use it to construct the
   * log message.
   */
  std::ostringstream &stream();

  static void setOutputFile(std::ostream &ostream);
  static void setLogLevel(LogLevel level);
  static LogLevel getLogLevel();
  static bool isDebug() { return (log_level_ == LogLevel_Verbose); }

protected:
  const char *filename_; // name of file
  int lineno_;           // line number in file
  LogType type_;
  std::ostringstream msg_;

private:
  static std::ostream *ostream_;
  static htm::LogLevel log_level_;
};

} // namespace htm

#endif // NTA_LOG_ITEM_HPP
