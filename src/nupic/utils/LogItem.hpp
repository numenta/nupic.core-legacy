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
 * LogItem interface
 */

#ifndef NTA_LOG_ITEM_HPP
#define NTA_LOG_ITEM_HPP

#include <iostream>
#include <sstream>

namespace nupic {

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
  typedef enum { debug, info, warn, error } LogLevel;
  /**
   * Record information to be logged
   */
  LogItem(const char *filename, int line, LogLevel level);

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

protected:
  const char *filename_; // name of file
  int lineno_;           // line number in file
  LogLevel level_;
  std::ostringstream msg_;

private:
  static std::ostream *ostream_;
};

} // namespace nupic

#endif // NTA_LOG_ITEM_HPP
