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
 * LogItem implementation
 */

#include <iostream> // cerr
#include <nupic/types/Exception.hpp>
#include <nupic/utils/LogItem.hpp>
#include <stdexcept> // runtime_error

using namespace nupic;

std::ostream *LogItem::ostream_ = nullptr;

void LogItem::setOutputFile(std::ostream &ostream) { ostream_ = &ostream; }

LogItem::LogItem(const char *filename, int line, LogLevel level)
    : filename_(filename), lineno_(line), level_(level), msg_("") {}

LogItem::~LogItem() {
  std::string slevel;
  switch (level_) {
  case debug:
    slevel = "DEBUG:";
    break;
  case warn:
    slevel = "WARN: ";
    break;
  case info:
    slevel = "INFO: ";
    break;
  case error:
    slevel = "ERR:";
    break;
  default:
    slevel = "Unknown: ";
    break;
  }

  if (ostream_ == nullptr)
    ostream_ = &(std::cerr);

  (*ostream_) << slevel << "  " << msg_.str();

  if (level_ == error)
    (*ostream_) << " [" << filename_ << " line " << lineno_ << "]";

  (*ostream_) << std::endl;
}

std::ostringstream &LogItem::stream() { return msg_; }
