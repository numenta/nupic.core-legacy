/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
 * LogItem implementation
 */

#include <iostream> // cerr
#include <htm/types/Exception.hpp>
#include <htm/utils/LogItem.hpp>
#include <stdexcept> // runtime_error

using namespace htm;

// Initialize static members
std::ostream *LogItem::ostream_ = nullptr;
htm::LogLevel LogItem::log_level_ = LogLevel::LogLevel_None;

// Static functions
void LogItem::setOutputFile(std::ostream &ostream) { ostream_ = &ostream; }
void LogItem::setLogLevel(LogLevel level) { log_level_ = level; }
LogLevel LogItem::getLogLevel() {return log_level_; }


// Constructor
// Construct a new instance for each item to be logged.
LogItem::LogItem(const char *filename, int line, htm::LogType type)
    : filename_(filename), lineno_(line), type_(type), msg_("") {}

LogItem::~LogItem() {
  std::string slevel;
  switch (type_) {
  case LogType_debug:
    slevel = "DEBUG:";
    break;
  case LogType_warn:
    slevel = "WARN: ";
    break;
  case LogType_info:
    slevel = "INFO: ";
    break;
  case LogType_error:
    slevel = "ERR:";
    break;
  default:
    slevel = "Unknown: ";
    break;
  }

  if (ostream_ == nullptr)
    ostream_ = &(std::cerr);

  (*ostream_) << slevel << "  " << msg_.str();

  if (type_ == LogType_error)
    (*ostream_) << " [" << filename_ << " line " << lineno_ << "]";

  (*ostream_) << std::endl;
}

std::ostringstream &LogItem::stream() { return msg_; }
