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

/** @file */

//----------------------------------------------------------------------

#include "LoggingException.hpp"
#include "LogItem.hpp"
#include <iostream>
using namespace nupic;

LoggingException::~LoggingException() throw() {
  if (!alreadyLogged_) {
    // Let LogItem do the work for us. This code is a bit complex
    // because LogItem was designed to be used from a logging macro
    LogItem *li = new LogItem(filename_.c_str(), lineno_, LogItem::error);
    li->stream() << getMessage();
    delete li;

    alreadyLogged_ = true;
  }
}
