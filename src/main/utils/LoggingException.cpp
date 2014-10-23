/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file */

//----------------------------------------------------------------------

#include "LoggingException.hpp"
#include "LogItem.hpp"
#include <iostream>
using namespace nta;

LoggingException::~LoggingException() throw()
{
  if (!alreadyLogged_) {
    // Let LogItem do the work for us. This code is a bit complex
    // because LogItem was designed to be used from a logging macro
#if (defined(NTA_PLATFORM_win32) || defined(NTA_PLATFORM_win64)) && defined(NTA_COMPILER_MSVC) && defined(NDEBUG)
    LogItem *li = new LogItem(filename_.c_str(), lineno_, LogItem::info);
#else
    LogItem *li = new LogItem(filename_.c_str(), lineno_, LogItem::error);
#endif
	li->stream() << getMessage();
    delete li;
    alreadyLogged_ = true;
  }
}

