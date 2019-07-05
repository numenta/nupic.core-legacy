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
Environment Implementation
*/

#include <cctype> // toupper
#include <htm/os/Env.hpp>
#include <htm/utils/Log.hpp>
#include <stdlib.h>
#include <algorithm> // std::transform

using namespace htm;

bool Env::get(const std::string& name, std::string& value) {
  // warning: not reentrent or threadsafe
  char *ret = std::getenv(name.c_str());
  if (ret == nullptr)
    return false;
  value = ret;
  return true;
}


void Env::set(const std::string& name, const std::string& value) {
#if defined(NTA_OS_WINDOWS)
  _putenv_s(name.c_str(), value.c_str());
#else
  setenv(name.c_str(), value.c_str(), 1);
#endif
}

void Env::unset(const std::string& name) {
#if defined(NTA_OS_WINDOWS)
  _putenv_s(name.c_str(), "");
#else
  unsetenv(name.c_str());
#endif
}

// Bad practice to use the environ pointer.
//char ** Env::environ_ = nullptr;
//
//#if defined(NTA_OS_DARWIN)
//  #include <crt_externs.h>
//#else
//  extern char **environ;
//#endif


char **Env::getenv() {
    // @todo  if we find we need it.
    throw std::runtime_error("Not implemented");
//    if (environ_ != nullptr)
//    return environ_;
//
//#if defined(NTA_OS_DARWIN)
//  environ_ = *_NSGetEnviron();
//#else
//  environ_ = environ;
//#endif
//
//  return environ_;
}

static std::string
_getOptionEnvironmentVariable(const std::string &optionName) {
  std::string result = "NTA_";
  result += optionName;
  std::transform(result.begin(), result.end(), result.begin(), toupper);
  return result;
}

bool Env::isOptionSet(const std::string &optionName) {
  std::string envName = _getOptionEnvironmentVariable(optionName);
  std::string value;
  bool found = get(envName, value);
  return found;
}

std::string Env::getOption(const std::string &optionName, std::string defaultValue) {
  std::string envName = _getOptionEnvironmentVariable(optionName);
  std::string value;
  bool found = get(envName, value);
  if (!found)
    return defaultValue;
  else
    return value;
}
