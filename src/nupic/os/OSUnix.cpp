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
 * Unix Implementations for the OS class
 */

#if !defined(NTA_OS_WINDOWS)

#include <apr-1/apr_errno.h>
#include <apr-1/apr_network_io.h>
#include <apr-1/apr_time.h>
#include <cstdlib>
#include <fstream>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>
#include <sys/types.h>
#include <unistd.h> // getuid()

using namespace nupic;

std::string OS::getErrorMessage() {
  char buff[1024];
  apr_status_t st = apr_get_os_error();
  ::apr_strerror(st, buff, 1024);
  return std::string(buff);
}

std::string OS::getHomeDir() {
  std::string home;
  bool found = Env::get("HOME", home);
  if (!found)
    NTA_THROW << "'HOME' environment variable is not defined";
  return home;
}

std::string OS::getUserName() {
  std::string username;
  bool found = Env::get("USER", username);

  // USER isn't always set inside a cron job
  if (!found)
    found = Env::get("LOGNAME", username);

  if (!found) {
    NTA_WARN << "OS::getUserName -- USER and LOGNAME environment variables are "
                "not set. Using userid = "
             << getuid();
    std::stringstream ss("");
    ss << getuid();
    username = ss.str();
  }

  return username;
}

int OS::getLastErrorCode() { return errno; }

std::string OS::getErrorMessageFromErrorCode(int errorCode) {
  std::stringstream errorMessage;
  char errorBuffer[1024];
  errorBuffer[0] = '\0';

#if defined(__APPLE__) || (defined(NTA_ARCH_64) && defined(NTA_OS_SPARC))
  int result = ::strerror_r(errorCode, errorBuffer, 1024);
  if (result == 0)
    errorMessage << errorBuffer;
#else
  char *result = ::strerror_r(errorCode, errorBuffer, 1024);
  if (result != nullptr)
    errorMessage << errorBuffer;
#endif
  else
    errorMessage << "Error code " << errorCode;
  return errorMessage.str();
}

#endif
