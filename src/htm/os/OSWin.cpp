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
 * Win32 Implementations for the OS class
 */

#if defined(NTA_OS_WINDOWS)
#include <windows.h>
#include <shlobj.h>

#include <htm/os/OS.hpp>
#include <htm/os/Path.hpp>
#include <htm/os/Directory.hpp>
#include <htm/os/Env.hpp>
#include <htm/os/Path.hpp>
#include <htm/utils/Log.hpp>


using namespace htm;

std::string OS::getHomeDir() {
  std::string homeDrive;
  std::string homePath;
  bool found = Env::get("HOMEDRIVE", homeDrive);
  NTA_CHECK(found) << "'HOMEDRIVE' environment variable is not defined";
  found = Env::get("HOMEPATH", homePath);
  NTA_CHECK(found) << "'HOMEPATH' environment variable is not defined";
  return homeDrive + homePath;
}

std::string OS::getUserName() {
  std::string username;
  bool found = Env::get("USERNAME", username);
  NTA_CHECK(found) << "Environment variable USERNAME is not defined";

  return username;
}

int OS::getLastErrorCode() { return ::GetLastError(); }

std::string OS::getErrorMessageFromErrorCode(int errorCode) {
  // Retrieve the system error message for the last-error code
  LPVOID lpMsgBuf;

  DWORD msgLen = ::FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPSTR)&lpMsgBuf, 0, NULL);

  std::ostringstream errMessage;
  if (msgLen > 0) {
    errMessage.write((LPSTR)lpMsgBuf, msgLen);
  } else {
    errMessage << "code: " << errorCode;
  }

  LocalFree(lpMsgBuf);

  return errMessage.str();
}

std::string OS::getErrorMessage() {
  return getErrorMessageFromErrorCode(getLastErrorCode());
}


#endif //#if defined(NTA_OS_WINDOWS)

