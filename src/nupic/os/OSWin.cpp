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
 * Win32 Implementations for the OS class
 */

#if defined(NTA_OS_WINDOWS)
#include <shlobj.h>
#include <windows.h>

#include <boost/shared_ptr.hpp>
#include <nupic/os/Directory.hpp>
#include <nupic/os/DynamicLibrary.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

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
