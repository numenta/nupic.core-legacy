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

/**
 * @file
 */

#include <gtest/gtest.h>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>

using namespace nupic;

TEST(OSTest, Basic) {
#if defined(NTA_OS_WINDOWS)

#else
  // save the parts of the environment we'll be changing
  std::string savedHOME;
  bool isHomeSet = Env::get("HOME", savedHOME);

  Env::set("HOME", "/home1/myhome");
  Env::set("USER", "user1");
  Env::set("LOGNAME", "logname1");

  EXPECT_STREQ("/home1/myhome", OS::getHomeDir().c_str()) << "OS::getHomeDir";
  bool caughtException = false;
  Env::unset("HOME");
  std::string dummy;
  try {
    dummy = OS::getHomeDir();
  } catch (...) {
    caughtException = true;
  }
  ASSERT_TRUE(caughtException) << "getHomeDir -- HOME not set";
  // restore HOME
  if (isHomeSet) {
    Env::set("HOME", savedHOME);
  }

#endif

  // Test getUserName()
  {
#if defined(NTA_OS_WINDOWS)
    Env::set("USERNAME", "123");
    ASSERT_TRUE(OS::getUserName() == "123");
#else
    // case 1 - USER defined
    Env::set("USER", "123");
    ASSERT_TRUE(OS::getUserName() == "123");

    // case 2 - USER not defined, LOGNAME defined
    Env::unset("USER");
    Env::set("LOGNAME", "456");
    ASSERT_TRUE(OS::getUserName() == "456");

    // case 3 - USER and LOGNAME not defined
    Env::unset("LOGNAME");

    std::stringstream ss("");
    ss << getuid();
    ASSERT_TRUE(OS::getUserName() == ss.str());
#endif
  }

  // Test getStackTrace()
  {
#if defined(NTA_OS_WINDOWS)
//    std::string stackTrace = OS::getStackTrace();
//    ASSERT_TRUE(!stackTrace.empty());
//
//    stackTrace = OS::getStackTrace();
//    ASSERT_TRUE(!stackTrace.empty());
#endif
  }

  // Test executeCommand()
  {
    std::string output = OS::executeCommand("echo ABCDefg");

    ASSERT_TRUE(output == "ABCDefg\n");
  }
}
