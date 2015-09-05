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

/**
 * @file
 */

#include "OSTest.hpp"
#include <nupic/os/Env.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/os/Directory.hpp>

using namespace nupic;

OSTest::OSTest() {};

OSTest::~OSTest() {};


void OSTest::RunTests()
{
#if defined(NTA_OS_WINDOWS)

#else
  // save the parts of the environment we'll be changing
  std::string savedHOME;
  bool isHomeSet = Env::get("HOME", savedHOME);
  
  Env::set("HOME", "/home1/myhome");
  Env::set("USER", "user1");
  Env::set("LOGNAME", "logname1");
  
  TESTEQUAL2_STR("OS::getHomeDir", "/home1/myhome", OS::getHomeDir().c_str());
  bool caughtException = false;
  Env::unset("HOME");
  std::string dummy;
  try {
    dummy = OS::getHomeDir();
  } catch (...) {
    caughtException = true;
  }
  TEST2("getHomeDir -- HOME not set", caughtException == true);
  // restore HOME
  if (isHomeSet) {
    Env::set("HOME", savedHOME);
  }


#endif

  // Test getUserName()
  {
#if defined(NTA_OS_WINDOWS)
    Env::set("USERNAME", "123");
    TEST(OS::getUserName() == "123");    
#else
    // case 1 - USER defined
    Env::set("USER", "123");
    TEST(OS::getUserName() == "123");

    // case 2 - USER not defined, LOGNAME defined
    Env::unset("USER");
    Env::set("LOGNAME", "456");
    TEST(OS::getUserName() == "456");

    // case 3 - USER and LOGNAME not defined
    Env::unset("LOGNAME");
    
    std::stringstream ss("");
    ss << getuid();
    TEST(OS::getUserName() == ss.str());
#endif
  }
  

  // Test getStackTrace()
  {
#if defined(NTA_OS_WINDOWS)
//    std::string stackTrace = OS::getStackTrace();
//    TEST(!stackTrace.empty());  
//
//    stackTrace = OS::getStackTrace();
//    TEST(!stackTrace.empty());
#endif  
  }

  // Test executeCommand()
  {
    std::string output = OS::executeCommand("echo ABCDefg");

    TEST(output == "ABCDefg\n");
  }


}

