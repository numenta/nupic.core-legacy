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


#include <nupic/os/Env.hpp>
#include <gtest/gtest.h>

using namespace nupic;


TEST(EnvTest, Basic)
{
  std::string name;
  std::string value;
  bool result;
  
  // get value that is not set
  value = "DONTCHANGEME";
  result = Env::get("NOTDEFINED", value);
  ASSERT_FALSE(result);
  EXPECT_STREQ("DONTCHANGEME", value.c_str());
  
  // get value that should be set
  value = "";
  result = Env::get("PATH", value);
  ASSERT_TRUE(result);
  ASSERT_TRUE(value.length() > 0) << "get path value";
  
  // set a value
  name = "myname";
  value = "myvalue";
  Env::set(name, value);
  
  // retrieve it
  value = "";
  result = Env::get(name, value);
  ASSERT_TRUE(result);
  EXPECT_STREQ("myvalue", value.c_str());
  
  // set it to something different
  value = "mynewvalue";
  Env::set(name, value);
  
  // retrieve the new value
  result = Env::get(name, value);
  ASSERT_TRUE(result);
  EXPECT_STREQ("mynewvalue", value.c_str());
  
  // delete the value
  value = "DONTCHANGEME";
  Env::unset(name);
  result = Env::get(name, value);
  ASSERT_FALSE(result);
  EXPECT_STREQ("DONTCHANGEME", value.c_str());
  
  // delete a value that is not set
  // APR response is not documented. Will see a warning if 
  // APR reports an error. 
  // Is there any way to do an actual test here? 
  Env::unset(name);
}

