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
#include <nupic/os/Env.hpp>

using namespace nupic;

TEST(EnvTest, Basic) {
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
