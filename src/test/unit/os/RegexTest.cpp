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
 * Implementation for Directory test
 */

#include <gtest/gtest.h>
#include <nupic/os/Regex.hpp>

using namespace std;
using namespace nupic;

TEST(RegexTest, Basic) {

  ASSERT_TRUE(regex::match(".*", ""));
  ASSERT_TRUE(regex::match(".*", "dddddfsdsgregegr"));
  ASSERT_TRUE(regex::match("d.*", "d"));
  ASSERT_TRUE(regex::match("^d.*", "ddsfffdg"));
  ASSERT_TRUE(!regex::match("d.*", ""));
  ASSERT_TRUE(!regex::match("d.*", "a"));
  ASSERT_TRUE(!regex::match("^d.*", "ad"));
  ASSERT_TRUE(!regex::match("Sensor", "CategorySensor"));

  ASSERT_TRUE(regex::match("\\\\", "\\"));

  //  ASSERT_TRUE(regex::match("\\w", "a"));
  //  ASSERT_TRUE(regex::match("\\d", "3"));
  //  ASSERT_TRUE(regex::match("\\w{3}", "abc"));
  //  ASSERT_TRUE(regex::match("^\\w{3}$", "abc"));
  //  ASSERT_TRUE(regex::match("[\\w]{3}", "abc"));

  ASSERT_TRUE(regex::match("[A-Za-z0-9_]{3}", "abc"));

  // Invalid expression tests (should throw)
  try {
    ASSERT_TRUE(regex::match("", ""));
    ASSERT_TRUE(false);
  } catch (...) {
  }

  try {
    ASSERT_TRUE(regex::match("xyz[", ""));
    ASSERT_TRUE(false);
  } catch (...) {
  }
}
