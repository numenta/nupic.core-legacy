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
 * Implementation of Fraction test
 */

#include <gtest/gtest.h>
#include <nupic/types/Exception.hpp>

using namespace nupic;

TEST(ExceptionTest, Basic) {
  try {
    throw nupic::Exception("FFF", 123, "MMM");
  } catch (const Exception &e) {
    ASSERT_EQ(std::string(e.getFilename()), std::string("FFF"));
    ASSERT_EQ(e.getLineNumber(), 123);
    ASSERT_EQ(std::string(e.getMessage()), std::string("MMM"));
    ASSERT_EQ(std::string(e.getStackTrace()), std::string(""));
  }

  try {
    throw nupic::Exception("FFF", 123, "MMM", "TB");
  } catch (const Exception &e) {
    ASSERT_EQ(std::string(e.getFilename()), std::string("FFF"));
    ASSERT_EQ(e.getLineNumber(), 123);
    ASSERT_EQ(std::string(e.getMessage()), std::string("MMM"));
    ASSERT_EQ(std::string(e.getStackTrace()), std::string("TB"));
  }
}
