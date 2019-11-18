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
 * Implementation of Fraction test
 */

#include <gtest/gtest.h>
#include <htm/types/Exception.hpp>
#include <htm/utils/Log.hpp>

namespace testing {
    
using namespace htm;

TEST(ExceptionTest, Basic) {
  try {
    throw htm::Exception("FFF", 123ul, "MMM");
  } catch (const Exception &e) {
    EXPECT_STREQ(e.what(), "Exception: FFF(123) message: MMM");
    ASSERT_EQ(std::string(e.getFilename()), std::string("FFF"));
    ASSERT_EQ(e.getLineNumber(), 123ul);
    ASSERT_EQ(std::string(e.getMessage()), std::string("MMM"));
    ASSERT_EQ(std::string(e.getStackTrace()), std::string(""));
  }

  try {
    throw htm::Exception("FFF", 123ul, "MMM", "TB");
  } catch (const Exception &e) {
    EXPECT_STREQ(e.what(), "Exception: FFF(123) message: MMM");
    ASSERT_EQ(std::string(e.getFilename()), std::string("FFF"));
    ASSERT_EQ(e.getLineNumber(), 123l);
    ASSERT_EQ(std::string(e.getMessage()), std::string("MMM"));
    ASSERT_EQ(std::string(e.getStackTrace()), std::string("TB"));
  }

}

TEST(ExceptionTest, Argument_Streaming) {
  try {
  NTA_THROW << "This msg";
  } catch (const Exception &e) {
    EXPECT_STREQ(e.getMessage(), "This msg");
    EXPECT_STREQ(e.what(), "Exception: ExceptionTest.cpp(55) message: This msg");
  }
}

} // namespace testing
