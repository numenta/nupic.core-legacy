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

/** @file
 * Implementation of Fraction test
 */

#include "ExceptionTest.hpp"
#include <nupic/types/Exception.hpp>

using namespace nupic;

void ExceptionTest::RunTests()
{
  try
  {
    throw nupic::Exception("FFF", 123, "MMM");
  }
  catch (const Exception & e)
  {
    TEST(std::string(e.getFilename()) == std::string("FFF"));
    TEST(e.getLineNumber() == 123);
    TEST(std::string(e.getMessage()) == std::string("MMM"));
    TEST(std::string(e.getStackTrace()) == std::string(""));
  }

  try
  {
    throw nupic::Exception("FFF", 123, "MMM", "TB");
  }
  catch (const Exception & e)
  {
    TEST(std::string(e.getFilename()) == std::string("FFF"));
    TEST(e.getLineNumber() == 123);
    TEST(std::string(e.getMessage()) == std::string("MMM"));
    TEST(std::string(e.getStackTrace()) == std::string("TB"));
  }  
}

