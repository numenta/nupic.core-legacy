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
 * Implementation for Directory test
 */


#include <nupic/os/Regex.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace nupic;


TEST(RegexTest, Basic)
{
  
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
  try
  {
    ASSERT_TRUE(regex::match("", ""));
    ASSERT_TRUE(false);
  }
  catch (...)
  {
  }
   
  try
  {
    ASSERT_TRUE(regex::match("xyz[", ""));
    ASSERT_TRUE(false);
  }
  catch (...)
  {
  }
}

