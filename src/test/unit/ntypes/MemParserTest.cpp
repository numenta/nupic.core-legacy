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
 * Notes
 */

#include <nupic/ntypes/MemParser.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

using namespace nupic;

// -------------------------------------------------------
// Test using get methods
// -------------------------------------------------------
TEST(MemParserTest, getMethods) {
  std::stringstream ss;

  // Write one of each type to the stream
  unsigned long a = 10;
  long b = -20;
  double c = 1.5;
  float d = 1.6f;
  std::string e = "hello";

  ss << a << " " << b << " " << c << " " << d << " " << e << " ";

  // Read back
  MemParser in(ss, (UInt32)ss.str().size());

  unsigned long test_a = 0;
  in.get(test_a);
  ASSERT_EQ(a, test_a) << "get ulong";

  long test_b = 0;
  in.get(test_b);
  ASSERT_EQ(b, test_b) << "get long";

  double test_c = 0;
  in.get(test_c);
  ASSERT_EQ(c, test_c) << "get double";

  float test_d = 0;
  in.get(test_d);
  ASSERT_EQ(d, test_d) << "get float";

  std::string test_e = "";
  in.get(test_e);
  ASSERT_EQ(e, test_e) << "get string";

  // Test EOF
  ASSERT_ANY_THROW(in.get(test_e));
}

// -------------------------------------------------------
// Test passing in -1 for the size to read in entire stream
// -------------------------------------------------------
TEST(MemParserTest, PassInNegativeOne) {
  std::stringstream ss;

  // Write one of each type to the stream
  unsigned long a = 10;
  long b = -20;
  double c = 1.5;
  float d = 1.6f;
  std::string e = "hello";

  ss << a << " " << b << " " << c << " " << d << " " << e << " ";

  // Read back
  MemParser in(ss);

  unsigned long test_a = 0;
  in.get(test_a);
  ASSERT_EQ(a, test_a) << "get ulong b";

  long test_b = 0;
  in.get(test_b);
  ASSERT_EQ(b, test_b) << "get long b";

  double test_c = 0;
  in.get(test_c);
  ASSERT_EQ(c, test_c) << "get double b";

  float test_d = 0;
  in.get(test_d);
  ASSERT_EQ(d, test_d) << "get float b";

  std::string test_e = "";
  in.get(test_e);
  ASSERT_EQ(e, test_e) << "get string b";

  // Test EOF
  ASSERT_ANY_THROW(in.get(test_e));
}

// -------------------------------------------------------
// Test using >> operator
// -------------------------------------------------------
TEST(MemParserTest, RightShiftOperator) {
  std::stringstream ss;

  // Write one of each type to the stream
  unsigned long a = 10;
  long b = -20;
  double c = 1.5;
  float d = 1.6f;
  std::string e = "hello";

  ss << a << " " << b << " " << c << " " << d << " " << e << " ";

  // Read back
  MemParser in(ss, (UInt32)ss.str().size());

  unsigned long test_a = 0;
  long test_b = 0;
  double test_c = 0;
  float test_d = 0;
  std::string test_e = "";
  in >> test_a >> test_b >> test_c >> test_d >> test_e;
  ASSERT_EQ(a, test_a) << ">> ulong";
  ASSERT_EQ(b, test_b) << ">> long";
  ASSERT_EQ(c, test_c) << ">> double";
  ASSERT_EQ(d, test_d) << ">> float";
  ASSERT_EQ(e, test_e) << ">> string";

  // Test EOF
  ASSERT_ANY_THROW(in >> test_e);
}

// -------------------------------------------------------
// Test reading trying to read an int when we have a string
// -------------------------------------------------------
TEST(MemParserTest, ReadIntWhenStrig) {
  std::stringstream ss;
  ss << "hello";

  // Read back
  MemParser in(ss, (UInt32)ss.str().size());

  // Test EOF
  long v;
  ASSERT_ANY_THROW(in.get(v));
}
