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
 * Implementation of Dimensions test
 */

#include <sstream>
#include <gtest/gtest.h>
#include <nupic/ntypes/Dimensions.hpp>

using namespace nupic;

class DimensionsTest : public ::testing::Test {
public:

};

TEST_F(DimensionsTest, EmptyDimensions) {
  // empty dimensions (unspecified)
  Dimensions d;
  ASSERT_TRUE(d.isUnspecified());
  ASSERT_TRUE(d.isInvalid());
  ASSERT_TRUE(!d.isDontcare());
  EXPECT_EQ(d.getCount(), 0u);
  ASSERT_ANY_THROW(UInt x = d[0]);
  EXPECT_STREQ("[unspecified]", d.toString().c_str());
  ASSERT_EQ(0u, d.size());
}

TEST_F(DimensionsTest, DontCareDimensions) {
  // dontcare dimensions [0]
  Dimensions d;
  d.push_back(0u);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(d.isDontcare());
  ASSERT_TRUE(d.isInvalid());
  EXPECT_STREQ("[dontcare]", d.toString().c_str());
  ASSERT_EQ(d.getCount(), 0u);
  ASSERT_EQ(1u, d.size());
}

TEST_F(DimensionsTest, InvalidDimensions) {
  // invalid dimensions
  Dimensions d;
  d.push_back(1u);
  d.push_back(0u);
  ASSERT_FALSE(d.isUnspecified());
  ASSERT_FALSE(d.isDontcare());
  ASSERT_TRUE(d.isInvalid());
  EXPECT_STREQ("[ 1 0 ] (invalid)", d.toString().c_str());
  ASSERT_EQ(d.getCount(), 0u);
  ASSERT_EQ(1u, d[0]);
  ASSERT_EQ(0u, d[1]);
  ASSERT_ANY_THROW(d[2] = 2);
  ASSERT_EQ(2u, d.size());
}

TEST_F(DimensionsTest, ValidDimensions) {
  // valid dimensions [2,3]
  // two rows, three columns
  Dimensions d;
  d.push_back(2u);
  d.push_back(3u);
  ASSERT_FALSE(d.isUnspecified());
  ASSERT_FALSE(d.isDontcare());
  ASSERT_FALSE(d.isInvalid());
  EXPECT_STREQ("[2 3]", d.toString().c_str());
  ASSERT_EQ(2u, d[0]);
  ASSERT_EQ(3u, d[1]);
  ASSERT_ANY_THROW(UInt x = d[2]);
  ASSERT_EQ(6u, d.getCount());
  ASSERT_EQ(2u, d.size());
}


TEST_F(DimensionsTest, AlternateConstructor) {
  // alternate constructor
  std::vector<UInt> x;
  x.push_back(2);
  x.push_back(5);
  Dimensions d(x);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(!d.isDontcare());
  ASSERT_TRUE(!d.isInvalid());
  ASSERT_TRUE(d.isSpecified());

  Dimensions c(2, 5);
  ASSERT_TRUE(c == d);

  Dimensions e(std::vector<UInt>({ 2,5 }));
  ASSERT_TRUE(e == d);

  ASSERT_EQ(2u, d[0]);
  ASSERT_EQ(5u,  d[1]);
  ASSERT_ANY_THROW(UInt y = d[2]);
  ASSERT_EQ(2u, d.size());
}

TEST_F(DimensionsTest, Overloads) {
  Dimensions d1 = { 1,2,3 };
  Dimensions d2;
  std::stringstream ss;
  ss << d1;
  ss >> d2;
  EXPECT_EQ(d1, d2);

}

