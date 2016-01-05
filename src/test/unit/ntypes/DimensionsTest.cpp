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
 * Implementation of BasicType test
 */

#include <nupic/ntypes/Dimensions.hpp>
#include <gtest/gtest.h>

using namespace nupic;

class DimensionsTest : public ::testing::Test
{
public:
  DimensionsTest()
  {
    zero.push_back(0);

    one_two.push_back(1);
    one_two.push_back(2);

    three_four.push_back(3);
    three_four.push_back(4);
  }

  Coordinate zero; // [0];
  Coordinate one_two; // [1,2]
  Coordinate three_four; // [3,4]

  //internal helper method
  static std::string vecToString(std::vector<size_t> vec)
  {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); i++)
    {
      ss << vec[i];
      if (i != vec.size()-1)
        ss << " ";
    }
    ss << "]";
    return ss.str();
  }
};


TEST_F(DimensionsTest, EmptyDimensions)
{
  // empty dimensions (unspecified)
  Dimensions d;
  ASSERT_TRUE(d.isUnspecified());
  ASSERT_TRUE(d.isValid());
  ASSERT_TRUE(!d.isDontcare());
  ASSERT_ANY_THROW(d.getCount());
  ASSERT_ANY_THROW(d.getDimension(0));
  EXPECT_STREQ("[unspecified]", d.toString().c_str());
  ASSERT_ANY_THROW(d.getIndex(one_two));
  ASSERT_ANY_THROW(d.getCount());
  ASSERT_ANY_THROW(d.getDimension(0));
  ASSERT_EQ((unsigned int)0, d.getDimensionCount());
}

TEST_F(DimensionsTest, DontCareDimensions)
{
  // dontcare dimensions [0]
  Dimensions d;
  d.push_back(0);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(d.isDontcare());
  ASSERT_TRUE(d.isValid());
  EXPECT_STREQ("[dontcare]", d.toString().c_str());
  ASSERT_ANY_THROW(d.getIndex(zero));
  ASSERT_ANY_THROW(d.getCount());
  ASSERT_EQ((unsigned int)0, d.getDimension(0));
  ASSERT_EQ((unsigned int)1, d.getDimensionCount());
}

TEST_F(DimensionsTest, InvalidDimensions)
{
  // invalid dimensions
  Dimensions d;
  d.push_back(1);
  d.push_back(0);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(!d.isDontcare());
  ASSERT_TRUE(!d.isValid());
  EXPECT_STREQ("[1 0] (invalid)", d.toString().c_str());
  ASSERT_ANY_THROW(d.getIndex(one_two));
  ASSERT_ANY_THROW(d.getCount());
  ASSERT_EQ((unsigned int)1, d.getDimension(0));
  ASSERT_EQ((unsigned int)0, d.getDimension(1));
  ASSERT_ANY_THROW(d.getDimension(2));
  ASSERT_EQ((unsigned int)2, d.getDimensionCount());
}

TEST_F(DimensionsTest, ValidDimensions)
{
  // valid dimensions [2,3]
  // two rows, three columns
  Dimensions d;
  d.push_back(2);
  d.push_back(3);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(!d.isDontcare());
  ASSERT_TRUE(d.isValid());
  EXPECT_STREQ("[2 3]", d.toString().c_str());
  ASSERT_EQ((unsigned int)2, d.getDimension(0));
  ASSERT_EQ((unsigned int)3, d.getDimension(1));
  ASSERT_ANY_THROW(d.getDimension(2));
  ASSERT_EQ((unsigned int)6, d.getCount());
  ASSERT_EQ((unsigned int)5, d.getIndex(one_two));
  ASSERT_EQ((unsigned int)2, d.getDimensionCount());
}

TEST_F(DimensionsTest, Check2DXMajor)
{
  //check a two dimensional matrix for proper x-major ordering
  std::vector<size_t> x;
  x.push_back(4);
  x.push_back(5);
  Dimensions d(x);
  size_t testDim1 = 4;
  size_t testDim2 = 5;
  for(size_t i = 0; i < testDim1; i++)
  {
    for(size_t j = 0; j < testDim2; j++)
    {
      Coordinate testCoordinate;
      testCoordinate.push_back(i);
      testCoordinate.push_back(j);

      ASSERT_EQ(i+j*testDim1, d.getIndex(testCoordinate));
      ASSERT_EQ(vecToString(testCoordinate),
                vecToString(d.getCoordinate(i+j*testDim1)));
    }
  }
}

TEST_F(DimensionsTest, Check3DXMajor)
{
  //check a three dimensional matrix for proper x-major ordering
  std::vector<size_t> x;
  x.push_back(3);
  x.push_back(4);
  x.push_back(5);
  Dimensions d(x);
  size_t testDim1 = 3;
  size_t testDim2 = 4;
  size_t testDim3 = 5;
  for(size_t i = 0; i < testDim1; i++)
  {
    for(size_t j = 0; j < testDim2; j++)
    {
      for(size_t k = 0; k < testDim3; k++)
      {
        Coordinate testCoordinate;
        testCoordinate.push_back(i);
        testCoordinate.push_back(j);
        testCoordinate.push_back(k);

        ASSERT_EQ(i +
                  j*testDim1 +
                  k*testDim1*testDim2, d.getIndex(testCoordinate));

        ASSERT_EQ(vecToString(testCoordinate),
                  vecToString(d.getCoordinate(i +
                                              j*testDim1 +
                                              k*testDim1*testDim2)));
      }
    }
  }
}

TEST_F(DimensionsTest, AlternateConstructor)
{ 
  // alternate constructor
  std::vector<size_t> x;
  x.push_back(2);
  x.push_back(5);
  Dimensions d(x);
  ASSERT_TRUE(!d.isUnspecified());
  ASSERT_TRUE(!d.isDontcare());
  ASSERT_TRUE(d.isValid());
  
  ASSERT_EQ((unsigned int)2, d.getDimension(0));
  ASSERT_EQ((unsigned int)5, d.getDimension(1));
  ASSERT_ANY_THROW(d.getDimension(2));
  ASSERT_EQ((unsigned int)2, d.getDimensionCount());
}
