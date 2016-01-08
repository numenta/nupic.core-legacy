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

#include <nupic/types/Fraction.hpp>
#include <sstream>
#include <limits>
#include <nupic/types/Types.hpp>
#include <gtest/gtest.h>

using namespace nupic;

TEST(FractionTest, All)
{
  //create fractions
  Fraction(1);
  Fraction(0);
  Fraction(-1);
  Fraction(1, 2);
  Fraction(2, 1);
  Fraction(-1, 2);
  Fraction(-2, 1);
  Fraction(2, 3);
  Fraction(3, 2);
  // Current overflow cutoff of 10 million
  const static int tooLarge = 20000000;

  ASSERT_ANY_THROW(Fraction(3, 0));
  ASSERT_ANY_THROW(Fraction(-3, 0));
  ASSERT_ANY_THROW(Fraction(0, 0));
  ASSERT_ANY_THROW(Fraction(tooLarge, 0));
  ASSERT_ANY_THROW(Fraction(tooLarge, 1));
  ASSERT_ANY_THROW(Fraction(0, tooLarge));
  // There is some strange interaction with the SHOULDFAIL macro here. 
  // Without this syntax, the compiler thinks we're declaring a new variable
  // tooLarge of type Fraction (which masks the old tooLarge). 
  ASSERT_ANY_THROW(Fraction x(tooLarge));
  ASSERT_ANY_THROW(Fraction(20000000));
  ASSERT_ANY_THROW(Fraction(-tooLarge));
  ASSERT_ANY_THROW(Fraction(-tooLarge, 0));
  ASSERT_ANY_THROW(Fraction(0, -tooLarge));
  ASSERT_ANY_THROW(Fraction(-tooLarge));
  ASSERT_ANY_THROW(new Fraction(std::numeric_limits<int>::max()));
  ASSERT_ANY_THROW(new Fraction(std::numeric_limits<int>::min()));
    
  //Test isNaturalNumber() (natural numbers must be nonnegative)
  ASSERT_TRUE(Fraction(1).isNaturalNumber());
  ASSERT_TRUE(Fraction(0).isNaturalNumber());
  ASSERT_TRUE(!Fraction(-1).isNaturalNumber());
  ASSERT_TRUE(!Fraction(1, 2).isNaturalNumber());
  ASSERT_TRUE(Fraction(2, 1).isNaturalNumber());
  ASSERT_TRUE(!Fraction(-1, 2).isNaturalNumber());
  ASSERT_TRUE(!Fraction(-2, 1).isNaturalNumber());
  ASSERT_TRUE(!Fraction(3, 2).isNaturalNumber());
  ASSERT_TRUE(!Fraction(-3, 2).isNaturalNumber());

  //Test getNumerator()
  ASSERT_EQ(2, Fraction(2, 1).getNumerator());
  ASSERT_EQ(0, Fraction(0, 1).getNumerator());
  ASSERT_EQ(-2, Fraction(-2, 1).getNumerator());
  ASSERT_EQ(2, Fraction(2, -2).getNumerator());
  ASSERT_EQ(0, Fraction(0, -2).getNumerator());
  ASSERT_EQ(-2, Fraction(-2, -2).getNumerator());

  //Test getDenominator()
  ASSERT_EQ(1, Fraction(0).getDenominator());
  ASSERT_EQ(1, Fraction(2).getDenominator());
  ASSERT_EQ(-2, Fraction(0, -2).getDenominator());
  ASSERT_EQ(-2, Fraction(-2, -2).getDenominator());
  
  //Test setNumerator()
  Fraction b(1);
  b.setNumerator(0);
  ASSERT_EQ(0, b.getNumerator());
  b = Fraction(2, 3);
  b.setNumerator(-2);
  ASSERT_EQ(-2, b.getNumerator());
  b = Fraction(2, -3);
  b.setNumerator(2);
  ASSERT_EQ(2, b.getNumerator());

  //Test setDenominator()
  ASSERT_ANY_THROW(Fraction(1).setDenominator(0));
  b = Fraction(1);
  b.setDenominator(2);
  ASSERT_EQ(2, b.getDenominator());
  b = Fraction(-2, 3);
  b.setDenominator(5);
  ASSERT_EQ(5, b.getDenominator());

  //Test setFraction()
  ASSERT_ANY_THROW(Fraction(1).setFraction(1, 0));
  ASSERT_ANY_THROW(Fraction(-2).setFraction(-3, 0));
  b = Fraction(2);
  b.setFraction(1, 1);
  ASSERT_TRUE(Fraction(1) == b);
  b = Fraction(1);
  b.setFraction(-1, 2);
  ASSERT_TRUE(Fraction(-1, 2) == b);
  b = Fraction(0);
  b.setFraction(-6, 4);
  ASSERT_TRUE(Fraction(-6, 4) == b);
  
  //Test computeGCD()
  ASSERT_EQ((UInt32)5, Fraction::computeGCD(5, 10));
  ASSERT_EQ((UInt32)1, Fraction::computeGCD(1, 1));
  ASSERT_EQ((UInt32)1, Fraction::computeGCD(0, 1));
  ASSERT_EQ((UInt32)3, Fraction::computeGCD(3, 0));
  ASSERT_EQ((UInt32)1, Fraction::computeGCD(1, 0));
  ASSERT_EQ((UInt32)1, Fraction::computeGCD(1, -1));

  //Test computeLCM
  ASSERT_EQ((UInt32)10, Fraction::computeLCM(5, 2));
  ASSERT_EQ((UInt32)1, Fraction::computeLCM(1, 1));
  ASSERT_EQ((UInt32)0, Fraction::computeLCM(0, 0));
  ASSERT_EQ((UInt32)0, Fraction::computeLCM(0, -1));
  ASSERT_EQ((UInt32)0 , Fraction::computeLCM(-1, 2));
  
  //Test reduce()
  Fraction a = Fraction(1);
  a.reduce();
  ASSERT_EQ(1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(2, 2);
  a.reduce();
  ASSERT_EQ(1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(-1);
  a.reduce();
  ASSERT_EQ(-1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(-1, -1);
  a.reduce();
  ASSERT_EQ(1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(2, -2);
  a.reduce();
  ASSERT_EQ(-1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(-2, 2);
  a.reduce();
  ASSERT_EQ(-1, a.getNumerator());
  ASSERT_EQ(1, a.getDenominator());
  a = Fraction(20, 6);
  a.reduce();
  ASSERT_EQ(10, a.getNumerator());
  ASSERT_EQ(3, a.getDenominator());
  a = Fraction(-2, 6);
  a.reduce();
  ASSERT_EQ(-1, a.getNumerator());
  ASSERT_EQ(3, a.getDenominator());

  //Test *
  Fraction one = Fraction(1);
  Fraction zero = Fraction(0);
  Fraction neg_one = Fraction(-1);
  ASSERT_TRUE(one == one*one);
  ASSERT_TRUE(one == neg_one*neg_one);
  ASSERT_TRUE(zero == zero*one);
  ASSERT_TRUE(zero == zero*zero);
  ASSERT_TRUE(zero == zero*neg_one);
  ASSERT_TRUE(neg_one == one*neg_one);
  ASSERT_TRUE(neg_one == neg_one*one);
  ASSERT_TRUE(Fraction(10) == one*Fraction(20, 2));

  ASSERT_TRUE(one == one*1);
  ASSERT_TRUE(one == one*1);
  ASSERT_TRUE(zero == zero*1);
  ASSERT_TRUE(zero == zero*1);
  ASSERT_TRUE(zero == zero*-1);
  ASSERT_TRUE(zero == zero*-1);
  ASSERT_TRUE(-1 == one*-1);
  ASSERT_TRUE(-1 == neg_one*1);
  ASSERT_TRUE(Fraction(10) == one*10);
  ASSERT_TRUE(Fraction(10) == neg_one*-10);
  
  //Test /
  ASSERT_TRUE(one == one/one);
  ASSERT_TRUE(zero == zero/one);
  ASSERT_TRUE(zero == zero/neg_one);
  ASSERT_TRUE(Fraction(-0) == zero/neg_one);
  ASSERT_ANY_THROW(one/zero);
  ASSERT_TRUE(Fraction(3, 2) == Fraction(3)/Fraction(2));
  ASSERT_TRUE(Fraction(2, -3) == Fraction(2)/Fraction(-3));

  //Test -
  ASSERT_TRUE(zero == one - one);
  ASSERT_TRUE(neg_one == zero - one);
  ASSERT_TRUE(one == zero - neg_one);
  ASSERT_TRUE(zero == neg_one - neg_one);
  ASSERT_TRUE(Fraction(1, 2) == Fraction(3, 2) - one);
  ASSERT_TRUE(Fraction(-1, 2) == Fraction(-3, 2) - neg_one);

  //Test +
  ASSERT_TRUE(zero == neg_one + one);
  ASSERT_TRUE(one == zero + one);
  ASSERT_TRUE(one == (neg_one + one) + one);
  ASSERT_TRUE(one == one + zero);
  ASSERT_TRUE(Fraction(-2) == neg_one + neg_one);
  ASSERT_TRUE(Fraction(1, 2) == Fraction(-1, 2) + one);
  ASSERT_TRUE(Fraction(-3, 2) == neg_one + Fraction(-1, 2));

  //Test %
  ASSERT_TRUE(Fraction(1, 2) == Fraction(3, 2) % one);
  ASSERT_TRUE(Fraction(-1, 2) == Fraction(-1, 2) % one);
  ASSERT_TRUE(Fraction(3, 2) == Fraction(7, 2) % Fraction(2));
  ASSERT_TRUE(Fraction(-1, 2) == Fraction(-3, 2) % one);
  ASSERT_TRUE(Fraction(-1, 2) == Fraction(-3, 2) % neg_one);
  ASSERT_TRUE(Fraction(1, 2) == Fraction(3, 2) % neg_one);
  ASSERT_ANY_THROW(Fraction(1, 2) % Fraction(0));
  ASSERT_ANY_THROW(Fraction(-3,2) % Fraction(0, -2));

  //Test <
  ASSERT_TRUE(zero < one);
  ASSERT_TRUE(!(one < zero));
  ASSERT_TRUE(!(zero < zero));
  ASSERT_TRUE(!(one < one));
  ASSERT_TRUE(Fraction(1, 2) < one);
  ASSERT_TRUE(Fraction(-3, 2) < Fraction(1, -2));
  ASSERT_TRUE(Fraction(-1, 2) < Fraction(3, 2));

  //Test >
  ASSERT_TRUE(one > zero);
  ASSERT_TRUE(!(zero > zero));
  ASSERT_TRUE(!(one > one));
  ASSERT_TRUE(!(zero > one));
  ASSERT_TRUE(one > Fraction(1, 2));
  ASSERT_TRUE(Fraction(1, -2) > Fraction(-3, 2));
  ASSERT_TRUE(Fraction(1, 2) > Fraction(-3, 2));

  //Test <=
  ASSERT_TRUE(zero <= one);
  ASSERT_TRUE(!(one <= zero));
  ASSERT_TRUE(Fraction(1, 2) <= one);
  ASSERT_TRUE(Fraction(-3, 2) <= Fraction(1, -2));
  ASSERT_TRUE(Fraction(-1, 2) <= Fraction(3, 2));
  ASSERT_TRUE(zero <= zero);
  ASSERT_TRUE(one <= one);
  ASSERT_TRUE(neg_one <= neg_one);
  ASSERT_TRUE(Fraction(-7, 4) <= Fraction(14, -8));

  //Test >=
  ASSERT_TRUE(one >= zero);
  ASSERT_TRUE(!(zero >= one));
  ASSERT_TRUE(one >= Fraction(1, 2));
  ASSERT_TRUE(Fraction(1, -2) >= Fraction(-3, 2));
  ASSERT_TRUE(Fraction(1, 2) >= Fraction(-3, 2));
  ASSERT_TRUE(zero >= zero);
  ASSERT_TRUE(one >= one);
  ASSERT_TRUE(neg_one >= neg_one);
  ASSERT_TRUE(Fraction(-7, 4) >= Fraction(14, -8));

  //Test ==
  ASSERT_TRUE(one == one);
  ASSERT_TRUE(zero == zero);
  ASSERT_TRUE(!(one == zero));
  ASSERT_TRUE(Fraction(1, 2) == Fraction(2, 4));
  ASSERT_TRUE(Fraction(-1, 2) == Fraction(2, -4));
  ASSERT_TRUE(Fraction(0, 1) == Fraction(0, -1));
  ASSERT_TRUE(Fraction(0, 1) == Fraction(0, 2));

  //Test <<
  std::stringstream ss;
  ss << Fraction(3, 4);
  EXPECT_STREQ("3/4", ss.str().c_str());
  ss.str("");
  ss << Fraction(-2, 4);
  EXPECT_STREQ("-1/2", ss.str().c_str());
  ss.str("");
  ss << Fraction(0, 1);
  EXPECT_STREQ("0", ss.str().c_str());
  ss.str("");
  ss << Fraction(0, -1);
  EXPECT_STREQ("0", ss.str().c_str());
  ss.str("");
  ss << Fraction(1, -2);
  EXPECT_STREQ("-1/2", ss.str().c_str());
  ss.str("");
  ss << Fraction(3, 1);
  EXPECT_STREQ("3", ss.str().c_str());
  ss.str("");
  ss << Fraction(-3, 1);
  EXPECT_STREQ("-3", ss.str().c_str());
  ss.str("");
  ss << Fraction(6, 2);
  EXPECT_STREQ("3", ss.str().c_str());
  ss.str("");
  ss << Fraction(6, -2);
  EXPECT_STREQ("-3", ss.str().c_str());
  ss.str("");
  ss << Fraction(-1, -1);
  EXPECT_STREQ("1", ss.str().c_str());
  ss.str("");
  ss << Fraction(-2, -2);
  EXPECT_STREQ("1", ss.str().c_str());
  ss.str("");

  //Test fromDouble()
  ASSERT_TRUE(one == Fraction::fromDouble(1.0));
  ASSERT_TRUE(zero == Fraction::fromDouble(0.0));
  ASSERT_TRUE(Fraction(1, 2) == Fraction::fromDouble(0.5));
  ASSERT_TRUE(Fraction(-1, 2) == Fraction::fromDouble(-0.5));
  ASSERT_TRUE(Fraction(333, 1000) == Fraction::fromDouble(.333));
  ASSERT_TRUE(Fraction(1, 3) == Fraction::fromDouble(.3333333));
  ASSERT_TRUE(Fraction(1, -3) == Fraction::fromDouble(-.33333333));
  ASSERT_ANY_THROW(Fraction::fromDouble((double)(tooLarge)));
  ASSERT_ANY_THROW(Fraction::fromDouble(1.0/(double)(tooLarge)));
  ASSERT_ANY_THROW(Fraction::fromDouble(-(double)tooLarge));
  ASSERT_ANY_THROW(Fraction::fromDouble(-1.0/(double)(tooLarge)));
  ASSERT_ANY_THROW(Fraction::fromDouble(std::numeric_limits<double>::max()));
  ASSERT_ANY_THROW(Fraction::fromDouble(std::numeric_limits<double>::min()));
  ASSERT_ANY_THROW(Fraction::fromDouble(-std::numeric_limits<double>::max()));
  ASSERT_ANY_THROW(Fraction::fromDouble(-std::numeric_limits<double>::min()));

  //Test toDouble()
  ASSERT_EQ(0.0, Fraction(0).toDouble());
  ASSERT_EQ(0.0, Fraction(-0).toDouble());
  ASSERT_EQ(0.0, Fraction(0, 1).toDouble());
  ASSERT_EQ(0.5, Fraction(1, 2).toDouble());
  ASSERT_EQ(-0.5, Fraction(-1, 2).toDouble());
  ASSERT_EQ(-0.5, Fraction(1, -2).toDouble());
  ASSERT_EQ(0.5, Fraction(-1, -2).toDouble());

}
