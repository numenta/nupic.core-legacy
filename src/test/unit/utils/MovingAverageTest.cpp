/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

#include "nupic/utils/MovingAverage.hpp"
#include "gtest/gtest.h"
#include <tuple>

using namespace nupic::util;

TEST(moving_average, instance)
{
  MovingAverage m{3};
  float newAverage;
  std::vector<float> expectedWindow;

  expectedWindow = {3.0};
  m.next(3);
  newAverage = m.getCurrentAvg();
  ASSERT_EQ(newAverage, 3.0);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 3.0);

  expectedWindow = {3.0, 4.0};
  m.next(4);
  newAverage = m.getCurrentAvg();
  ASSERT_EQ(newAverage, 3.5);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 7.0);

  expectedWindow = {3.0, 4.0, 5.0};
  m.next(5);
  newAverage = m.getCurrentAvg();
  ASSERT_EQ(newAverage, 4.0);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 12.0);

  expectedWindow = {4.0, 5.0, 6.0};
  m.next(6);
  newAverage = m.getCurrentAvg();
  ASSERT_EQ(newAverage, 5.0);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 15.0);
};


TEST(moving_average, SlidingWindowInit)
{
  std::vector<float> existingHistorical {3.0, 4.0, 5.0};
  MovingAverage m{3, existingHistorical};
  ASSERT_EQ(m.getSlidingWindow(), existingHistorical);

  MovingAverage m2{3};
  std::vector<float> emptyVector;
  ASSERT_EQ(m2.getSlidingWindow(), emptyVector);
}

TEST(moving_average, EqualsOperator)
{

    MovingAverage ma{3};
    MovingAverage maP{3};
    ASSERT_EQ(ma, maP);

    MovingAverage maN{10};
    //    ASSERT_NE(ma, maN);

    MovingAverage mb{2, std::vector<float> {3.0, 4.0, 5.0} };
    MovingAverage mbP{2, std::vector<float> {3.0, 4.0, 5.0} };
    ASSERT_EQ(mb, mbP);

    mbP.next(6);
    mb.next(6);
    ASSERT_EQ(mb, mbP);   
}
