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

#include <tuple>

#include "gtest/gtest.h"

#include "nupic/types/Types.hpp"
#include "nupic/utils/MovingAverage.hpp"

using namespace nupic;
using namespace nupic::util;


TEST(MovingAverage, Instance)
{
  MovingAverage m{3};
  Real32 newAverage;

  {
    std::vector<Real32> expectedWindow = {3.0};
    m.compute(3);
    newAverage = m.getCurrentAvg();
    ASSERT_EQ(newAverage, 3.0);
    ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
    ASSERT_EQ(m.getTotal(), 3.0);
  }

  {
    std::vector<Real32> expectedWindow = {3.0, 4.0};
    m.compute(4);
    newAverage = m.getCurrentAvg();
    ASSERT_EQ(newAverage, 3.5);
    ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
    ASSERT_EQ(m.getTotal(), 7.0);
  }

  {
    std::vector<Real32> expectedWindow = {3.0, 4.0, 5.0};
    m.compute(5);
    newAverage = m.getCurrentAvg();
    ASSERT_EQ(newAverage, 4.0);
    ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
    ASSERT_EQ(m.getTotal(), 12.0);
  }

  {
    std::vector<Real32> expectedWindow = {4.0, 5.0, 6.0};
    m.compute(6);
    newAverage = m.getCurrentAvg();
    ASSERT_EQ(newAverage, 5.0);
    ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
    ASSERT_EQ(m.getTotal(), 15.0);
  }
};


TEST(MovingAverage, SlidingWindowInit)
{
  std::vector<Real32> existingHistorical = {3.0, 4.0, 5.0};
  MovingAverage m{3, existingHistorical};
  ASSERT_EQ(m.getSlidingWindow(), existingHistorical);

  MovingAverage m2{3};
  std::vector<Real32> emptyVector;
  ASSERT_EQ(m2.getSlidingWindow(), emptyVector);
}


TEST(MovingAverage, EqualsOperator)
{
  MovingAverage ma{3};
  MovingAverage maP{3};
  ASSERT_EQ(ma, maP);

  MovingAverage maN{10};
  ASSERT_NE(ma, maN);

  MovingAverage mb{2, {3.0, 4.0, 5.0}};
  MovingAverage mbP{2, {3.0, 4.0, 5.0}};
  ASSERT_EQ(mb, mbP);

  mbP.compute(6);
  mb.compute(6);
  ASSERT_EQ(mb, mbP);
}
