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

TEST(moving_average, testMovingAverage)
{
  std::vector<float> historicalValues {};
  float total = 0;
  int windowSize = 3;
  float newAverage = 0;

  std::vector<float> expectedHistorical{3.0};
  std::tie(newAverage, total) = MovingAverage::compute(historicalValues, total, 3, windowSize);
  ASSERT_EQ(newAverage, 3.0);
  ASSERT_EQ(historicalValues, expectedHistorical);
  ASSERT_FLOAT_EQ(total, 3.0);

  expectedHistorical = {3.0, 4.0};
  std::tie(newAverage, total) = MovingAverage::compute(historicalValues, total, 4, windowSize);
  ASSERT_EQ(newAverage, 3.5);
  ASSERT_EQ(historicalValues, expectedHistorical);
  ASSERT_EQ(total, 7.0);

  expectedHistorical = {3.0, 4.0, 5.0};
  std::tie(newAverage, total) = MovingAverage::compute(historicalValues, total, 5, windowSize);
  ASSERT_EQ(newAverage, 4.0);
  ASSERT_EQ(historicalValues, expectedHistorical);
  ASSERT_EQ(total, 12.0);

  expectedHistorical = {4.0, 5.0, 6.0};
  std::tie(newAverage, total) = MovingAverage::compute(historicalValues, total, 6, windowSize);
  ASSERT_EQ(newAverage, 5.0);
  ASSERT_EQ(historicalValues, expectedHistorical);
  ASSERT_EQ(total, 15.0);
};


TEST(moving_average, instance)
{
  MovingAverage m{3};
  float newAverage;
  std::vector<float> expectedWindow;

  expectedWindow = {3.0};
  newAverage = m.next(3);
  ASSERT_EQ(newAverage, 3.0);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 3.0);

  expectedWindow = {3.0, 4.0};
  newAverage = m.next(4);
  ASSERT_EQ(newAverage, 3.5);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 7.0);

  expectedWindow = {3.0, 4.0, 5.0};
  newAverage = m.next(5);
  ASSERT_EQ(newAverage, 4.0);
  ASSERT_EQ(m.getSlidingWindow(), expectedWindow);
  ASSERT_EQ(m.getTotal(), 12.0);

  expectedWindow = {4.0, 5.0, 6.0};
  newAverage = m.next(6);
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

/*
  @unittest.skipUnless(
      capnp, "pycapnp is not installed, skipping serialization test.")
  def testMovingAverageReadWrite(self):
    ma = MovingAverage(windowSize=3)

    ma.next(3)
    ma.next(4.5)
    ma.next(5)

    proto1 = MovingAverageProto.new_message()
    ma.write(proto1)

    # Write the proto to a temp file and read it back into a new proto
    with tempfile.TemporaryFile() as f:
      proto1.write(f)
      f.seek(0)
      proto2 = MovingAverageProto.read(f)

    resurrectedMa = MovingAverage.read(proto2)

    newAverage = ma.next(6)
    self.assertEqual(newAverage, resurrectedMa.next(6))
    self.assertListEqual(ma.getSlidingWindow(),
                         resurrectedMa.getSlidingWindow())
    self.assertEqual(ma.total, resurrectedMa.total)
    self.assertTrue(ma, resurrectedMa) #using the __eq__ method


  def testSerialization(self):
    """serialization using pickle"""
    ma = MovingAverage(windowSize=3)

    ma.next(3)
    ma.next(4.5)
    ma.next(5)

    stored = pickle.dumps(ma)
    restored = pickle.loads(stored)
    self.assertEqual(restored, ma) 
    self.assertEqual(ma.next(6), restored.next(6))


  def testEquals(self):
    ma = MovingAverage(windowSize=3)
    maP = MovingAverage(windowSize=3)
    self.assertEqual(ma, maP)
    
    maN = MovingAverage(windowSize=10)
    self.assertNotEqual(ma, maN)

    ma = MovingAverage(windowSize=2, existingHistoricalValues=[3.0, 4.0, 5.0])
    maP = MovingAverage(windowSize=2, existingHistoricalValues=[3.0, 4.0, 5.0])
    self.assertEqual(ma, maP)
    maP.next(6)
    self.assertNotEqual(ma, maP)
    ma.next(6)
    self.assertEqual(ma, maP)
*/
