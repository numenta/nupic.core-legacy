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
  std::vector<float> historical_values {};
  float total = 0;
  int window_size = 3;
  float new_average = 0;

  std::vector<float> expected_historical{3.0};
  std::tie(new_average, total) = MovingAverage::compute(historical_values, total, 3, window_size);
  ASSERT_EQ(new_average, 3.0);
  ASSERT_EQ(historical_values, expected_historical);
  ASSERT_FLOAT_EQ(total, 3.0);

  expected_historical = {3.0, 4.0};
  std::tie(new_average, total) = MovingAverage::compute(historical_values, total, 4, window_size);
  ASSERT_EQ(new_average, 3.5);
  ASSERT_EQ(historical_values, expected_historical);
  ASSERT_EQ(total, 7.0);

  expected_historical = {3.0, 4.0, 5.0};
  std::tie(new_average, total) = MovingAverage::compute(historical_values, total, 5, window_size);
  ASSERT_EQ(new_average, 4.0);
  ASSERT_EQ(historical_values, expected_historical);
  ASSERT_EQ(total, 12.0);

  expected_historical = {4.0, 5.0, 6.0};
  std::tie(new_average, total) = MovingAverage::compute(historical_values, total, 6, window_size);
  ASSERT_EQ(new_average, 5.0);
  ASSERT_EQ(historical_values, expected_historical);
  ASSERT_EQ(total, 15.0);
};


TEST(moving_average, instance)
{
  MovingAverage m{3};
  float new_average;
  std::vector<float> expected_window;

  expected_window = {3.0};
  new_average = m.next(3);
  ASSERT_EQ(new_average, 3.0);
  ASSERT_EQ(m.get_sliding_window(), expected_window);
  ASSERT_EQ(m.get_total(), 3.0);

  expected_window = {3.0, 4.0};
  new_average = m.next(4);
  ASSERT_EQ(new_average, 3.5);
  ASSERT_EQ(m.get_sliding_window(), expected_window);
  ASSERT_EQ(m.get_total(), 7.0);

  expected_window = {3.0, 4.0, 5.0};
  new_average = m.next(5);
  ASSERT_EQ(new_average, 4.0);
  ASSERT_EQ(m.get_sliding_window(), expected_window);
  ASSERT_EQ(m.get_total(), 12.0);

  expected_window = {4.0, 5.0, 6.0};
  new_average = m.next(6);
  ASSERT_EQ(new_average, 5.0);
  ASSERT_EQ(m.get_sliding_window(), expected_window);
  ASSERT_EQ(m.get_total(), 15.0);
};


TEST(moving_average, SlidingWindowInit)
{
  std::vector<float> existing_historical {3.0, 4.0, 5.0};
  MovingAverage m{3, existing_historical};
  ASSERT_EQ(m.get_sliding_window(), existing_historical);

  MovingAverage m2{3};
  std::vector<float> empty_vector;
  ASSERT_EQ(m2.get_sliding_window(), empty_vector);
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
