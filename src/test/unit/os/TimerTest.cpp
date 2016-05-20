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

/**
 * @file
 */

#define SLEEP_MICROSECONDS (100 * 1000)

#include <nupic/utils/Log.hpp>
#include <nupic/os/Timer.hpp>
#include <math.h> // fabs
#include <apr-1/apr_time.h>
#include <gtest/gtest.h>

using namespace nupic;

TEST(TimerTest, Basic)
{
// Tests are minimal because we have no way to run performance-sensitive tests in a controlled
// environment.

  Timer t1;
  Timer t2(/* startme= */ true);

  ASSERT_FALSE(t1.isStarted());
  ASSERT_EQ(t1.getElapsed(), 0.0);
  ASSERT_EQ(t1.getStartCount(), 0);
  EXPECT_STREQ("[Elapsed: 0 Starts: 0]", t1.toString().c_str());

  apr_sleep(SLEEP_MICROSECONDS);

  ASSERT_TRUE(t2.isStarted());
  ASSERT_EQ(t2.getStartCount(), 1);
  ASSERT_GT(t2.getElapsed(), 0);
  Real64 t2elapsed = t2.getElapsed();

  t1.start();
  apr_sleep(SLEEP_MICROSECONDS);
  t1.stop();

  t2.stop();
  ASSERT_EQ(t1.getStartCount(), 1);
  ASSERT_GT(t1.getElapsed(), 0);
  ASSERT_GT(t2.getElapsed(), t2elapsed);
  ASSERT_GT(t2.getElapsed(), t1.getElapsed());

  t1.start();
  t1.stop();
  ASSERT_EQ(t1.getStartCount(), 2);
}

TEST(TimerTest, Drift)
{
// Test start/stop delay accumulation
  Timer t;
  const UInt EPOCHS = 1000000; // 1M
  const UInt EPSILON = 5; // tolerate 5us drift on 1M restarts
  for(UInt i=0; i<EPOCHS; i++){
    t.start();
    t.stop(); //immediately
  }
  ASSERT_LT(t.getElapsed(), EPSILON);
}
