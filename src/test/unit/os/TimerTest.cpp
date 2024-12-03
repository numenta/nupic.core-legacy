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

/**
 * @file
 */

#define SLEEP_MICROSECONDS (100 * 1000)

#include <apr-1/apr_time.h>
#include <gtest/gtest.h>
#include <math.h> // fabs
#include <nupic/os/Timer.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic;

TEST(TimerTest, Basic) {
  // Tests are minimal because we have no way to run performance-sensitive tests
  // in a controlled environment.

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

TEST(TimerTest, Drift) {
  // Test start/stop delay accumulation
  Timer t;
  const UInt EPOCHS = 1000000; // 1M
  const UInt EPSILON = 5;      // tolerate 5us drift on 1M restarts
  for (UInt i = 0; i < EPOCHS; i++) {
    t.start();
    t.stop(); // immediately
  }
  ASSERT_LT(t.getElapsed(), EPSILON);
}
