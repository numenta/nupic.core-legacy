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
 * Timer interface
 */

#ifndef NTA_TIMER2_HPP
#define NTA_TIMER2_HPP

#include <nupic/types/Types.hpp>
#include <string>

namespace nupic {

/**
 * @Responsibility
 * Simple stopwatch services
 *
 * @Description
 * A timer object is a stopwatch. You can start it, stop it, read the
 * elapsed time, and reset it. It is very convenient for performance
 * measurements.
 *
 * Uses the most precise and lowest overhead timer available on a given system.
 *
 */
class Timer {
public:
  /**
   * Create a stopwatch
   *
   * @param startme  If true, the timer is started when created
   */
  Timer(bool startme = false);

  /**
   * Start the stopwatch
   */
  void start();

  /**
   * Stop the stopwatch. When restarted, time will accumulate
   */
  void stop();

  /**
   * If stopped, return total elapsed time.
   * If started, return current elapsed time but don't stop the clock
   * return the value in seconds;
   */
  Real64 getElapsed() const;

  /**
   * Reset the stopwatch, setting accumulated time to zero.
   */
  void reset();

  /**Train
   * Return the number of time the stopwatch has been started.
   */
  UInt64 getStartCount() const;

  /**
   * Returns true is the stopwatch is currently running
   */
  bool isStarted() const;

  std::string toString() const;

private:
  // internally times are stored as ticks
  UInt64 prevElapsed_; // total time as of last stop() (in ticks)
  UInt64 start_;       // time that start() was called (in ticks)
  UInt64 nstarts_;     // number of times start() was called
  bool started_;       // true if was started

}; // class Timer

} // namespace nupic

#endif // NTA_TIMER2_HPP
