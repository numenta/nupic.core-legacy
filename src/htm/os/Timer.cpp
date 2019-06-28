/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file
 * Generic OS Implementations for the OS class
 */

#include <htm/os/Timer.hpp>

#include <htm/utils/Random.hpp>
#include <htm/utils/Log.hpp>
#include <algorithm> //max
#include <cmath> //for sins
#include <sstream>
#include <vector>

namespace htm {

Timer::Timer(bool startme) {
  nstarts_ = 0;
  start_ = 0;
  prevElapsed_ = 0;

  reset();
  if (startme)
    start();
}


void Timer::start() {
	if (!started_) {
	  start_time_ = my_clock::now();
	  nstarts_++;
	  started_ = true;
	}
}


/**
* Stop the stopwatch. When restarted, time will accumulate
*/
void Timer::stop() {
    // stop the stopwatch
	if (started_)
	{
	    const auto diff = my_clock::now() - start_time_;
	    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

	    prevElapsed_ += nanoseconds.count();

	    started_ = false;
	}
}


/* In seconds, msec resolution */
Real64 Timer::getElapsed() const {
	htm::UInt64 elapsed = prevElapsed_;
	if (started_)
	{
	    const auto diff = my_clock::now() - start_time_;
	    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);

	    elapsed += nanoseconds.count();
	}

	return static_cast<Real64>(elapsed) / TO_SECONDS;
}


void Timer::reset() {
	prevElapsed_ = 0;
	start_ = 0;
	nstarts_ = 0;
	started_ = false;
}

UInt64 Timer::getStartCount() const { return nstarts_; }

bool Timer::isStarted() const { return started_; }

std::string Timer::toString() const {
  std::stringstream ss;
  ss << "[Elapsed: " << getElapsed() << " Starts: " << getStartCount();
  if (isStarted())
    ss << " (running)";
  ss << "]";
  return ss.str();
}


/**
 * Estimate speed (CPU & load) of the current system.
 * Tests must perform relative to this value
 */
float Timer::getSpeed() {
  if (SPEED == -1) {

    // This code just wastes CPU time to estimate speed.
    Timer t(true);

    Random rng(42);
    // Make 10 million 4-byte Reals.  Each synapse take approx 30-40 bytes to
    // represent, so this is enough memory for 1 million synapses.
    std::vector<Real> data(10000000);
    for( Size i = 0; i < data.size(); i++ ) {
      data[i] = (Real)rng.getUInt32(80085);
      auto t  = data[i];
      data[i] = data[data.size()-i-1];
      data[data.size()-i-1] = t;
    }
    // Hurt the cache with random accesses.
    rng.shuffle(begin(data), end(data));
    // Test floating-point arithmatic.
    std::vector<Real> sins;
    for (auto d : data) {
      sins.push_back( sin(d) / cos(d) );
    }
    data = rng.sample<Real>(sins, 666);
    NTA_CHECK(data.size() == 666);
    t.stop();
    SPEED = std::max(1.0, t.getElapsed());
    NTA_INFO << "Timer::getSpeed() -> " << SPEED << " seconds.";
  }
  return (float)SPEED;
}

} // namespace htm
