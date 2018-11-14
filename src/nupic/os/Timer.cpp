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
 * Generic OS Implementations for the OS class
 */

#include <nupic/os/Timer.hpp>
#include <sstream>

namespace nupic {

Timer::Timer(bool startme) {
  nstarts_ = 0;
  start_ = 0;
  prevElapsed_ = 0;

  reset();
  if (startme)
    start();
}

void Timer::start() {
	if (started_ == false) {
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
	if (started_ == true)
	{
	    auto diff = my_clock::now() - start_time_;
	    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

	    prevElapsed_ += milliseconds.count();

	    //start_ = 0;
	    started_ = false;
	}
}

// in seconds, msec resolution
Real64 Timer::getElapsed() const {
	nupic::UInt64 elapsed = prevElapsed_;
	if (started_)
	{
	    auto diff = my_clock::now() - start_time_;
	    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(diff);

	    elapsed += milliseconds.count();
	}

	return static_cast<Real64>(elapsed) / 1000.0;
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

} // namespace nupic
