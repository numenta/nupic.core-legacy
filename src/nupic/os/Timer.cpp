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

// Define a couple of platform-specific helper functions

#if defined(NTA_OS_WINDOWS)

#include <windows.h>
static nupic::UInt64 ticksPerSec_ = 0;
static nupic::UInt64 initialTicks_ = 0;

// initTime is called by the constructor, so it will always
// have been called by the time we call getTicksPerSec or getCurrentTime
static inline void initTime()
{
  if (initialTicks_ == 0)
  {
    LARGE_INTEGER f;
    QueryPerformanceCounter(&f);
    initialTicks_ = (nupic::UInt64)(f.QuadPart);

    QueryPerformanceFrequency(&f);
    ticksPerSec_ = (nupic::UInt64)(f.QuadPart);
  }
}

static inline nupic::UInt64 getTicksPerSec()
{
  return ticksPerSec_;
}


static nupic::UInt64 getCurrentTime()
{
  LARGE_INTEGER v;
  QueryPerformanceCounter(&v);
  return (nupic::UInt64)(v.QuadPart) - initialTicks_;
}

#elif defined(NTA_OS_DARWIN)

// This include defines a UInt64 type that conflicts with the nupic::UInt64 type.
// Because of this, all UInt64 is explicitly qualified in the interface. 
#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>

// must be linked with -framework CoreServices

static uint64_t initialTicks_ = 0;
static nupic::UInt64 ticksPerSec_ = 0;

static inline void initTime()
{
  if (initialTicks_ == 0)
    initialTicks_ = mach_absolute_time();
  if (ticksPerSec_ == 0)
  {
    mach_timebase_info_data_t sTimebaseInfo;
    mach_timebase_info(&sTimebaseInfo);
    ticksPerSec_ = (nupic::UInt64)(1e9 * (uint64_t)sTimebaseInfo.denom /
        (uint64_t)sTimebaseInfo.numer);
  }
}

static inline nupic::UInt64 getCurrentTime()
{
  return (nupic::UInt64)(mach_absolute_time() - initialTicks_);
}

static inline nupic::UInt64 getTicksPerSec()
{
  return ticksPerSec_;
}


#else
// linux
#include <sys/time.h>

static nupic::UInt64 initialTicks_ = 0;

static inline void initTime()
{
  if (initialTicks_ == 0)
  {
    struct timeval t;
    ::gettimeofday(&t, nullptr);
    initialTicks_ = nupic::UInt64((t.tv_sec * 1e6) + t.tv_usec);
  }
}

static inline nupic::UInt64 getCurrentTime()
{
  struct timeval t;
  ::gettimeofday(&t, nullptr);
  nupic::UInt64 ticks = nupic::UInt64((t.tv_sec * 1e6) + t.tv_usec);
  return ticks - initialTicks_;
}



static inline nupic::UInt64 getTicksPerSec()
{
  return (nupic::UInt64)(1e6);
}

#endif

namespace nupic
{

  Timer::Timer(bool startme)  
  {
    initTime();
    reset();
    if (startme)
      start();
  }
  
  
  void Timer::start() 
  { 
    if (started_ == false) 
    {
      start_ = getCurrentTime();
      nstarts_++;
      started_ = true;
    }
  }
  
  /**
  * Stop the stopwatch. When restarted, time will accumulate
  */
  
  void Timer::stop() 
  {  // stop the stopwatch
    if (started_ == true) 
    {
      prevElapsed_ += (getCurrentTime() - start_);
      start_ = 0;
      started_ = false;
    }
  }
  
  Real64 Timer::getElapsed() const
  {   
    nupic::UInt64 elapsed = prevElapsed_;
    if (started_) 
    {
      elapsed += (getCurrentTime() - start_);
    }   
  
    return (Real64)(elapsed) / (Real64)getTicksPerSec();
  }
  
  void Timer::reset() 
  {
    prevElapsed_ = 0;
    start_ = 0;
    nstarts_ = 0;
    started_ = false;
  }
  
  UInt64 Timer::getStartCount() const
  { 
    return nstarts_; 
  }
  
  bool Timer::isStarted() const
  { 
    return started_; 
  }
  
  
  std::string Timer::toString() const
  {
    std::stringstream ss;
    ss << "[Elapsed: " << getElapsed() << " Starts: " << getStartCount();
    if (isStarted())
      ss << " (running)";
    ss << "]";
    return ss.str();
  }

}  // namespace nupic
