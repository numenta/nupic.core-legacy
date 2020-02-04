/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, Numenta, Inc.
 *               Jan 2020, David Keeney,   dkeeney@gmail.com
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
 * Implementation of the DateEncoder
 *  ported from htm/encoders/date.py
 */

#include <memory> // make_shared()
#include <time.h> // localtime(), struct tm
#include <iostream> // cerr

#include <htm/encoders/DateEncoder.hpp>
#include <htm/encoders/ScalarEncoder.hpp>
#include <htm/os/Path.hpp>  // trim(), split()

#define VERBOSE   if (args_.verbose) std::cerr << "[          ] "

namespace htm {

enum bucketType {SEASON=0, DAYOFWEEK, WEEKEND, CUSTOM, HOLIDAY, TIMEOFDAY};


DateEncoder::DateEncoder(const DateEncoderParameters &parameters) { initialize(parameters); }

void DateEncoder::initialize(const DateEncoderParameters &parameters) {
  args_ = parameters;

  // Check parameters
  size_t size = 0;

  // Season Attribute 
  if (args_.season_width != 0) {
    ScalarEncoderParameters p;
    // Ignore leapyear differences -- assume 366 days in a year
    // Radius = 91.5 days = length of default season
    // Value is number of days since beginning of year (0 - 366)
    p.minimum = 0.0;
    p.maximum = 366.0;
    p.periodic = true;
    p.activeBits = args_.season_width;
    p.radius = args_.season_radius;
    seasonEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create Season Encoder: " 
            << (p.maximum - p.minimum) / seasonEncoder_->parameters.radius << " buckets, "
            << seasonEncoder_->parameters.activeBits << " bits per bucket, width " 
            << seasonEncoder_->size << std::endl;
    size += seasonEncoder_->size;

    bucketMap_[SEASON] = buckets_.size();
    buckets_.push_back(0.0);
  }

  // Day-of-week attribute
  if (args_.dayOfWeek_width != 0) {
    ScalarEncoderParameters p;
    // Value is day of week (floating point)
    // Radius is 1 day
    p.minimum = 0.0;
    p.maximum = 7.0;
    p.periodic = true;
    p.activeBits = args_.dayOfWeek_width;
    p.radius = args_.dayOfWeek_radius;
    dayOfWeekEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create DayOfWeek Encoder: " 
            << (p.maximum - p.minimum) / dayOfWeekEncoder_->parameters.radius << " categories, " 
            << dayOfWeekEncoder_->parameters.activeBits << " bits per bucket, width " 
            << dayOfWeekEncoder_->size << std::endl;
    size += dayOfWeekEncoder_->size;

    bucketMap_[DAYOFWEEK] = buckets_.size();
    buckets_.push_back(0.0);
  }

  // Weekend attribute
  if (args_.weekend_width != 0) {
    ScalarEncoderParameters p;
    // Binary value.
    p.minimum = 0.0;
    p.maximum = 1.0;
    p.category = true;
    p.activeBits = args_.weekend_width;
    weekendEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create Weekend Encoder: " 
            << (p.maximum - p.minimum) / weekendEncoder_->parameters.radius << " categories, " 
            << weekendEncoder_->parameters.activeBits << " bits per bucket, width " 
            << weekendEncoder_->size << std::endl;
    size += weekendEncoder_->size;

    bucketMap_[WEEKEND] = buckets_.size();
    buckets_.push_back(0.0);
  }

  // Custom Days attribute
  if (args_.custom_width != 0) {
    // We have a vector of strings, each string can be a day name or a list of day names for a category.
    // NOTE: Unlike holidays, this does not ramp up on day before or down on day after.
    NTA_CHECK(args_.custom_days.size() > 0) << "DateEncoder: customDays parameter; Please provide a list of strings "
                                               "containing days to define custom days categorys.";
    std::map<std::string, int> daymap = {{"sun", 0}, {"mon", 1}, {"tue", 2}, {"wed", 3},
                                         {"thu", 4}, {"fri", 5}, {"sat", 6}};

    for (size_t i = 0; i < args_.custom_days.size(); i++) {
      std::string daysToParse = args_.custom_days[i];
      std::transform(daysToParse.begin(), daysToParse.end(), daysToParse.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::vector<std::string> cust = split(daysToParse, ',');
      for (std::string day : cust) {
        day = trim(day);
        NTA_CHECK(day.length() >= 3) << "DateEncoder: custom; parse error near " << day;
        auto it = daymap.find(day.substr(0, 3));
        NTA_CHECK(it != daymap.end()) << "DayEncoder: custom; parse error near " << day;
        customDays_.insert(it->second);
      }
    }
    ScalarEncoderParameters p;
    p.activeBits = args_.custom_width;
    p.minimum = 0.0;
    p.maximum = 1.0;
    p.category = true;
    customDaysEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create customDays Encoder: boolean, On or Off, " << customDaysEncoder_->parameters.activeBits
            << " bits per bucket, width " << customDaysEncoder_->size << std::endl;
    size += customDaysEncoder_->size;

    bucketMap_[CUSTOM] = buckets_.size();
    buckets_.push_back(0.0);
  }

  // Holiday attribute
  if (args_.holiday_width != 0) {
    for (std::vector<int> day : args_.holiday_dates) {
      NTA_CHECK(day.size() == 2 || day.size() == 3) << "DateEncoder: holiday, expecting {mon,day} or {year,mon,day}.";
    }

    ScalarEncoderParameters p;
    // A "continuous" binary value. = 1 on the holiday itself and smooth ramp
    // 0->1 on the day before the holiday and 1->0 on the day after the holiday.
    p.minimum = 0.0;
    p.maximum = 2.0;
    p.radius = 1.0;
    p.periodic = true;
    p.activeBits = args_.holiday_width;
    holidayEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create holiday Encoder: " 
            << (p.maximum - p.minimum) / holidayEncoder_->parameters.radius << " buckets, "
            << holidayEncoder_->parameters.activeBits << " bits per bucket, width " 
            << holidayEncoder_->size << std::endl;
    size += holidayEncoder_->size;

    bucketMap_[HOLIDAY] = buckets_.size();
    buckets_.push_back(0.0);
  }

  // Time-of-day attribute
  if (args_.timeOfDay_width != 0) {
    // Value is time of day in hours
    // Radius = 4 hours, e.g. morning, afternoon, evening, early night,  late night, etc.
    //  size should be (max-min)/radius buckets X width
    //  bits should wrap if beyond maxinum.
    ScalarEncoderParameters p;
    p.minimum = 0.0;
    p.maximum = 24.0;
    p.periodic = true;
    p.activeBits = args_.timeOfDay_width;
    p.radius = args_.timeOfDay_radius;
    timeOfDayEncoder_ = std::make_shared<ScalarEncoder>(p);
    VERBOSE << "  create TimeOfDay Encoder: " 
            << (int)ceil((p.maximum - p.minimum) / timeOfDayEncoder_->parameters.radius) << " buckets, "
            << timeOfDayEncoder_->parameters.activeBits << " bits per bucket, width " 
            << timeOfDayEncoder_->size << std::endl;
    size += timeOfDayEncoder_->size;

    bucketMap_[TIMEOFDAY] = buckets_.size();
    buckets_.push_back(0.0);
  }


  NTA_CHECK(size > 0u) << "DateEncoder: No parameters were provided.";
  BaseEncoder::initialize({static_cast<UInt32>(size)});
}


///////////////////////////////////////////////


// from Unix time
void DateEncoder::encode(std::time_t input, SDR &output) {
  if (input == 0) {
    // If no time is given (is 0), use the current time.
    input = time(0);
  }
  struct std::tm timeinfo = *std::localtime(&input);
  encode(timeinfo, output);
}

 // from python datetime
void DateEncoder::encode(std::chrono::system_clock::time_point time_point, SDR &output) { 
  std::time_t input = std::chrono::system_clock::to_time_t(time_point);
  struct std::tm timeinfo = *std::localtime(&input);
  encode(timeinfo, output);
}

/**
 * encode time from struct tm 
 */
void DateEncoder::encode(struct std::tm timeinfo, SDR &output) {
  // -------------------------------------------------------------------------
  // Encode each sub-field
  std::vector<const SDR *> sdrs;
  SDR season_output;
  SDR dayOfWeek_output;
  SDR weekend_output;
  SDR customDay_output;
  SDR holiday_output;
  SDR timeOfDay_output;
  
   VERBOSE << "DateEncoder for " 
           <<  std::string(asctime(&timeinfo)).substr(0, 24) 
           << ((timeinfo.tm_isdst)?" dst":"") 
           << std::endl;
  
  if (seasonEncoder_) {
    // Number the days into the year starting at 0 for Jan 1.
    Real64 dayOfYear = static_cast<Real64>(timeinfo.tm_yday);
    season_output = SDR(seasonEncoder_->dimensions);
    seasonEncoder_->encode(dayOfYear, season_output);
    buckets_[bucketMap_[SEASON]] = std::floor(dayOfYear/seasonEncoder_->parameters.radius);
    VERBOSE << "  season: " << dayOfYear << " ==> " << season_output;
    sdrs.push_back(&season_output);
  }
  if (dayOfWeekEncoder_) {
    // shift tm_wday so monday is 0.
    Real64 dayOfWeek = static_cast<Real64>((timeinfo.tm_wday + 6) % 7);
    dayOfWeek_output = SDR(dayOfWeekEncoder_->dimensions);
    dayOfWeekEncoder_->encode(dayOfWeek, dayOfWeek_output);
    buckets_[bucketMap_[DAYOFWEEK]] = dayOfWeek - std::fmod(dayOfWeek, dayOfWeekEncoder_->parameters.radius);
    VERBOSE << "  dayOfWeek: " << dayOfWeek << " ==> " << dayOfWeek_output;
    sdrs.push_back(&dayOfWeek_output);
  }
  if (weekendEncoder_) {
    // Weekend is defined as: friday(5) evenng(after 6pm), saturday(6), and sunday(0)
    Real64 val;
    if (timeinfo.tm_wday == 0 or timeinfo.tm_wday == 6 or (timeinfo.tm_wday == 5 and timeinfo.tm_hour > 18)) {
      val = 1.0;
    } else {
      val = 0.0;
    }
    weekend_output = SDR(weekendEncoder_->dimensions);
    weekendEncoder_->encode(val, weekend_output);
    buckets_[bucketMap_[WEEKEND]] = val;
    VERBOSE << "  weekend: " << val << " ==> " << weekend_output;
    sdrs.push_back(&weekend_output);
  }

  if (customDaysEncoder_) {
    Real64 customDay = 0.0;
    if (customDays_.find(timeinfo.tm_wday) != customDays_.end()) {
        customDay = 1.0;
    }
    customDay_output = SDR(customDaysEncoder_->dimensions);
    customDaysEncoder_->encode(customDay, customDay_output);
    buckets_[bucketMap_[CUSTOM]] = customDay;
    VERBOSE << "  custom Day: " << customDay << " ==> " << customDay_output;
    sdrs.push_back(&customDay_output);
  }

  if (holidayEncoder_) {
    // A "continuous" binary value. = 1 on the holiday itself and smooth ramp
    //  0->1 on the day before the holiday and 1->0 on the day after the holiday.
    // holidays is a list of holidays that occur on a fixed date every year
    Real64 val = 0.0;
    double SECONDS_PER_DAY = 86400.0;
    time_t input = std::mktime(&timeinfo);
    for (auto h : args_.holiday_dates) {
      std::time_t hdate;
      if (h.size() == 3) {
        hdate = mktime(h[0], h[1], h[2]);
      } else {
        hdate = mktime(timeinfo.tm_year + 1900, h[0], h[1]);
      }
      if (input > hdate) {
        // start of holiday is in the past.
        std::time_t diff = input - hdate;
        if (diff < SECONDS_PER_DAY) {
          // return 1 on the holiday itself
          val = 1.0;
          break;
        } else if (diff < SECONDS_PER_DAY * 2.0) {
          // Next day, ramp smoothly from 1 -> 0
          val = 1.0 + ((diff - SECONDS_PER_DAY) / SECONDS_PER_DAY);
          break;
        }
      } else {
        // start of holiday is in the future.
        std::time_t diff = hdate - input;
        if (diff < SECONDS_PER_DAY) {
          // holiday starts tomarrow
          // ramp smoothly from 0 -> 1 on the previous day
          val = 1.0 - diff / SECONDS_PER_DAY;
          break;
        }
      }
    }
    holiday_output = SDR(holidayEncoder_->dimensions);
    holidayEncoder_->encode(val, holiday_output);
    buckets_[bucketMap_[HOLIDAY]] = std::floor(val);
    VERBOSE << "  holiday: " << val << " ==> " << holiday_output;
    sdrs.push_back(&holiday_output);
  }
  if (timeOfDayEncoder_) {
    Real64 timeOfDay = timeinfo.tm_hour + timeinfo.tm_min / 60.0f + timeinfo.tm_sec / (60.0 * 60.0);
    timeOfDay_output = SDR(timeOfDayEncoder_->dimensions);
    timeOfDayEncoder_->encode(timeOfDay, timeOfDay_output);
    buckets_[bucketMap_[TIMEOFDAY]] = timeOfDay - std::fmod(timeOfDay, timeOfDayEncoder_->parameters.radius);
    VERBOSE << "  timeOfDay: " << timeOfDay << "hrs ==> " << timeOfDay_output;
    sdrs.push_back(&timeOfDay_output);
  }
  if (sdrs.size() > 1)
    output.concatenate(sdrs);
  else
    output = *sdrs[0];
  VERBOSE << "  Result: ==> " << output << std::endl;
}


bool DateEncoder::operator==(const DateEncoder &other) const {
  if (args_.season_width != other.args_.season_width)
    return false;
  if (args_.season_radius != other.args_.season_radius)
    return false;
  if (args_.dayOfWeek_width != other.args_.dayOfWeek_width)
    return false;
  if (args_.dayOfWeek_radius != other.args_.dayOfWeek_radius)
    return false;
  if (args_.weekend_width != other.args_.weekend_width)
    return false;
  if (args_.holiday_width != other.args_.holiday_width)
    return false;
  if (args_.holiday_dates != other.args_.holiday_dates)
    return false;
  if (args_.timeOfDay_width != other.args_.timeOfDay_width)
    return false;
  if (args_.timeOfDay_radius != other.args_.timeOfDay_radius)
    return false;
  if (args_.custom_width != other.args_.custom_width)
    return false;
  if (args_.custom_days != other.args_.custom_days)
    return false;
  if (args_.holiday_dates.size() != other.args_.holiday_dates.size())
    return false;
  for (size_t i = 0; i < args_.holiday_dates.size(); i++ ) {
    if (args_.holiday_dates[i] != other.args_.holiday_dates[i])
      return false;
  }
  if (args_.custom_days.size() != other.args_.custom_days.size())
    return false;
  for (size_t i = 0; i < args_.custom_days.size(); i++) {
    if (args_.custom_days[i] != other.args_.custom_days[i])
      return false;
  }

  return true;
}


time_t DateEncoder::mktime(int year, int mon, int day, int hr, int min, int sec) { 
  struct tm tm; 
  tm.tm_year = year - 1900;
  tm.tm_mon = mon - 1;
  tm.tm_mday = day;
  tm.tm_hour = hr;
  tm.tm_min = min;
  tm.tm_sec = sec;
  tm.tm_isdst = -1;
  time_t t = std::mktime(&tm);
  return t;
}


std::ostream &operator<<(std::ostream &out, const DateEncoder &self) {
  out << "DateEncoder \n";
  out << "  season_width:   " << self.parameters.season_width << ",\n";
  out << "  season_radius:   " << self.parameters.season_radius << ",\n";
  out << "  dayOfWeek_width: " << self.parameters.dayOfWeek_width << ",\n";
  out << "  dayOfWeek_radius:  " << self.parameters.dayOfWeek_radius << ",\n";
  out << "  weekend_width:  " << self.parameters.weekend_width << ",\n";
  out << "  holiday_width:" << self.parameters.holiday_width << ",\n";
  out << "  timeOfDay_width:" << self.parameters.timeOfDay_width << ",\n";
  out << "  timeOfDay_radius:" << self.parameters.timeOfDay_radius << ",\n";
  out << "  custom_width:" << self.parameters.custom_width << ",\n";
  if (self.parameters.custom_days.size() == 1)
    out << "  custom_days:" << self.parameters.custom_days[0] << std::endl;
  else {
    out << "  custom_days:\n";
    for (auto custom : self.parameters.custom_days) {
      out << "    " << custom << "\n";
    }
  }
  return out;
}


} // end namespace htm
