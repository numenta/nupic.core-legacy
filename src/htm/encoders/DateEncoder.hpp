/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, Numenta, Inc.
 *               2020, David Keeney, dkeeney@gmail.com
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
 *
 * Description:
 * The DateEncoder encodes up to 6 attributes of a timestamp value into an array
 * of 0's and 1's.
 *
 * The input is a timestamp which is unix date/time; an integral value representing the number of
 * seconds elapsed since 00:00 hours, Jan 1, 1970 UTC (the unix EPOCH).  Some platforms (unix and linux)
 * allow negitive numbers as the timestamp which allows time before EPOCH to be expressed.
 * However some platforms (windows) allow only positive numbers.  If the type time_t on your computer
 * is is 32bits then the timestamp will not allow dates after Jan 18, 2038. By default, on windows
 * it is 64bit but on some older 32bit linux machines time_t is 32bit. google "Y2K38".
 *
 * The output is an array containing 0's except for a contiguous block of 1's for each
 * attribute member. This is held in an SDR container although technically this is not
 * a sparse representation. It is normally passed to a SpatialPooler which will turn
 * this into a true sparse representation.
 */


#ifndef NTA_DATE_ENCODER_HPP
#define NTA_DATE_ENCODER_HPP

#include <time.h> // struct tm
#include <chrono> // system_clock, std::chrono::system_clock::time_point

#include <htm/types/Types.hpp>
#include <htm/encoders/BaseEncoder.hpp>
#include <htm/encoders/ScalarEncoder.hpp>

namespace htm {


/**
 * The DateEncoderParameters structure is used to pass configuration parameters to 
 * the DateEncoder. These Six (6) members define the total number of bits in the output.
 *     Members:  season, dayOfWeek, weekend, holiday, timeOfDay, customDays
 *
 * Each member is a separate attribute of a date/time that can be activated
 * by providing a width parameter and sometimes a radius parameter.
 * Each is implemented separately using a ScalarEncoder and the results
 * are concatinated together.
 *
 * The width attribute determines the number of bits to be used for each member.
 * and 0 means don't use.  The width is like a weighting to indicate the relitive importance
 * of this member to the overall data value.
 *
 * The radius attribute indicates the size of the bucket; the quantization size.
 * All values in the same bucket generate the same pattern.
 *
 * To avoid problems with leap year, consider a year to have 366 days.
 * The timestamp will be converted to components such as time and dst based on 
 * local timezone and location (see localtime()).
 *
 */
struct DateEncoderParameters {
  /**
   *  Member: season -  The portion of the year. Unit is day.  Default radius is
   *                    91.5 days which gives 4 seasons per year.
   */
  UInt season_width = 0u;       // how many bits to apply to season
  Real32 season_radius = 91.5f; // days per season

  /**
   * Member: dayOfWeek -  Day of week where monday = 0, tuesday = 1, etc.
   *                      The timestamp is compared against day:noon,
   *                      so encodings of a day switch representation on midnight.
   */
  UInt dayOfWeek_width = 0u;      // how many bits to apply to day of week.
  Real32 dayOfWeek_radius = 1.0f; // every day is a separate bucket.

  /**
   * Member: weekend     - Is a weekend or not, boolean: 0, 1. A block of bits either 0s or 1s.
   *                       Note: the implementation treats "weekend" as starting Fri 6pm, till Sun midnight
   */
  UInt weekend_width = 0u; //  how many bits to apply for a weekend.

  /**
   * Member: holiday     - Is a holiday or not, boolean: 0, 1
   *                       Each holiday is either "month, day" or "year, month, day".
   *                       The former will use the same month day every year eg: {12, 25} for Christmas.
   *                       The latter will be a one off holiday eg: {2018, 4, 1} for Easter Sunday 2018.
   *                       By default the only holiday is December 25.
   */
  UInt holiday_width = 0; //  how many bits to apply for a holiday.
  std::vector<std::vector<int>> holiday_dates = {{12,25}};

  /**
   * Member: timeOfday   - Time of day, where midnight = 0, noon=12, etc. units = hour.
   *                       NOTE: if the date is daylight savings time, the bits will be set based on local time.
   */
  UInt timeOfDay_width = 0u;      //  how many bits to apply for time of day.
  Real32 timeOfDay_radius = 4.0f; //  as default every 4 hrs is a bucket.  Use 0.25f for 15min buckets, etc.

  /**
   * Member: customDays   - A way to custom encode specific groups of days of the week. The value is 1.0 if the day is in
   *                        any one of the given ranges listed in the custom_days list.  The custom days list
   *                        is a vector of strings. Each string can be something like "Monday" or "mon", or 
   *                        a list like "mon,wed,fri".
   */
  UInt custom_width = 0u;               // how many bits to apply for custom day(s) of week.
  std::vector<std::string> custom_days; // list of day ranges.

  /**
   * verbose:  when true, displays some debug info for each time member that is actuvated.
   */
  bool verbose = false;
};


/**
 * DateEncoder is the class which implements the Date Encoder.
 */

class DateEncoder : public BaseEncoder<time_t> {
public:
  DateEncoder(){};
  DateEncoder(const DateEncoderParameters &parameters);
  void initialize(const DateEncoderParameters &parameters);

  /**
   * Const Access to the parameters with which the DateEncoder was configured.
   * Some parameters may have been modified/computed during initialize().
   */
  const DateEncoderParameters &parameters = args_;

  /**
   * Turn on/off verbose on-the-fly.
   */
  void setVerbose(bool verbose) { args_.verbose = verbose; }

  /**
   * Const Access to the buckets configured with this encoder.
   * For each attribute encoded, this is the quantized value used as title in Classifier.
   * This only provides values for the attributes that were enabled.
   * The order is: season, dayofweek, weekend, custom, holiday, timeofday
   */
  const std::vector<Real64> &buckets = buckets_;

  /**
   * encode the input and generate the output pattern.
   * The input is unix time, same as would be generated by time(0)
   * which is seconds since EPOCH, Jan 1, 1970.
   * Inputs of time_point or struc tm are converted to time_t.
   *
   * Output is an array of 0's and 1's in an SDR container.
   */
  void encode(std::time_t input, SDR &output) override;                  // unix EPOCH time
  void encode(std::chrono::system_clock::time_point, SDR &output);  // python datetime
  void encode(struct std::tm input, SDR &output);

  /**
   * Serialization Facility.
   */
  CerealAdapter; // see Serializable.hpp
  // FOR Cereal Serialization
  template <class Archive> void save_ar(Archive &ar) const {
    std::string name = "DateEncoder";
    ar(cereal::make_nvp("name", name));
    ar(cereal::make_nvp("season_width", args_.season_width));
    ar(cereal::make_nvp("season_radius", args_.season_radius));
    ar(cereal::make_nvp("dayOfWeek_width", args_.dayOfWeek_width));
    ar(cereal::make_nvp("dayOfWeek_radius", args_.dayOfWeek_radius));
    ar(cereal::make_nvp("weekend_width", args_.weekend_width));
    ar(cereal::make_nvp("holiday_width", args_.holiday_width));
    ar(cereal::make_nvp("holiday_dates", args_.holiday_dates));
    ar(cereal::make_nvp("timeOfDay_width", args_.timeOfDay_width));
    ar(cereal::make_nvp("timeOfDay_radius", args_.timeOfDay_radius));
    ar(cereal::make_nvp("custom_width", args_.custom_width));
    ar(cereal::make_nvp("custom_days", args_.custom_days));
    ar(cereal::make_nvp("verbose", args_.verbose));
  }

  // FOR Cereal Deserialization
  template <class Archive> void load_ar(Archive &ar) {
    std::string name;
    ar(cereal::make_nvp("name", name));
    NTA_CHECK(name == "DateEncoder") << "DateEncoder: load_ar() bad decoding.";
    ar(cereal::make_nvp("season_width", args_.season_width));
    ar(cereal::make_nvp("season_radius", args_.season_radius));
    ar(cereal::make_nvp("dayOfWeek_width", args_.dayOfWeek_width));
    ar(cereal::make_nvp("dayOfWeek_radius", args_.dayOfWeek_radius));
    ar(cereal::make_nvp("weekend_width", args_.weekend_width));
    ar(cereal::make_nvp("holiday_width", args_.holiday_width));
    ar(cereal::make_nvp("holiday_dates", args_.holiday_dates));
    ar(cereal::make_nvp("timeOfDay_width", args_.timeOfDay_width));
    ar(cereal::make_nvp("timeOfDay_radius", args_.timeOfDay_radius));
    ar(cereal::make_nvp("custom_width", args_.custom_width));
    ar(cereal::make_nvp("custom_days", args_.custom_days));
    ar(cereal::make_nvp("verbose", args_.verbose));
    initialize(args_);
  }

  ~DateEncoder() override{};

  bool operator==(const DateEncoder &other) const;
  inline bool operator!=(const DateEncoder &other) const { return !operator==(other); }

  // a convenience method to generate unix EPOCH time values.
  static time_t mktime(int year, int mon, int day, int hr=0, int min=0, int sec=0);

private:
  DateEncoderParameters args_;

  // fields populated by initialize()
  std::shared_ptr<ScalarEncoder> seasonEncoder_;
  std::shared_ptr<ScalarEncoder> dayOfWeekEncoder_;
  std::shared_ptr<ScalarEncoder> weekendEncoder_;
  std::shared_ptr<ScalarEncoder> customDaysEncoder_;
  std::shared_ptr<ScalarEncoder> holidayEncoder_;
  std::shared_ptr<ScalarEncoder> timeOfDayEncoder_;
  std::set<int> customDays_;

  // Titles from the last encoding
  size_t bucketMap_[6];
  std::vector<Real64> buckets_;

}; // end class DateEncoder

std::ostream &operator<<(std::ostream &out, const DateEncoder &self);


} // end namespace htm
#endif // end NTA_DATE_ENCODER_HPP
