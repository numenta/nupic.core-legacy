/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * Unit tests for the DateEncoder
 */

#include "gtest/gtest.h"
#include <htm/encoders/DateEncoder.hpp>
#include <vector>

namespace testing {

using namespace htm;

#define VERBOSE if (verbose) std::cerr << "[          ] "
static bool verbose = false; // turn this on to print extra stuff for debugging.

struct DateValueCase {
  std::vector<UInt> time;           // year,mon,day,hr,min,sec
  std::vector<Real64> bucket;       // titles for samples (quantized values)
  std::vector<UInt> expectedOutput; // Sparse indices of active bits.
};


// test helper routine
void doDateValueCases(DateEncoder &e, std::vector<DateValueCase> cases) {
  for (auto c : cases) {
    SDR expectedOutput(e.dimensions);
    std::sort(c.expectedOutput.begin(), c.expectedOutput.end());
    expectedOutput.setSparse(c.expectedOutput);

    time_t input = DateEncoder::mktime(c.time[0], c.time[1], c.time[2], c.time[3], c.time[4], (c.time.size()>5)?c.time[5]:0);
    NTA_CHECK(input != -1) << "DateEncoder: Input date is invalid or before Jan 1 1970.";

    SDR actualOutput(e.dimensions);
    e.encode(input, actualOutput);

    EXPECT_EQ(e.buckets, c.bucket);
    EXPECT_EQ(actualOutput, expectedOutput);
  }
}



TEST(DateEncoderTest, season) {

  DateEncoderParameters p;
  p.verbose = verbose;
  p.season_width = 5;   // 5 bits per season; bits 0-4=winter, 5-8=spring, etc.
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time          bucket  expected output
      {{2020, 1, 1, 0, 0},     {0.0}, {0, 1, 2, 3, 4}},       // New years day, midnight local time
      {{2019, 12, 11, 14, 45}, {3.0}, {0, 1, 2, 3, 19}},      // winter, Wed, afternoon
      {{2010, 11, 4, 14, 55},  {3.0}, {0, 1, 17, 18, 19}},    // Nov 4th, fall, thursday
      {{2019, 7, 4, 0, 0},     {2.0}, {10, 11, 12, 13, 14}},  // July 4, summer, holiday
      {{2019, 4, 21, 0, 0},    {1.0}, {6, 7, 8, 9, 10}},      // sunday, Easter holiday
      {{2017, 4, 17, 0, 0},    {1.0}, {6, 7, 8, 9, 10}},      // start of my special day
      {{2017, 4, 17, 22, 59},  {1.0}, {6, 7, 8, 9, 10}},      // end of my special day
      {{1988, 5, 29, 20, 00},  {1.0}, {8, 9, 10, 11, 12}},    // Sun afternoon, a weekend
      {{1988, 5, 27, 20, 00},  {1.0}, {8, 9, 10, 11, 12}}     // Fri afternoon, a weekend
  };

  doDateValueCases(encoder, cases);
}

TEST(DateEncoderTest, dayOfWeek) {

  DateEncoderParameters p;
  p.verbose = verbose;
  p.dayOfWeek_width = 2;   // 2 bits per day; bits 0,1=monday, 2,3=tues, 4,5=wed, etc.
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time          bucket    expected
      {{2020, 1, 1, 0, 0},     {2.0}, {4, 5}},  // Wed, New years day 2020, midnight local time
      {{2019, 12, 11, 14, 45}, {2.0}, {4, 5}},  // Wed, afternoon, winter
      {{2010, 11, 4, 14, 55},  {3.0}, {6, 7}},  // Thu, Nov 4th, fall
      {{2019, 7, 4, 0, 0},     {3.0}, {6, 7}},  // Thu, July 4, summer, holiday
      {{2019, 4, 21, 0, 0},    {6.0}, {12, 13}},// Sun, Easter holiday
      {{2017, 4, 17, 0, 0},    {0.0}, {0, 1}},  // Mon, start of my special day
      {{2017, 4, 17, 22, 59},  {0.0}, {0, 1}},  // Mon, end of my special day
      {{1988, 5, 29, 20, 00},  {6.0}, {12, 13}},// Sun, afternoon, a weekend
      {{1988, 5, 27, 20, 00},  {4.0}, {8, 9}}   // Fri, afternoon, a weekend
  };

  doDateValueCases(encoder, cases);
}


TEST(DateEncoderTest, weekend) {
  // Weekend defined as Fri after noon until Sun midnight.
  DateEncoderParameters p;
  p.verbose = verbose;
  p.weekend_width = 2; // 2 bits per state (bits 0,1=off, bits 2,3=on) to indicate weekend.
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time             bucket  expected
      {{2020, 1, 1, 0, 0},        {0.0}, {0, 1}},  // Wed, New years day 2020, midnight local time
      {{2019, 12, 11, 14, 45},    {0.0}, {0, 1}},  // Wed, afternoon, winter
      {{2010, 11, 4, 14, 55},     {0.0}, {0, 1}},  // Thu, Nov 4th, fall
      {{2019, 7, 4, 0, 0},        {0.0}, {0, 1}},  // Thu, July 4, summer, holiday
      {{2019, 4, 21, 0, 0},       {1.0}, {2, 3}},  // sunday, Easter holiday
      {{2017, 4, 17, 0, 0},       {0.0}, {0, 1}},  // Mon, start of my special day
      {{2017, 4, 17, 22, 59},     {0.0}, {0, 1}},  // Mon, end of my special day
      {{1988, 5, 29, 20, 00},     {1.0}, {2, 3}},  // Sun, afternoon, a weekend
      {{1988, 5, 27, 11, 00},     {0.0}, {0, 1}},  // Fri, morning, not a weekend
      {{1988, 5, 27, 20, 00},     {1.0}, {2, 3}}   // Fri, afternoon, a weekend
  };

  doDateValueCases(encoder, cases);
}


TEST(DateEncoderTest, holiday) {
 
  DateEncoderParameters p;
  p.verbose = verbose;
  p.holiday_width = 4; // 4 bits per state (bits 0-3=off, bits 4-8=on) to indicate holiday.
  p.holiday_dates = {{2020, 1, 1}, {7, 4}, {2019, 4, 21}};
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time             bucket    expected
      {{2019, 12, 31, 0, 0},     {0.0}, {0, 1, 2, 3}}, // off - 24 hrs before new years day
      {{2019, 12, 31, 12, 0},    {0.0}, {2, 3, 4, 5}}, // 50% - noon on day before new years day
      {{2020, 1, 1, 0, 0},       {1.0}, {4, 5, 6, 7}}, // on  - start of New years day 2020, midnight local time
      {{2020, 1, 1, 12, 0},      {1.0}, {4, 5, 6, 7}}, // on  - noon on new years day
      {{2020, 1, 1, 23, 59},     {1.0}, {4, 5, 6, 7}}, // on  - at end of new years day
      {{2020, 1, 2, 12, 0},      {1.0}, {0, 1, 6, 7}}, // 50% - noon day after new years day
      {{2020, 1, 3, 0, 0},       {0.0}, {0, 1, 2, 3}}, // off - 24hrs after new years day end
      {{2019, 12, 11, 14, 45},   {0.0}, {0, 1, 2, 3}}, // off - Wed, afternoon, winter
      {{2010, 11, 4, 14, 55},    {0.0}, {0, 1, 2, 3}}, // off - Thu, Nov 4th, fall
      {{2019, 7, 4, 0, 0},       {1.0}, {4, 5, 6, 7}}, // on  - Thu, July 4, summer, holiday
      {{2019, 4, 21, 0, 0},      {1.0}, {4, 5, 6, 7}}, // on  - Sunday, Easter holiday
      {{2017, 4, 17, 0, 0},      {0.0}, {0, 1, 2, 3}}, // off - Mon, start of my special day
  };

  doDateValueCases(encoder, cases);
}

TEST(DateEncoderTest, timeOfDay) {

  DateEncoderParameters p;
  p.verbose = verbose;
  p.timeOfDay_width = 4; // 4 bits per bucket to indicate time ranges.
  p.timeOfDay_radius = 4;  // One bucket per 4 hrs, means 6 buckets with wrap. 
                           // bits 0,3=midnight-4am, 
                           // bits 4,7=4am-8am, 
                           // bits 8,11=8am-noon, 
                           // bits 12,15=noon-4pm,
                           // bits 16,19=4pm-8pm,
                           // bits 20,23=8pm-midnight
                           // bits will slide between those ranges. Full width is 24 bits.
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time          bucket    expected
      {{2020, 1, 1, 0, 0},     { 0.0}, {0, 1, 2, 3}},    // hr=0       Wed, New years day 2020, midnight local time
      {{2019, 12, 11, 14, 45}, {12.0}, {15, 16, 17, 18}},// hr=14.75   Wed, afternoon, winter
      {{2010, 11, 4, 14, 55},  {12.0}, {15, 16, 17, 18}},// hr=14.9167 Thu, Nov 4th, fall dst
      {{2019, 7, 4, 0, 0},     { 0.0}, {0, 1, 2, 3}},    // hr=0       Thu, July 4, summer, holiday, nidnight dst
      {{2019, 4, 21, 12, 0},   {12.0}, {12, 13, 14, 15}},// hr=12      Sun, Easter holiday dst, noon dst
      {{2017, 4, 17, 1, 0},    { 0.0}, {1, 2, 3, 4}},    // hr=1       Mon, start of my special day 1:00am dst
      {{2017, 4, 17, 22, 59},  {20.0}, {0, 1, 2, 23}},   // hr=22.9833 Mon, end of my special day dst
      {{1988, 5, 29, 20, 00},  {20.0}, {20, 21, 22, 23}},// hr=20      Sun, afternoon, a weekend
      {{1988, 5, 27, 11, 00},  { 8.0}, {11, 12, 13, 14}},// hr=11      Fri, morning, not a weekend
      {{1988, 5, 27, 20, 00},  {20.0}, {20, 21, 22, 23}} // hr=20      Fri, afternoon, a weekend
  };

  doDateValueCases(encoder, cases);
}

TEST(DateEncoderTest, customDay) {

  DateEncoderParameters p;
  p.verbose = verbose;
  p.custom_width = 2; // 2 bits binary, bits 0,1=off bits 2,3=on.  It eather is or it is not in any of the days.
  p.custom_days = {"Monday", "Mon, Wed, Fri"}; 
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time            bucket   expected
      {{2020, 1, 1, 0, 0},     {1.0},    {2, 3}}, // Wed, New years day 2020, midnight local time
      {{2019, 12, 11, 14, 45}, {1.0},    {2, 3}}, // Wed, afternoon, winter
      {{2010, 11, 4, 14, 55},  {0.0},    {0, 1}}, // Thu, Nov 4th, fall
      {{2019, 7, 4, 0, 0},     {0.0},    {0, 1}}, // Thu, July 4, summer, holiday
      {{2019, 4, 21, 0, 0},    {0.0},    {0, 1}}, // sunday, Easter holiday
      {{2017, 4, 17, 0, 0},    {1.0},    {2, 3}}, // Mon, start of my special day
      {{2017, 4, 17, 22, 59},  {1.0},    {2, 3}},  // Mon, end of my special day
      {{1988, 5, 29, 20, 00},  {0.0},    {0, 1}},  // Sun, afternoon, a weekend
      {{1988, 5, 27, 11, 00},  {1.0},    {2, 3}},  // Fri, morning, not a weekend
      {{1988, 5, 27, 20, 00},  {1.0},    {2, 3}}   // Fri, afternoon, a weekend
  };

  doDateValueCases(encoder, cases);
}


TEST(DateEncoderTest, combined) {
  DateEncoderParameters p;
  p.verbose = verbose;
  p.season_width = 5;    // 5 bits per season; bits 0-4=winter, 5-8=spring, etc.
  p.dayOfWeek_width = 2; // 2 bits per day; bits 0,1=monday, 2,3=tues, 4,5=wed, etc.
  p.weekend_width = 2;   // 2 bits per state (bits 0,1=off, bits 2,3=on) to indicate weekend.
  p.custom_width = 2;    // 2 bits binary, bits 0,1=off bits 2,3=on.  It eather is or it is not in any of the days.
  p.custom_days = {"Monday", "Mon, Wed, Fri"};
  p.holiday_width = 2;   // 2 bits per state (bits 0,1=off, bits 2,3=on) to indicate holiday.
  p.holiday_dates = {{2020, 1, 1}, {7, 4}, {2019, 4, 21}};
  p.timeOfDay_width = 4; // 4 bits per bucket to indicate time ranges.
  p.timeOfDay_radius = 4;// One bucket per 4 hrs, means 6 buckets with wrap.
  DateEncoder encoder(p);

  static std::vector<DateValueCase> cases = {
      //    date/time               buckets             expected
      {{2020, 1, 1, 0, 0},     {0, 2, 0, 1, 1, 0},  {0, 1, 2, 3, 4, 24, 25, 34, 35, 40, 41, 44, 45, 46, 47, 48, 49}},
      {{2019, 12, 11, 14, 45}, {3, 2, 0, 1, 0, 12}, {0, 1, 2, 3, 19, 24, 25, 34, 35, 40, 41, 42, 43, 61, 62, 63, 64}},
      {{2010, 11, 4, 14, 55},  {3, 3, 0, 0, 0, 12}, {0, 1, 17, 18, 19, 26, 27, 34, 35, 38, 39, 42, 43, 61, 62, 63, 64}},
      {{2019, 7, 4, 0, 0},     {2, 3, 0, 0, 1, 0},  {10, 11, 12, 13, 14, 26, 27, 34, 35, 38, 39, 44, 45, 46, 47, 48, 49}},
      {{2019, 4, 21, 0, 0},    {1, 6, 1, 0, 1, 0},  {6, 7, 8, 9, 10, 32, 33, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49}},
      {{2017, 4, 17, 0, 0},    {1, 0, 0, 1, 0, 0},  {6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 49}},
      {{2017, 4, 17, 22, 59},  {1, 0, 0, 1, 0, 20}, {6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 69}},
      {{1988, 5, 29, 20, 00},  {1, 6, 1, 0, 0, 20}, {8, 9, 10, 11, 12, 32, 33, 36, 37, 38, 39, 42, 43, 66, 67, 68, 69}},
      {{1988, 5, 27, 11, 00},  {1, 4, 0, 1, 0, 8},  {8, 9, 10, 11, 12, 28, 29, 34, 35, 40, 41, 42, 43, 57, 58, 59, 60}},
      {{1988, 5, 27, 20, 00},  {1, 4, 1, 1, 0, 20}, {8, 9, 10, 11, 12, 28, 29, 36, 37, 40, 41, 42, 43, 66, 67, 68, 69}}};

  doDateValueCases(encoder, cases);
}


TEST(DateEncoderTest, Serialization) {
  DateEncoderParameters p;
  p.verbose = verbose;
  p.season_width = 5;    // 5 bits per season; bits 0-4=winter, 5-8=spring, etc.
  p.dayOfWeek_width = 2; // 2 bits per day; bits 0,1=monday, 2,3=tues, 4,5=wed, etc.
  p.weekend_width = 2;   // 2 bits per state (bits 0,1=off, bits 2,3=on) to indicate weekend.
  p.custom_width = 2;    // 2 bits binary, bits 0,1=off bits 2,3=on.  It eather is or it is not in any of the days.
  p.custom_days = {"Monday", "Mon, Wed, Fri"}; 
  p.holiday_width = 2;   // 2 bits per state (bits 0,1=off, bits 2,3=on) to indicate holiday.
  p.holiday_dates = {{2020, 1, 1}, {7, 4}, {2019, 4, 21}};
  p.timeOfDay_width = 4; // 4 bits per bucket to indicate time ranges.
  p.timeOfDay_radius = 4;// One bucket per 4 hrs, means 6 buckets with wrap.
  DateEncoder encoder(p);

  std::stringstream buf;
  encoder.save(buf, JSON);

  //std::cerr << "SERIALIZED:" << std::endl << buf.str() << std::endl;
  buf.seekg(0);

  DateEncoder enc2;
  enc2.load(buf, JSON);
  EXPECT_EQ(encoder, enc2);

  static std::vector<DateValueCase> cases = {
      //    date/time               buckets              expected
      {{2020, 1, 1, 0, 0},     {0, 2, 0, 1, 1, 0},  {0, 1, 2, 3, 4, 24, 25, 34, 35, 40, 41, 44, 45, 46, 47, 48, 49}},
      {{2019, 12, 11, 14, 45}, {3, 2, 0, 1, 0, 12}, {0, 1, 2, 3, 19, 24, 25, 34, 35, 40, 41, 42, 43, 61, 62, 63, 64}},
      {{2010, 11, 4, 14, 55},  {3, 3, 0, 0, 0, 12}, {0, 1, 17, 18, 19, 26, 27, 34, 35, 38, 39, 42, 43, 61, 62, 63, 64}},
      {{2019, 7, 4, 0, 0},     {2, 3, 0, 0, 1, 0},  {10, 11, 12, 13, 14, 26, 27, 34, 35, 38, 39, 44, 45, 46, 47, 48, 49}},
      {{2019, 4, 21, 0, 0},    {1, 6, 1, 0, 1, 0},  {6, 7, 8, 9, 10, 32, 33, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49}},
      {{2017, 4, 17, 0, 0},    {1, 0, 0, 1, 0, 0},  {6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 49}},
      {{2017, 4, 17, 22, 59},  {1, 0, 0, 1, 0, 20}, {6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 69}},
      {{1988, 5, 29, 20, 00},  {1, 6, 1, 0, 0, 20}, {8, 9, 10, 11, 12, 32, 33, 36, 37, 38, 39, 42, 43, 66, 67, 68, 69}},
      {{1988, 5, 27, 11, 00},  {1, 4, 0, 1, 0, 8},  {8, 9, 10, 11, 12, 28, 29, 34, 35, 40, 41, 42, 43, 57, 58, 59, 60}},
      {{1988, 5, 27, 20, 00},  {1, 4, 1, 1, 0, 20}, {8, 9, 10, 11, 12, 28, 29, 36, 37, 40, 41, 42, 43, 66, 67, 68, 69}}};

  doDateValueCases(enc2, cases);
}

} // namespace testing
