/* ----------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, David Keeney, dkeeney@gmail.com
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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
namespace py = pybind11;

#include <htm/encoders/DateEncoder.hpp>
#include <htm/types/Sdr.hpp>

namespace htm_ext
{
  using namespace htm;

  void init_DateEncoder(py::module& m)
  {
  
    //     DateEncoderParameters
    
    py::class_<DateEncoderParameters> py_DateEncParams(m, "DateEncoderParameters",
        R"(

  The DateEncoderParameters structure is used to pass configuration parameters to 
  the DateEncoder. These Six (6) members define the total number of bits in the output.
      Members:  season, dayOfWeek, weekend, holiday, timeOfDay, customDays
 
  Each member is a separate attribute of a date/time that can be activated
  by providing a width parameter and sometimes a radius parameter.
  Each is implemented separately using a ScalarEncoder and the results
  are concatinated together.
 
  The width attribute determines the number of bits to be used for each member.
  and 0 means don't use.  The width is like a weighting to indicate the relitive importance
  of this member to the overall data value.
 
  The radius attribute indicates the size of the bucket; the quantization size.
  All values in the same bucket generate the same pattern.
 
  To avoid problems with leap year, consider a year to have 366 days.
  The timestamp will be converted to components such as time and dst based on 
  local timezone and location (see localtime()).
)");

    py_DateEncParams.def(py::init<>(), R"()");

    py_DateEncParams.def_readwrite("season_width", &DateEncoderParameters::season_width,
R"( (int)how many bits to apply to season.)");

    py_DateEncParams.def_readwrite("season_radius", &DateEncoderParameters::season_radius,
R"(season_radius)");

    py_DateEncParams.def_readwrite("dayOfWeek_width", &DateEncoderParameters::dayOfWeek_width,
R"(how many bits to apply to day of week.)");

    py_DateEncParams.def_readwrite("dayOfWeek_radius", &DateEncoderParameters::dayOfWeek_radius,
R"( (double) how many days in a bucket. Default is 1.0 days. )");

    py_DateEncParams.def_readwrite("weekend_width", &DateEncoderParameters::weekend_width,
R"(How many bits to apply to the weekend attribute. )");

    py_DateEncParams.def_readwrite("holiday_width", &DateEncoderParameters::holiday_width,
R"( How many bits to apply to a holiday attribute. )");

    py_DateEncParams.def_readwrite("holiday_dates", &DateEncoderParameters::holiday_dates,
R"(Each holiday is either [month, day] or [year, month, day].
   The former will use the same month day every year eg: [12, 25] for Christmas.
   The latter will be a one off holiday eg: [2018, 4, 1] for Easter Sunday 2018.)");

    py_DateEncParams.def_readwrite("timeOfDay_width", &DateEncoderParameters::timeOfDay_width,
R"( How many bits to apply to time-of-day attribute. )");

    py_DateEncParams.def_readwrite("timeOfDay_radius", &DateEncoderParameters::timeOfDay_radius,
R"( (double) How many hrs are in a bucket. Default is every 4 hrs is a bucket.  Use 0.25f for 15min buckets, etc.)");

    py_DateEncParams.def_readwrite("custom_width", &DateEncoderParameters::custom_width,
R"( (int) How many bits to apply to custom days.  A way to custom encode specific groups of days of the week as a category.)");

     py_DateEncParams.def_readwrite("custom_days", &DateEncoderParameters::custom_days,
R"( (vector of strings) The custom days list is a vector of strings. Each string can be something like "Monday" or "mon", or 
  a list like "mon,wed,fri". )");

   py_DateEncParams.def_readwrite("verbose", &DateEncoderParameters::verbose,
R"( (bool)when true, displays some debug info for each time member that is actuvated.  )");


    py::class_<DateEncoder> py_DateEnc(m, "DateEncoder",
R"( * The DateEncoder encodes up to 6 attributes of a timestamp value into an array of 0's and 1's.

 The input is a timestamp which is unix date/time; an integral value representing the number of
 seconds elapsed since 00:00 hours, Jan 1, 1970 UTC (the unix EPOCH).  Some platforms (unix and linux)
 allow negitive numbers as the timestamp which allows time before EPOCH to be expressed.
 However some platforms (windows) allow only positive numbers.  If the type time_t on your computer
 is is 32bits then the timestamp will not allow dates after Jan 18, 2038. By default, on windows
 it is 64bit but on some older 32bit linux machines time_t is 32bit. google "Y2K38".

 The output is an array containing 0's except for a contiguous block of 1's for each
 attribute member. This is held in an SDR container although technically this is not
 a sparse representation. It is normally passed to a SpatialPooler which will turn
 this into a true sparse representation.
)");


    //    DateEncoder
    //
    // This version of constructor is consistant with ScalarEncoder, it uses the parameter structure.
    // Use  the wrapper 'import htm.encoders.date_encoder" found in date_encoder.py to be 
    // compatable with .py version of DateEncoder.
    //
    // To use this interface, use something like this:
    //
    //    import datetime
    //    from htm.bindings.encoders import DateEncoder, DateEncoderParameters
    //    from htm.bindings.sdr import SDR, Metrics
    //
    //    p = DateEncoderParameters()
    //    p.dateTime_width = 5
    //    encoder = DateEncoder(p)
    //
    //    d = datetime.datetime(2010, 11, 4, 14, 55)
    //    SDR sdr = encoder(d)
    //
    py_DateEnc.def(py::init<DateEncoderParameters&>(), R"()");
    
    py_DateEnc.def_property_readonly("parameters",
        [](const DateEncoder &self) { return self.parameters; },
R"(Contains the parameter structure which this encoder uses internally. All
fields are filled in.)");

    py_DateEnc.def_property_readonly("dimensions",
        [](const DateEncoder &self) { return self.dimensions; });
    py_DateEnc.def_property_readonly("size",
        [](const DateEncoder &self) { return self.size; });


    py_DateEnc.def("encode", [](DateEncoder &self, std::chrono::system_clock::time_point time_point) {
        auto output = new SDR( self.dimensions );
        self.encode( time_point, *output );
        return output; },
R"(Encodes a .py datetime.datetime into an SDR structure. )", 
      py::return_value_policy::take_ownership);
      
      py_DateEnc.def("encode", [](DateEncoder &self, std::chrono::system_clock::time_point time_point, SDR* output) {
        self.encode( time_point, *output );
        return output; },
R"(Encodes a .py datetime.datetime into an SDR structure. )");
  }

}
