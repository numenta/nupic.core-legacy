/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, Numenta, Inc.
 *
 * Author: David Keeney, Jan 2020,  dkeeney@gmail.com
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
 * Implementation of the DateEncoder Region
 */

#include <htm/regions/DateEncoderRegion.hpp>

#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include "htm/utils/Random.hpp"
#include <htm/utils/Log.hpp>

#include <memory>
#include <stdlib.h>

namespace htm {


/* static */ Spec *DateEncoderRegion::createSpec() {
  Spec *ns = new Spec();
  ns->parseSpec(R"(
  {name: "DateEncoderRegion",
      parameters: {
          season_width:     {type: UInt32, default: "0"},
          season_radius:    {type: Real32, default: "91.5"},
          dayOfWeek_width:  {type: UInt32, default: "0"},
          dayOfWeek_radius: {type: Real32, default: "1.0"},
          weekend_width:    {type: UInt32, default: "0"},
          holiday_width:    {type: UInt32, default: "0"},
          timeOfDay_width:  {type: UInt32, default: "0"},
          timeOfDay_radius: {type: Real32, default: "4.0"},
          custom_width:     {type: UInt32, default: "0"},
          custom_days:      {description: "A list of day names to be included in the set, i.e. 'mon,tue,fri'",
                             type: String,    default: ""},
          holiday_dates:    {description: "A list of holiday dates in format of 'month,day' or 'year,month,day', ie [[12,25],[2020,05,04]]",
                             type: String,    default: "[[12,25]]"},
          verbose:          {description: "if true, display debug info for each member encoded.",
                             type: Bool,   default: "false", access: ReadWrite },
          size:             {description: "Total width of encoded output.",
                             type: UInt32, default: "", access: ReadOnly },
          noise:            {description: "amount of noise to add to the output SDR. 0.01 is 1%",
                             type: Real32, default: "0.0", access: ReadWrite },
          sensedTime:       {description: "The value to encode. Unix EPOCH time. Overriden by input 'values'. A value of 0 means current time.",
                             type: Int64, default: "0", access: ReadWrite }},
      inputs: {
          values:           {description: "Values to encode. Overrides sensedTime.",
                             type: Int64, count: 1, isDefaultInput: yes, isRegionLevel: yes}
      }, 
      outputs: {
          bucket:          {description: "Quantized samples based on the radius. One sample for each attribute used. Becomes the title for this sample in Classifier.",
                             type: Real64, count: 0, isDefaultOutput: false, isRegionLevel: false },
          encoded:          {description: "Encoded bits. Not a true Sparse Data Representation.",
                             type: SDR,    count: 0, isDefaultOutput: yes, isRegionLevel: yes }}
      } )");

  return ns;
}


DateEncoderRegion::DateEncoderRegion(const ValueMap &par, Region *region) : RegionImpl(region) {
  rnd_ = Random(42);
  spec_.reset(createSpec());
  ValueMap params = ValidateParameters(par, spec_.get());
    
  DateEncoderParameters args;
  args.season_width = params.getScalarT<UInt32>("season_width");
  args.season_radius = params.getScalarT<Real32>("season_radius");
  args.dayOfWeek_width = params.getScalarT<UInt32>("dayOfWeek_width");
  args.dayOfWeek_radius = params.getScalarT<Real32>("dayOfWeek_radius");
  args.weekend_width = params.getScalarT<UInt32>("weekend_width");
  args.holiday_width = params.getScalarT<UInt32>("holiday_width");
  args.timeOfDay_width = params.getScalarT<UInt32>("timeOfDay_width");
  args.timeOfDay_radius = params.getScalarT<Real32>("timeOfDay_radius");
  args.custom_width = params.getScalarT<UInt32>("custom_width");
  args.verbose = params.getScalarT<bool>("verbose");

  noise_ = params.getScalarT<Real32>("noise");
  sensedTime_ = static_cast<time_t>(params.getScalarT<Int64>("sensedTime"));

  // Parse holiday_dates.   expecting "[[m,d],[y,m,d],...]"
  args.holiday_dates.clear();
  std::string dates = params.getString("holiday_dates", spec_->parameters.getByName("holiday_dates").defaultValue);
  std::vector<int> date;
  char buffer[1000];
  memcpy(buffer, dates.c_str(), std::min(sizeof(buffer), dates.size()));
  buffer[dates.size()] = '\0';
  char *ptr = buffer;
  do {
    if (isdigit(*ptr)) {
      int i = strtol(ptr, &ptr, 10);
      date.push_back(i);
    } else if (*ptr++ == ']') {
      if (date.size() > 0) args.holiday_dates.push_back(date);
      date.clear();
    }
  } while (*ptr);

  //parse custom_days
  std::string days = params.getString("custom_days", "");
  args.custom_days = split(days, ',');

  encoder_ = std::make_shared<DateEncoder>(args);
}

DateEncoderRegion::DateEncoderRegion(ArWrapper &wrapper, Region *region)
    : RegionImpl(region) {
  cereal_adapter_load(wrapper);
}
DateEncoderRegion::~DateEncoderRegion() {}

void DateEncoderRegion::initialize() {}

Dimensions DateEncoderRegion::askImplForOutputDimensions(const std::string &name) {
  if (name == "encoded") {
    // get the dimensions determined by the encoder (comes from parameters.size).
    Dimensions encoderDim(encoder_->dimensions); // get dimensions from encoder
    return encoderDim;
  } else if (name == "bucket") {
    // get the dimensions determined by the number of attributes enabled in the encoder.
    Dimensions bucketDim(static_cast<UInt32>(encoder_->buckets.size()));
    return bucketDim;
  }  return RegionImpl::askImplForOutputDimensions(name);
}

void DateEncoderRegion::compute() {
  if (hasInput("values")) {
    Array &a = getInput("values")->getData();
    sensedTime_ = (time_t)((Int64 *)(a.getBuffer()))[0];
  }
  SDR &output = getOutput("encoded")->getData().getSDR();
  encoder_->encode(sensedTime_, output);

  // Add some noise.
  // noise_ = 0.01 means change 1% of the SDR for each iteration, this makes a random sequence, but seemingly stable
  if (noise_ != 0.0f)
    output.addNoise(noise_, rnd_);

  // get the bucket values for each attribute configured.
  Array &bucket_array = getOutput("bucket")->getData();
  Real64 *ptr = reinterpret_cast<Real64*>(bucket_array.getBuffer());
  for (size_t i = 0; i < encoder_->buckets.size(); i++) {
    ptr[i] = encoder_->buckets[i];
  }
}


void DateEncoderRegion::setParameterInt64(const std::string &name, Int64 index, Int64 value) {
  if (name == "sensedTime")  sensedTime_ = static_cast<time_t>(value);
  else  RegionImpl::setParameterInt64(name, index, value);
}
void DateEncoderRegion::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  if (name == "noise") noise_ = value;
  else RegionImpl::setParameterReal32(name, index, value);
}
void DateEncoderRegion::setParameterBool(const std::string &name, Int64 index, bool value) {
  if (name == "verbose")
    encoder_->setVerbose(value);
  else
    RegionImpl::setParameterBool(name, index, value);
}

Int64 DateEncoderRegion::getParameterInt64(const std::string &name, Int64 index) {
  if (name == "sensedTime") { return static_cast<Int64>(sensedTime_);}
  else return RegionImpl::getParameterInt64(name, index);
}

Real32 DateEncoderRegion::getParameterReal32(const std::string &name, Int64 index) {
  if (name == "season_radius")
    return encoder_->parameters.season_radius;
  else if (name == "dayOfWeek_radius")
    return encoder_->parameters.dayOfWeek_radius;
  else if (name == "timeOfDay_radius")
    return encoder_->parameters.timeOfDay_radius;
  else if (name == "noise")
    return noise_;
  else
    return RegionImpl::getParameterReal32(name, index);
}

UInt32 DateEncoderRegion::getParameterUInt32(const std::string &name, Int64 index) {
  if (name == "season_width")
    return encoder_->parameters.season_width;
  else if (name == "dayOfWeek_width")
    return encoder_->parameters.dayOfWeek_width;
  else if (name == "weekend_width")
    return encoder_->parameters.weekend_width;
  else if (name == "holiday_width")
    return encoder_->parameters.holiday_width;
  else if (name == "timeOfDay_width")
    return encoder_->parameters.timeOfDay_width;
  else if (name == "custom_width")
    return encoder_->parameters.custom_width;
  else if (name == "size")
    return encoder_->size;
  else
    return RegionImpl::getParameterUInt32(name, index);
}

bool DateEncoderRegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "verbose") return encoder_->parameters.verbose;
  else  return RegionImpl::getParameterBool(name, index);
}

std::string DateEncoderRegion::getParameterString(const std::string& name, Int64 index) {
  if (name == "custom_days") {
    std::string buffer;
    for (size_t i = 0; i < encoder_->parameters.custom_days.size(); i++) {
      if (i != 0)
        buffer += ",";
      buffer += encoder_->parameters.custom_days[i];
    }
    return buffer;
  } else if (name == "holiday_dates") {
    std::string buffer;
    if (encoder_->parameters.custom_days.size()) {
      buffer = "[";
      for (size_t i = 0; i < encoder_->parameters.holiday_dates.size(); i++) {
        if (i > 0)
          buffer += ",";
        buffer += "[";
        for (size_t j = 0; j < encoder_->parameters.holiday_dates[i].size(); j++) {
          if (j > 0)
            buffer += ",";
          buffer += std::to_string(encoder_->parameters.holiday_dates[i][j]);
        }
        buffer += "]";
      }
      buffer += "]";
    }
    return buffer;
  } else
    return RegionImpl::getParameterString(name, index);
}

bool DateEncoderRegion::operator==(const RegionImpl &other) const {
  if (other.getType() != "DateEncoderRegion") return false;
  const DateEncoderRegion& o = reinterpret_cast<const DateEncoderRegion &>(other);
  if (encoder_->parameters.season_width != o.encoder_->parameters.season_width) 
    return false;
  if (encoder_->parameters.season_radius != o.encoder_->parameters.season_radius)
    return false;
  if (encoder_->parameters.dayOfWeek_width != o.encoder_->parameters.dayOfWeek_width)
    return false;
  if (encoder_->parameters.dayOfWeek_radius != o.encoder_->parameters.dayOfWeek_radius)
    return false;
  if (encoder_->parameters.weekend_width != o.encoder_->parameters.weekend_width)
    return false;
  if (encoder_->parameters.holiday_width != o.encoder_->parameters.holiday_width)
    return false;
  if (encoder_->parameters.holiday_dates != o.encoder_->parameters.holiday_dates)
    return false;
  if (encoder_->parameters.timeOfDay_width != o.encoder_->parameters.timeOfDay_width)
    return false;
  if (encoder_->parameters.timeOfDay_radius != o.encoder_->parameters.timeOfDay_radius)
    return false;
  if (encoder_->parameters.custom_width != o.encoder_->parameters.custom_width)
    return false;
  if (encoder_->parameters.custom_days != o.encoder_->parameters.custom_days)
    return false;
  if (encoder_->parameters.verbose != o.encoder_->parameters.verbose)
    return false;
  if (noise_ != o.noise_)
    return false;
  if (sensedTime_ != o.sensedTime_)
    return false;

  return true;
}


} // namespace htm
