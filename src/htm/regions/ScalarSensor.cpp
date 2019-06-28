/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * Implementation of the ScalarSensor Region
 */

#include <htm/regions/ScalarSensor.hpp>

#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/utils/Log.hpp>

namespace htm {

ScalarSensor::ScalarSensor(const ValueMap &params, Region *region)
    : RegionImpl(region) {
  params_.size = params.getScalarT<UInt32>("n");
  params_.activeBits = params.getScalarT<UInt32>("w");
  params_.resolution = params.getScalarT<Real64>("resolution");
  params_.radius = params.getScalarT<Real64>("radius");
  params_.minimum = params.getScalarT<Real64>("minValue");
  params_.maximum = params.getScalarT<Real64>("maxValue");
  params_.periodic = params.getScalarT<bool>("periodic");
  params_.clipInput = params.getScalarT<bool>("clipInput");

  encoder_ = std::make_shared<ScalarEncoder>( params_ );


  sensedValue_ = params.getScalarT<Real64>("sensedValue");
}

ScalarSensor::ScalarSensor(ArWrapper &wrapper, Region *region):RegionImpl(region) {
  cereal_adapter_load(wrapper);
}


void ScalarSensor::initialize() {
  // Normally a region will create the algorithm here, at the point
  // when the dimensions and parameters are known. But in this case
  // it is the encoder that determines the dimensions so it must be
  // allocated in the constructor.  
  // If parameters are changed after the encoder is instantiated 
  // then the encoder's initialization function must be called again 
  // to reset the dimensions in the BaseEncoder.  This is done in
  // askImplForOutputDimensions()
}


Dimensions ScalarSensor::askImplForOutputDimensions(const std::string &name) {

  if (name == "encoded") {
    // just in case parameters changed since instantiation, we call the
    // encoder's initialize() again. Note that if dimensions have been manually set, 
    // use those if same number of elements, else the dimensions are determined 
    // only by the encoder's algorithm.
    encoder_->initialize(params_); 

    // get the dimensions determined by the encoder.
    Dimensions encDim(encoder_->dimensions); // get dimensions from encoder
    Dimensions regionDim = getDimensions();  // get the region level dimensions.
    if (regionDim.isSpecified()) {
      // region level dimensions were explicitly specified.
      NTA_CHECK(regionDim.getCount() == encDim.getCount()) 
        << "Manually set dimensions are incompatible with encoder parameters; region: " 
        << regionDim << "  encoder: " << encDim;
      encDim = regionDim;
    }
    setDimensions(encDim);  // This output is 'isRegionLevel' so set region level dimensions.
    return encDim;
  } 
  else if (name == "bucket") {
    return 1;
  }
  // for any other output name, let RegionImpl handle it.
  return RegionImpl::askImplForOutputDimensions(name);
}

std::string ScalarSensor::executeCommand(const std::vector<std::string> &args,
                                         Int64 index) {
  NTA_THROW << "ScalarSensor::executeCommand -- commands not supported";
}

void ScalarSensor::compute()
{
  SDR &output = getOutput("encoded")->getData().getSDR();
  encoder_->encode((Real64)sensedValue_, output);
}

ScalarSensor::~ScalarSensor() {}

/* static */ Spec *ScalarSensor::createSpec() {
  auto ns = new Spec;

  ns->singleNodeOnly = true;

  /* ----- parameters ----- */
  ns->parameters.add("sensedValue",
                     ParameterSpec("Scalar input", NTA_BasicType_Real64,
                                   1,    // elementCount
                                   "",   // constraints
                                   "-1", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("n", ParameterSpec("The length of the encoding. Size of buffer",
                                        NTA_BasicType_UInt32,
                                        1,   // elementCount
                                        "",  // constraints
                                        "0", // defaultValue
                                        ParameterSpec::ReadWriteAccess));

  ns->parameters.add("w",
                     ParameterSpec("The number of active bits in the encoding. i.e. how sparse",
                                   NTA_BasicType_UInt32,
                                   1,   // elementCount
                                   "",  // constraints
                                   "0", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("resolution",
                     ParameterSpec("The resolution for the encoder",
                                   NTA_BasicType_Real64,
                                   1,   // elementCount
                                   "",  // constraints
                                   "0", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("radius", ParameterSpec("The radius for the encoder",
                                  NTA_BasicType_Real64,
                                  1,   // elementCount
                                  "",  // constraints
                                  "0", // defaultValue
                                  ParameterSpec::ReadWriteAccess));

  ns->parameters.add("minValue",
                     ParameterSpec("The minimum value for the input",
                                   NTA_BasicType_Real64,
                                   1,    // elementCount
                                   "",   // constraints
                                   "-1.0", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("maxValue",
                     ParameterSpec("The maximum value for the input",
                                   NTA_BasicType_Real64,
                                   1,    // elementCount
                                   "",   // constraints
                                   "+1.0", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("periodic",
                     ParameterSpec("Whether the encoder is periodic",
                                   NTA_BasicType_Bool,
                                   1,       // elementCount
                                   "",      // constraints
                                   "false", // defaultValue
                                   ParameterSpec::ReadWriteAccess));

  ns->parameters.add("clipInput",
                    ParameterSpec(
                                  "Whether to clip inputs if they're outside [minValue, maxValue]",
                                  NTA_BasicType_Bool,
                                  1,       // elementCount
                                  "",      // constraints
                                  "false", // defaultValue
                                  ParameterSpec::ReadWriteAccess));

  /* ----- outputs ----- */

  ns->outputs.add("encoded", OutputSpec("Encoded value", NTA_BasicType_SDR,
                                        0,    // elementCount
                                        true, // isRegionLevel
                                        true  // isDefaultOutput
                                        ));

  ns->outputs.add("bucket", OutputSpec("Bucket number for this sensedValue",
                                       NTA_BasicType_Int32,
                                       0,    // elementCount
                                       true, // isRegionLevel
                                       false // isDefaultOutput
                                       ));

  return ns;
}

Real64 ScalarSensor::getParameterReal64(const std::string &name, Int64 index) {
  if (name == "sensedValue") {
    return sensedValue_;
  }
  else {
    return RegionImpl::getParameterReal64(name, index);
  }
}

UInt32 ScalarSensor::getParameterUInt32(const std::string &name, Int64 index) {
  if (name == "n") {
    return (UInt32)encoder_->size;
  }
  else {
    return RegionImpl::getParameterUInt32(name, index);
  }
}

void ScalarSensor::setParameterReal64(const std::string &name, Int64 index, Real64 value) {
  if (name == "sensedValue") {
    sensedValue_ = value;
  } else {
	  RegionImpl::setParameterReal64(name, index, value);
  }
}

bool ScalarSensor::operator==(const RegionImpl &o) const {
  if (o.getType() != "ScalarSensor") return false;
  ScalarSensor& other = (ScalarSensor&)o;
  if (params_.minimum != other.params_.minimum) return false;
  if (params_.maximum != other.params_.maximum) return false;
  if (params_.clipInput != other.params_.clipInput) return false;
  if (params_.periodic != other.params_.periodic) return false;
  if (params_.activeBits != other.params_.activeBits) return false;
  if (params_.sparsity != other.params_.sparsity) return false;
  if (params_.size != other.params_.size) return false;
  if (params_.radius != other.params_.radius) return false;
  if (params_.resolution != other.params_.resolution) return false;
  if (sensedValue_ != other.sensedValue_) return false;

  return true;
}


} // namespace htm
