/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * Author: David Keeney, Nov. 2019
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

#include <htm/regions/RDSERegion.hpp>

#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include "htm/utils/Random.hpp"
#include <htm/utils/Log.hpp>

#include <memory>

namespace htm {


/* static */ Spec *RDSERegion::createSpec() {
  Spec *ns = new Spec();
  ns->parseSpec(R"(
  {name: "RDSERegion",
      parameters: {
          size:        {type: UInt32, default: "0"},
          activeBits:  {type: UInt32, default: "0"},
          sparsity:    {type: Real32, default: "0.0"},
          radius:      {type: Real32, default: "0.0"},
          resolution:  {type: Real32, default: "0.0"},
          category:    {type: Bool,   default: "false"},
          seed:        {type: UInt32, default: "0"},
          noise:       {description: "amount of noise to add to the output SDR. 0.01 is 1%",
                        type: Real32, default: "0.0", access: ReadWrite },
          sensedValue: {description: "The value to encode. Overriden by input 'values'.",
                        type: Real64, default: "0.0", access: ReadWrite }},
      inputs: {
          values:      {description: "Values to encode. Overrides sensedValue.",
                        type: Real64, count: 1, isDefaultInput: yes, isRegionLevel: yes}}, 
      outputs: {
          bucket:      {description: "Quantized sample based on the radius. Becomes the title for this sample in Classifier.",
                        type: Real64, count: 1, isDefaultOutput: false, isRegionLevel: false },
          encoded:     {description: "Encoded bits. Not a true Sparse Data Representation (SP does that).",
    type: SDR,    count: 0, isDefaultOutput: yes, isRegionLevel: yes }}
  } )");

  return ns;
}


RDSERegion::RDSERegion(const ValueMap &par, Region *region) : RegionImpl(region) {
  rnd_ = Random(42);
  spec_.reset(createSpec());
  ValueMap params = ValidateParameters(par, spec_.get());
    
  RDSE_Parameters args;
  args.size =       params.getScalarT<UInt32>("size");
  args.activeBits = params.getScalarT<UInt32>("activeBits");
  args.sparsity =   params.getScalarT<Real32>("sparsity");
  args.radius =     params.getScalarT<Real32>("radius");
  args.resolution = params.getScalarT<Real32>("resolution");
  args.category =   params.getScalarT<bool>("category");
  args.seed =       params.getScalarT<UInt32>("seed");

  encoder_ = std::make_shared<RandomDistributedScalarEncoder>(args);
  sensedValue_ = params.getScalarT<Real64>("sensedValue");
  noise_ = params.getScalarT<Real32>("noise");
}

RDSERegion::RDSERegion(ArWrapper &wrapper, Region *region)
    : RegionImpl(region) {
  cereal_adapter_load(wrapper);
}
RDSERegion::~RDSERegion() {}

void RDSERegion::initialize() { }

Dimensions RDSERegion::askImplForOutputDimensions(const std::string &name) {
  if (name == "encoded") {
    // get the dimensions determined by the encoder (comes from parameters.size).
    Dimensions encoderDim(encoder_->dimensions); // get dimensions from encoder
    return encoderDim;
  }  return RegionImpl::askImplForOutputDimensions(name);
}

void RDSERegion::compute() {
  if (hasInput("values")) {
    Array &a = getInput("values")->getData();
    sensedValue_ = ((Real64 *)(a.getBuffer()))[0];
  }
  if (!std::isfinite(sensedValue_))
    sensedValue_ = 0;  // prevents an exception in case of nan or inf
  //std::cout << "RDSERegion compute() sensedValue=" << sensedValue_ << std::endl;

  SDR &output = getOutput("encoded")->getData().getSDR();
  encoder_->encode((Real64)sensedValue_, output);

  // Add some noise.
  // noise_ = 0.01 means change 1% of the SDR for each iteration, this makes a random sequence, but seemingly stable
  if (noise_ != 0.0f)
    output.addNoise(noise_, rnd_);
    
  // output a quantized value (for use by Classifier)
  // This is a quantification of the data being encoded (the sample) 
  // and becomes the title in the Classifier.
  if (encoder_->parameters.radius != 0.0f) {
    Real64 *buf = (Real64 *)getOutput("bucket")->getData().getBuffer();
    buf[0] = sensedValue_ - std::fmod(sensedValue_, encoder_->parameters.radius);
    //std::cout << "RDSERegion compute() bucket=" << buf[0] << std::endl;
  }
  
}


void RDSERegion::setParameterReal64(const std::string &name, Int64 index, Real64 value) {
  if (name == "sensedValue")  sensedValue_ = value;
  else  RegionImpl::setParameterReal64(name, index, value);
}
void RDSERegion::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  if (name == "noise") noise_ = value;
  else RegionImpl::setParameterReal32(name, index, value);
}

Real64 RDSERegion::getParameterReal64(const std::string &name, Int64 index) {
  if (name == "sensedValue") { return sensedValue_;}
  else return RegionImpl::getParameterReal64(name, index);
}

Real32 RDSERegion::getParameterReal32(const std::string &name, Int64 index) {
  if (name == "resolution")    return encoder_->parameters.resolution;
  else if (name == "noise")    return noise_;
  else if (name == "radius")   return encoder_->parameters.radius;
  else if (name == "sparsity") return encoder_->parameters.sparsity;
  else return RegionImpl::getParameterReal32(name, index);
}

UInt32 RDSERegion::getParameterUInt32(const std::string &name, Int64 index) {
  if (name == "size")            return encoder_->parameters.size;
  else if (name == "activeBits") return encoder_->parameters.activeBits;
  else if (name == "seed")       return encoder_->parameters.seed;
  else return RegionImpl::getParameterUInt32(name, index);
}

bool RDSERegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "category") return encoder_->parameters.category;
  else  return RegionImpl::getParameterBool(name, index);
}

bool RDSERegion::operator==(const RegionImpl &other) const {
  if (other.getType() != "RDSERegion") return false;
  const RDSERegion &o = reinterpret_cast<const RDSERegion&>(other);
  if (encoder_->parameters.size != o.encoder_->parameters.size) 
    return false;
  if (encoder_->parameters.activeBits != o.encoder_->parameters.activeBits)
    return false;
  if (encoder_->parameters.sparsity != o.encoder_->parameters.sparsity)
    return false;
  if (encoder_->parameters.radius != o.encoder_->parameters.radius)
    return false;
  if (encoder_->parameters.resolution != o.encoder_->parameters.resolution)
    return false;
  if (encoder_->parameters.category != o.encoder_->parameters.category)
    return false;
  if (encoder_->parameters.seed != o.encoder_->parameters.seed)
    return false;
  if (sensedValue_ != o.sensedValue_) return false;

  return true;
}


} // namespace htm
