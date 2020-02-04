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
 * Defines the ScalarSensor
 */

#ifndef NTA_SCALAR_SENSOR_HPP
#define NTA_SCALAR_SENSOR_HPP

#include <string>
#include <vector>

#include <htm/engine/RegionImpl.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/encoders/ScalarEncoder.hpp>

namespace htm {
/**
 * A network region that encapsulates the ScalarEncoder.
 *
 * @b Description
 * A ScalarSensor encapsulates ScalarEncoders, connecting them to the Network
 * API. As a network runs, the client will specify new encoder inputs by
 * setting the "sensedValue" parameter. On each compute, the ScalarSensor will
 * encode its "sensedValue" to output.
 */
class ScalarSensor : public RegionImpl, Serializable {
public:
  ScalarSensor(const ValueMap &params, Region *region);
  ScalarSensor(ArWrapper& wrapper, Region *region);

  virtual ~ScalarSensor() override;

  static Spec *createSpec();

  virtual Real64 getParameterReal64(const std::string &name, Int64 index = -1) override;
  virtual UInt32 getParameterUInt32(const std::string &name, Int64 index = -1) override;
  virtual bool getParameterBool(const std::string &name, Int64 index = -1) override;
  virtual void setParameterReal64(const std::string &name, Int64 index, Real64 value) override;
  virtual void initialize() override;

  void compute() override;
  virtual std::string executeCommand(const std::vector<std::string> &args,
                                     Int64 index) override;

  virtual Dimensions askImplForOutputDimensions(const std::string &name) override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    ar(CEREAL_NVP(sensedValue_));
    ar(cereal::make_nvp("minimum", params_.minimum),
       cereal::make_nvp("maximum", params_.maximum),
       cereal::make_nvp("clipInput", params_.clipInput),
       cereal::make_nvp("periodic", params_.periodic),
       cereal::make_nvp("activeBits", params_.activeBits),
       cereal::make_nvp("sparsity", params_.sparsity),
       cereal::make_nvp("size", params_.size),
       cereal::make_nvp("radius", params_.radius),
       cereal::make_nvp("resolution", params_.resolution),
       cereal::make_nvp("sensedValue_", sensedValue_));
  }
  // FOR Cereal Deserialization
  // NOTE: the Region Implementation must have been allocated
  //       using the RegionImplFactory so that it is connected
  //       to the Network and Region objects. This will populate
  //       the region_ field in the Base class.
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(CEREAL_NVP(sensedValue_));
    ar(cereal::make_nvp("minimum", params_.minimum),
       cereal::make_nvp("maximum", params_.maximum),
       cereal::make_nvp("clipInput", params_.clipInput),
       cereal::make_nvp("periodic", params_.periodic),
       cereal::make_nvp("activeBits", params_.activeBits),
       cereal::make_nvp("sparsity", params_.sparsity),
       cereal::make_nvp("size", params_.size),
       cereal::make_nvp("radius", params_.radius),
       cereal::make_nvp("resolution", params_.resolution),
       cereal::make_nvp("sensedValue_", sensedValue_));
    encoder_ = std::make_shared<ScalarEncoder>( params_ );
    setDimensions(encoder_->dimensions); 
  }


  bool operator==(const RegionImpl &other) const override;
  inline bool operator!=(const ScalarSensor &other) const {
    return !operator==(other);
  }

private:
  Real64 sensedValue_;
  ScalarEncoderParameters params_;

  std::shared_ptr<ScalarEncoder> encoder_;
};
} // namespace htm

#endif // NTA_SCALAR_SENSOR_HPP
