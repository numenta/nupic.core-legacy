/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * Author: David Keeney, 2019   dkeeney@gmail.com
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
 * Defines RDSERegion, a Region implementation for the RandomDistributedScalarEncoder.
 */

#ifndef NTA_RDSEREGION_HPP
#define NTA_RDSEREGION_HPP

#include <string>
#include <vector>

#include <htm/engine/RegionImpl.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/encoders/RandomDistributedScalarEncoder.hpp>

namespace htm {
/**
 * A network region that encapsulates the RandomDistributedScalarEncoder.
 *
 * @b Description
 * A RDSERegion encapsulates RandomDistributedScalarEncoder, connecting it to the Network
 * API. As a network runs, the client will specify new encoder inputs by
 * setting the "sensedValue" parameter or connecting a link which provides values for "sensedValue". 
 * On each compute, the ScalarSensor will encode its "sensedValue" to output.
 */
class RDSERegion : public RegionImpl, Serializable {
public:
  RDSERegion(const ValueMap &params, Region *region);
  RDSERegion(ArWrapper &wrapper, Region *region);

  virtual ~RDSERegion() override;

  static Spec *createSpec();

  virtual Real64 getParameterReal64(const std::string &name, Int64 index = -1) override;
  virtual Real32 getParameterReal32(const std::string &name, Int64 index = -1) override;
  virtual UInt32 getParameterUInt32(const std::string &name, Int64 index = -1) override;
  virtual bool getParameterBool(const std::string &name,   Int64 index = -1) override;
  virtual void setParameterReal64(const std::string &name, Int64 index,  Real64 value) override;
  virtual void initialize() override;

  void compute() override;

  virtual Dimensions askImplForOutputDimensions(const std::string &name) override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    ar(CEREAL_NVP(sensedValue_));
    ar(cereal::make_nvp("encoder", encoder_));
  }
  // FOR Cereal Deserialization
  // NOTE: the Region Implementation must have been allocated
  //       using the RegionImplFactory so that it is connected
  //       to the Network and Region objects. This will populate
  //       the region_ field in the Base class.
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(CEREAL_NVP(sensedValue_));
    ar(cereal::make_nvp("encoder", encoder_));
    setDimensions(encoder_->dimensions); 
  }


  bool operator==(const RDSERegion &other) const;
  inline bool operator!=(const RDSERegion &other) const {
    return !operator==(other);
  }

private:
  Real64 sensedValue_;
  std::shared_ptr<RandomDistributedScalarEncoder> encoder_;
};
} // namespace htm

#endif // NTA_RDSEREGION_HPP
