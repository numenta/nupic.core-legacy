/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2020, Numenta, Inc.
 *
 * Author: David Keeney, Jan 2020   dkeeney@gmail.com
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
 * Defines DateEncoderRegion, a Region implementation for the DateEncoder algorithm.
 */

#ifndef NTA_DATE_ENCODER_REGION_HPP
#define NTA_DATE_ENCODER_REGION_HPP

#include <string>
#include <vector>

#include <htm/engine/RegionImpl.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/encoders/DateEncoder.hpp>

namespace htm {
/**
 * A network region that encapsulates the DateEncoder.
 *
 * @b Description
 * A DateEncoderRegion encapsulates DateEncoder, connecting it to the Network
 * API. As a network runs, the client will specify new encoder inputs by
 * setting the "sensedTime" parameter or connecting a link which provides values for "sensedTime". 
 * On each compute, the DateEncoder will encode its "sensedTime" to output.
 */
class DateEncoderRegion : public RegionImpl, Serializable {
public:
  DateEncoderRegion(const ValueMap &params, Region *region);
  DateEncoderRegion(ArWrapper &wrapper, Region *region);

  virtual ~DateEncoderRegion() override;

  static Spec *createSpec();

  virtual Int64 getParameterInt64(const std::string &name, Int64 index = -1) override;
  virtual Real32 getParameterReal32(const std::string &name, Int64 index = -1) override;
  virtual UInt32 getParameterUInt32(const std::string &name, Int64 index = -1) override;
  virtual bool getParameterBool(const std::string &name,   Int64 index = -1) override;
  virtual std::string getParameterString(const std::string &name, Int64 index) override;
  virtual void setParameterReal32(const std::string &name, Int64 index, Real32 value) override;
  virtual void setParameterInt64(const std::string &name, Int64 index, Int64 value) override;
  virtual void setParameterBool(const std::string &name, Int64 index, bool value) override;
  virtual void initialize() override;

  void compute() override;

  virtual Dimensions askImplForOutputDimensions(const std::string &name) override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    ar(CEREAL_NVP(sensedTime_));
    ar(CEREAL_NVP(noise_));
    ar(CEREAL_NVP(rnd_));
    ar(cereal::make_nvp("encoder", encoder_));
  }
  // FOR Cereal Deserialization
  // NOTE: the Region Implementation must have been allocated
  //       using the RegionImplFactory so that it is connected
  //       to the Network and Region objects. This will populate
  //       the region_ field in the Base class.
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(CEREAL_NVP(sensedTime_));
    ar(CEREAL_NVP(noise_));
    ar(CEREAL_NVP(rnd_));
    ar(cereal::make_nvp("encoder", encoder_));
    setDimensions(encoder_->dimensions); 
  }


  bool operator==(const RegionImpl &other) const override;
  inline bool operator!=(const DateEncoderRegion &other) const {
    return !operator==(other);
  }

private:
  time_t sensedTime_;
  Real32 noise_;
  Random rnd_;
  std::shared_ptr<DateEncoder> encoder_;
};
} // namespace htm

#endif // NTA_DATE_ENCODER_REGION_HPP
