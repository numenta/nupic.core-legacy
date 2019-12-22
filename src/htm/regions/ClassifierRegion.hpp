/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, Numenta, Inc.
 *
 * Author: David Keeney, Nov. 2019   dkeeney@gmail.com
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
 * Defines ClassifierRegion, a Region implementation for the Classifier algorithm (SDRClassifier.cpp).
 */

#ifndef NTA_CLASSIFIERREGION_HPP
#define NTA_CLASSIFIERREGION_HPP

#include <string>
#include <vector>

#include <htm/engine/RegionImpl.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/algorithms/SDRClassifier.hpp>

namespace htm {
/**
 * A network region that encapsulates the Classifier algorithm.
 *
 * @b Description
 * Definitions for the SDR Classifier.
 *
 * `Classifier` learns mapping from SDR->input value (encoder's output).
 * This is used when you need to "explain" the HTM network back to real-world,
 * ie. mapping SDRs back to digits in MNIST digit classification task.
 *
 *
 */
class ClassifierRegion : public RegionImpl, Serializable {
public:
  ClassifierRegion(const ValueMap &params, Region *region);
  ClassifierRegion(ArWrapper &wrapper, Region *region);

  virtual ~ClassifierRegion() override;

  static Spec *createSpec();

  virtual bool getParameterBool(const std::string &name,   Int64 index = -1) override;
  virtual void setParameterBool(const std::string &name, Int64 index, bool value) override;

  virtual void initialize() override;

  void compute() override;

  virtual Dimensions askImplForOutputDimensions(const std::string &name) override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive> void save_ar(Archive &ar) const {
    ar(cereal::make_nvp("learn", learn_));
    ar(cereal::make_nvp("bucketListMap", bucketListMap));
    ar(cereal::make_nvp("bucketList", bucketList));
    ar(cereal::make_nvp("classifier", classifier_));
  }
  // FOR Cereal Deserialization
  // NOTE: the Region Implementation must have been allocated
  //       using the RegionImplFactory so that it is connected
  //       to the Network and Region objects. This will populate
  //       the region_ field in the Base class.
  template<class Archive>
  void load_ar(Archive& ar) {
    ar(cereal::make_nvp("learn", learn_));
    ar(cereal::make_nvp("bucketListMap", bucketListMap));
    ar(cereal::make_nvp("bucketList", bucketList));
    ar(cereal::make_nvp("classifier", classifier_));
  }


  bool operator==(const RegionImpl &other) const override;
  inline bool operator!=(const ClassifierRegion &other) const {
    return !operator==(other);
  }

private:
  std::shared_ptr<Classifier> classifier_;
  bool learn_;

  std::map<Real64, UInt32> bucketListMap;  //  Map containing titles or buckets ordered by quantized values.
  std::vector<Real64> bucketList;          //  Vector of titles ordered by order in which they were first seen to match Classifier.
};
} // namespace htm

#endif // NTA_CLASSIFIERREGION_HPP
