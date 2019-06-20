/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Author: David Keeney, June, 2018  (ported from Python)
 * Copyright (C) 2018, Numenta, Inc.
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
 * Declarations for TMRegion class
 */

//----------------------------------------------------------------------

#ifndef NTA_TMREGION_HPP
#define NTA_TMREGION_HPP

#include <htm/engine/RegionImpl.hpp>
#include <htm/algorithms/TemporalMemory.hpp>

#include <htm/ntypes/Value.hpp>
//----------------------------------------------------------------------

namespace htm {
class TMRegion : public RegionImpl, Serializable {
  typedef void (*computeCallbackFunc)(const std::string &);
  typedef std::map<std::string, Spec> SpecMap;

public:
  TMRegion() = delete;
  TMRegion(const TMRegion &) = delete;
  TMRegion(const ValueMap &params, Region *region);
  TMRegion(ArWrapper& wrapper, Region *region);
  virtual ~TMRegion();

  /* -----------  Required RegionImpl Interface methods ------- */

  // Used by RegionImplFactory to create and cache
  // a nodespec. Ownership is transferred to the caller.
  static Spec *createSpec();

  std::string getNodeType() { return "TMRegion"; };

  // Compute outputs from inputs and internal state
  void compute() override;

  /**
   * Inputs/Outputs are made available in initialize()
   * It is always called after the constructor (or load from serialized state)
   */
  void initialize() override;

  CerealAdapter;  // see Serializable.hpp
  // FOR Cereal Serialization
  template<class Archive>
  void save_ar(Archive& ar) const {
    bool init = ((tm_) ? true : false);
    ar(cereal::make_nvp("numberOfCols", args_.numberOfCols));
    ar(cereal::make_nvp("cellsPerColumn", args_.cellsPerColumn));
    ar(cereal::make_nvp("activationThreshold", args_.activationThreshold));
    ar(cereal::make_nvp("initialPermanence", args_.initialPermanence));
    ar(cereal::make_nvp("connectedPermanence", args_.connectedPermanence));
    ar(cereal::make_nvp("maxNewSynapseCount", args_.maxNewSynapseCount));
    ar(cereal::make_nvp("permanenceIncrement", args_.permanenceIncrement));
    ar(cereal::make_nvp("permanenceDecrement", args_.permanenceDecrement));
    ar(cereal::make_nvp("predictedSegmentDecrement", args_.predictedSegmentDecrement));
    ar(cereal::make_nvp("seed", args_.seed));
    ar(cereal::make_nvp("maxSegmentsPerCell", args_.maxSegmentsPerCell));
    ar(cereal::make_nvp("maxSynapsesPerSegment", args_.maxSynapsesPerSegment));
    ar(cereal::make_nvp("externalPredictiveInputs", args_.externalPredictiveInputs));
    ar(cereal::make_nvp("checkInputs", args_.checkInputs));
    ar(cereal::make_nvp("learningMode", args_.learningMode));
    ar(cereal::make_nvp("sequencePos", args_.sequencePos));
    ar(cereal::make_nvp("iter", args_.iter));
    ar(cereal::make_nvp("orColumnOutputs", args_.orColumnOutputs));
    ar(cereal::make_nvp("init", init));
    if (init) {
      // Save the algorithm state
      ar(cereal::make_nvp("TM", tm_));
    }
  }

  // FOR Cereal Deserialization
  template<class Archive>
  void load_ar(Archive& ar) {
    bool init = false;
    ar(cereal::make_nvp("numberOfCols", args_.numberOfCols));
    ar(cereal::make_nvp("cellsPerColumn", args_.cellsPerColumn));
    ar(cereal::make_nvp("activationThreshold", args_.activationThreshold));
    ar(cereal::make_nvp("initialPermanence", args_.initialPermanence));
    ar(cereal::make_nvp("connectedPermanence", args_.connectedPermanence));
    ar(cereal::make_nvp("maxNewSynapseCount", args_.maxNewSynapseCount));
    ar(cereal::make_nvp("permanenceIncrement", args_.permanenceIncrement));
    ar(cereal::make_nvp("permanenceDecrement", args_.permanenceDecrement));
    ar(cereal::make_nvp("predictedSegmentDecrement", args_.predictedSegmentDecrement));
    ar(cereal::make_nvp("seed", args_.seed));
    ar(cereal::make_nvp("maxSegmentsPerCell", args_.maxSegmentsPerCell));
    ar(cereal::make_nvp("maxSynapsesPerSegment", args_.maxSynapsesPerSegment));
    ar(cereal::make_nvp("externalPredictiveInputs", args_.externalPredictiveInputs));
    ar(cereal::make_nvp("checkInputs", args_.checkInputs));
    ar(cereal::make_nvp("learningMode", args_.learningMode));
    ar(cereal::make_nvp("sequencePos", args_.sequencePos));
    ar(cereal::make_nvp("iter", args_.iter));
    ar(cereal::make_nvp("orColumnOutputs", args_.orColumnOutputs));
    ar(cereal::make_nvp("init", init));

    args_.outputWidth = (args_.orColumnOutputs)?args_.numberOfCols
                      : (args_.numberOfCols * args_.cellsPerColumn);
    if (init) {
      // Restore algorithm state
      ar(cereal::make_nvp("TM", tm_));
    }
  }

  bool operator==(const RegionImpl &other) const override;
  inline bool operator!=(const TMRegion &other) const {
    return !operator==(other);
  }

  // Per-node size (in elements) of the given output.
  // For per-region outputs, it is the total element count.
  // This method is called only for outputs whose size is not
  // specified in the spec.
  Dimensions askImplForOutputDimensions(const std::string &name) override;



  /* -----------  Optional RegionImpl Interface methods ------- */
  UInt32 getParameterUInt32(const std::string &name, Int64 index) override;
  Int32 getParameterInt32(const std::string &name, Int64 index) override;
  Real32 getParameterReal32(const std::string &name, Int64 index) override;
  bool getParameterBool(const std::string &name, Int64 index) override;
  std::string getParameterString(const std::string &name, Int64 index) override;

  void setParameterUInt32(const std::string &name, Int64 index,UInt32 value) override;
  void setParameterInt32(const std::string &name, Int64 index,Int32 value) override;
  void setParameterReal32(const std::string &name, Int64 index,Real32 value) override;
  void setParameterBool(const std::string &name, Int64 index,bool value) override;
  void setParameterString(const std::string &name, Int64 index, const std::string &s) override;

private:
  Dimensions columnDimensions_;

  // Note: to avoid deserialization problems due to differences in 
  //       how compilers deal with structure padding, do not allow
  //       any member to span an 64bit (8byte) boundary.
  struct {
    UInt32 numberOfCols;
    UInt32 cellsPerColumn;
    UInt32 activationThreshold;
    Real32 initialPermanence;
    Real32 connectedPermanence;
    UInt32 minThreshold;
    UInt32 maxNewSynapseCount;
    Real32 permanenceIncrement;
    Real32 permanenceDecrement;
    Real32 predictedSegmentDecrement;
    Int32 seed;
    Int32 maxSegmentsPerCell;
    Int32 maxSynapsesPerSegment;
    UInt32 externalPredictiveInputs;
    bool checkInputs;

    // parameters used by this class and not passed on
    bool learningMode;
    bool orColumnOutputs;

    // some local variables
    UInt32 padding; // to prevent the next field from spanning 8 byte boundary.
    UInt32 outputWidth; // columnCount *cellsPerColumn
    UInt32 sequencePos;
    Size iter;
  } args_;



  computeCallbackFunc computeCallback_;
  std::unique_ptr<TemporalMemory> tm_;
};

} // namespace htm

#endif // NTA_SPREGION_HPP
