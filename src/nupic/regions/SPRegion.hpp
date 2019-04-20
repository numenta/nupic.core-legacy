/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
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
 *
 * http://numenta.org/licenses/
 *
 * Author: David Keeney, Ported from Python April, 2018
 * ---------------------------------------------------------------------
 */

/** @file
 * Declarations for SPRegion class
 */

//----------------------------------------------------------------------

#ifndef NTA_SPREGION_HPP
#define NTA_SPREGION_HPP

#include <memory> //unique_ptr

#include <nupic/engine/RegionImpl.hpp>
#include <nupic/algorithms/SpatialPooler.hpp>
//----------------------------------------------------------------------


namespace nupic
{


class SPRegion  : public RegionImpl, Serializable
{		
  public:
    SPRegion(const ValueMap& params, Region *region);
    SPRegion(BundleIO& bundle, Region* region);
    SPRegion(ArWrapper& wrapper, Region *region) : RegionImpl(region) {
      // TODO:cereal  complete.
    }
    virtual ~SPRegion();


/* -----------  Required RegionImpl Interface methods ------- */

    // Used by RegionImplFactory to create and cache
    // a nodespec. Ownership is transferred to the caller.
    static Spec* createSpec();

    std::string getNodeType() { return "SPRegion"; };

    // Compute outputs from inputs and internal state
    void compute() override;
    std::string executeCommand(const std::vector<std::string>& args, Int64 index) override;

    /**
    * Inputs/Outputs are made available in initialize()
    * Region Impls are created at that time.
    */
    void initialize() override;

    void serialize(BundleIO& bundle) override;
    void deserialize(BundleIO& bundle) override;
		

		CerealAdapter;  // see Serializable.hpp
	  // FOR Cereal Serialization
	  template<class Archive>
	  void save_ar(Archive& ar) const {
	    bool init = ((sp_) ? true : false);
	    ar(cereal::make_nvp("inputWidth", args_.inputWidth));
	    ar(cereal::make_nvp("columnCount", args_.columnCount));
	    ar(cereal::make_nvp("potentialRadius", args_.potentialRadius));
	    ar(cereal::make_nvp("potentialPct", args_.potentialPct));
	    ar(cereal::make_nvp("globalInhibition", args_.globalInhibition));
	    ar(cereal::make_nvp("localAreaDensity", args_.localAreaDensity));
	    ar(cereal::make_nvp("numActiveColumnsPerInhArea", args_.numActiveColumnsPerInhArea));
	    ar(cereal::make_nvp("stimulusThreshold", args_.stimulusThreshold));
	    ar(cereal::make_nvp("synPermInactiveDec", args_.synPermInactiveDec));
	    ar(cereal::make_nvp("synPermActiveInc", args_.synPermActiveInc));
	    ar(cereal::make_nvp("synPermConnected", args_.synPermConnected));
	    ar(cereal::make_nvp("minPctOverlapDutyCycles", args_.minPctOverlapDutyCycles));
	    ar(cereal::make_nvp("dutyCyclePeriod", args_.dutyCyclePeriod));
	    ar(cereal::make_nvp("boostStrength", args_.boostStrength));
	    ar(cereal::make_nvp("seed", args_.seed));
	    ar(cereal::make_nvp("spVerbosity", args_.spVerbosity));
	    ar(cereal::make_nvp("wrapAround", args_.wrapAround));
	    ar(cereal::make_nvp("learningMode", args_.learningMode));
	    ar(cereal::make_nvp("init", init));
	    if (init) {
	      // save the output buffers
	      // The output buffers are saved as part of the Region Implementation.
	      cereal::size_type numBuffers = 0;
	      std::map<std::string, Output *> outputs = region_->getOutputs();
	      numBuffers = outputs.size();
	      ar(cereal::make_nvp("outputs", cereal::make_size_tag(numBuffers)));
	      for (auto iter : outputs) {
	        const Array &outputBuffer = iter.second->getData();
	        ar(cereal::make_map_item(iter.first, outputBuffer));
	      }
	      // Save the algorithm state
	      ar(cereal::make_nvp("SP", sp_));
	    }
		}

	  // FOR Cereal Deserialization
	  template<class Archive>
	  void load_ar(Archive& ar) {
	    bool init;
	    ar(cereal::make_nvp("inputWidth", args_.inputWidth));
	    ar(cereal::make_nvp("columnCount", args_.columnCount));
	    ar(cereal::make_nvp("potentialRadius", args_.potentialRadius));
	    ar(cereal::make_nvp("potentialPct", args_.potentialPct));
	    ar(cereal::make_nvp("globalInhibition", args_.globalInhibition));
	    ar(cereal::make_nvp("localAreaDensity", args_.localAreaDensity));
	    ar(cereal::make_nvp("numActiveColumnsPerInhArea", args_.numActiveColumnsPerInhArea));
	    ar(cereal::make_nvp("stimulusThreshold", args_.stimulusThreshold));
	    ar(cereal::make_nvp("synPermInactiveDec", args_.synPermInactiveDec));
	    ar(cereal::make_nvp("synPermActiveInc", args_.synPermActiveInc));
	    ar(cereal::make_nvp("synPermConnected", args_.synPermConnected));
	    ar(cereal::make_nvp("minPctOverlapDutyCycles", args_.minPctOverlapDutyCycles));
	    ar(cereal::make_nvp("dutyCyclePeriod", args_.dutyCyclePeriod));
	    ar(cereal::make_nvp("boostStrength", args_.boostStrength));
	    ar(cereal::make_nvp("seed", args_.seed));
	    ar(cereal::make_nvp("spVerbosity", args_.spVerbosity));
	    ar(cereal::make_nvp("wrapAround", args_.wrapAround));
	    ar(cereal::make_nvp("learningMode", args_.learningMode));
			ar(cereal::make_nvp("dim", dim_));  // from RegionImpl
	    ar(cereal::make_nvp("init", init));
	    if (init) {
	      // restore the output buffers
	      // The output buffers are saved as part of the Region Implementation.
	      cereal::size_type numBuffers;
	      ar(cereal::make_nvp("outputs", cereal::make_size_tag(numBuffers)));
	      for (cereal::size_type i = 0; i < numBuffers; i++) {
	        std::string name;
	        Array output;
	        ar(cereal::make_map_item(name, output));
	        Array& outputBuffer = getOutput(name)->getData();
	        outputBuffer = output;
	      }
	      // Restore algorithm state
	      algorithms::spatial_pooler::SpatialPooler* sp = new algorithms::spatial_pooler::SpatialPooler();
	      sp_.reset(sp);
	      ar(cereal::make_nvp("SP", sp_));
	    }
	  }



    // Per-node size (in elements) of the given output.
    // For per-region outputs, it is the total element count.
    // This method is called only for outputs whose size is not
    // specified in the spec and no region dimensions.
    size_t getNodeOutputElementCount(const std::string& outputName) const override;

		/* -----------  Optional RegionImpl Interface methods ------- */
    UInt32 getParameterUInt32(const std::string& name, Int64 index) override;
    Int32 getParameterInt32(const std::string& name, Int64 index) override;
    UInt64 getParameterUInt64(const std::string &name, Int64 index) override;
    Real32 getParameterReal32(const std::string& name, Int64 index) override;
    bool   getParameterBool(const std::string& name, Int64 index) override;
    std::string getParameterString(const std::string& name, Int64 index) override;
    void getParameterArray(const std::string& name, Int64 index, Array & array) override;
    size_t getParameterArrayCount(const std::string &name, Int64 index) override;


    void setParameterUInt32(const std::string& name, Int64 index, UInt32 value) override;
    void setParameterInt32(const std::string& name, Int64 index, Int32 value) override;
    void setParameterUInt64(const std::string &name, Int64 index, UInt64 value) override;
    void setParameterReal32(const std::string& name, Int64 index, Real32 value) override;
    void setParameterBool(const std::string& name, Int64 index, bool value) override;

	
private:
    SPRegion() = delete;  // empty constructor not allowed

    struct {
      UInt inputWidth;
      UInt columnCount;
      UInt potentialRadius;
      Real potentialPct;
      bool globalInhibition;
      Real localAreaDensity;
      UInt numActiveColumnsPerInhArea;
      UInt stimulusThreshold;
      Real synPermInactiveDec;
      Real synPermActiveInc;
      Real synPermConnected;
      Real minPctOverlapDutyCycles;
      UInt dutyCyclePeriod;
      Real boostStrength;
      Int  seed;
      UInt spVerbosity;
      bool wrapAround;
      bool learningMode;
    } args_;


    typedef void (*computeCallbackFunc)(const std::string &);
    computeCallbackFunc computeCallback_;

    std::string spatialImp_;         // SP variation selector. Currently not used.

    std::unique_ptr<algorithms::spatial_pooler::SpatialPooler> sp_;

};
} // namespace nupic

#endif // NTA_SPREGION_HPP
