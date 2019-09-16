/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2018, Numenta, Inc.
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
 * Author: David Keeney, April 2018
 * --------------------------------------------------------------------- */
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/engine/Input.hpp>
#include <htm/engine/Output.hpp>
#include <htm/engine/Region.hpp>
#include <htm/engine/Spec.hpp>
#include <htm/ntypes/Array.hpp>
#include <htm/ntypes/ArrayBase.hpp>
#include <htm/ntypes/Value.hpp>
#include <htm/regions/SPRegion.hpp>
#include <htm/utils/Log.hpp>

#define VERSION 1 // version for streaming serialization format

namespace htm {

SPRegion::SPRegion(const ValueMap &values, Region *region)
    : RegionImpl(region), computeCallback_(nullptr)  {
  // Note: the ValueMap gets destroyed on return so we need to get all of the
  // parameters out of the map and set aside so we can pass them to the SpatialPooler
  // algorithm when we create it during initialization().
  args_.columnCount = values.getScalarT<UInt32>("columnCount", 0);
  args_.potentialRadius = values.getScalarT<UInt32>("potentialRadius", 0);
  args_.potentialPct = values.getScalarT<Real32>("potentialPct", 0.5);
  args_.globalInhibition = values.getScalarT<bool>("globalInhibition", true);
  args_.localAreaDensity = values.getScalarT<Real32>("localAreaDensity", 0.05f);
  args_.stimulusThreshold = values.getScalarT<UInt32>("stimulusThreshold", 0);
  args_.synPermInactiveDec = values.getScalarT<Real32>("synPermInactiveDec", 0.008f);
  args_.synPermActiveInc = values.getScalarT<Real32>("synPermActiveInc", 0.05f);
  args_.synPermConnected = values.getScalarT<Real32>("synPermConnected", 0.1f);
  args_.minPctOverlapDutyCycles = values.getScalarT<Real32>("minPctOverlapDutyCycles", 0.001f);
  args_.dutyCyclePeriod = values.getScalarT<UInt32>("dutyCyclePeriod", 1000);
  args_.boostStrength = values.getScalarT<Real32>("boostStrength", 0.0f);
  args_.seed = values.getScalarT<Int32>("seed", 1);
  args_.spVerbosity = values.getScalarT<UInt32>("spVerbosity", 0);
  args_.wrapAround = values.getScalarT<bool>("wrapAround", true);
  spatialImp_ = values.getString("spatialImp", "");

  // variables used by this class and not passed on to the SpatialPooler class
  args_.learningMode = (1 == values.getScalarT<UInt32>("learningMode", true));

    // declare dimensions for bottomUpOut
  // specify dimensions using variable dim; syntax: "{dim: [2,3]}"
  // Alternatively, a 1D dimension can be setup with "{count: 6}"
  // Defaults to 1D size 64.
  if (!dim_.isSpecified())
    dim_ = Dimensions(args_.columnCount);
  else
    args_.columnCount = (UInt32)dim_.getCount();


}

SPRegion::SPRegion(ArWrapper &wrapper, Region *region)
  : RegionImpl(region), computeCallback_(nullptr)  {
  cereal_adapter_load(wrapper);
}


SPRegion::~SPRegion() {}

void SPRegion::initialize() {
  // Output buffers should already have been created diring initialize or deserialize.
  Output *out = getOutput("bottomUpOut");
  Array &outputBuffer = out->getData();
  NTA_CHECK(outputBuffer.getType() == NTA_BasicType_SDR);
  UInt32 columnCount = (UInt32)outputBuffer.getCount();
  if (columnCount == 0 || outputBuffer.getBuffer() == nullptr) {
    NTA_THROW << "SPRegion::initialize - Output buffer not set.\n";
  }

  // All input links should have been initialized during Network.initialize().
  // However, if nothing is connected it might not be. The SpatialPooler
  // algorithm requires input.
  //
  // If there are more than one input link (FAN-IN), the input buffer will be the
  // concatination of all incomming buffers.  
  Input *in = getInput("bottomUpIn");
  NTA_CHECK(in != nullptr);
  if (!in->hasIncomingLinks())
     NTA_THROW << "SPRegion::initialize - No input links were configured for this SP region.\n";
  Array &inputBuffer = in->getData();
  NTA_CHECK(inputBuffer.getType() == NTA_BasicType_SDR);
  args_.inputWidth = (UInt32)inputBuffer.getCount();
  if (args_.inputWidth == 0) {
    NTA_THROW << "SPRegion::initialize - No input buffer was allocated for this SP region.\n";
  }


  // Take the dimensions directly from the SDRs.
  std::vector<UInt32> inputDimensions = inputBuffer.getSDR().dimensions;
  std::vector<UInt32> columnDimensions = outputBuffer.getSDR().dimensions;

  // There is a restriction on SP that input and output must have the same 
  // number of dimensions.  So we add [1] dimensions to make them match.
  while(inputDimensions.size() < columnDimensions.size()) {
    inputDimensions.push_back(1);
    in->setDimensions(inputDimensions);
    inputBuffer.getSDR().initialize(inputDimensions);
  }
  while(inputDimensions.size() > columnDimensions.size()) {
    columnDimensions.push_back(1);
    out->setDimensions(columnDimensions);
    outputBuffer.getSDR().initialize(columnDimensions);
  }

  if (args_.potentialRadius == 0)
    args_.potentialRadius = args_.inputWidth;

  // instantiate a SpatialPooler.
  sp_ = std::unique_ptr<SpatialPooler>( new SpatialPooler(
      inputDimensions, columnDimensions, args_.potentialRadius,
      args_.potentialPct, args_.globalInhibition, args_.localAreaDensity,
      args_.stimulusThreshold,
      args_.synPermInactiveDec, args_.synPermActiveInc, args_.synPermConnected,
      args_.minPctOverlapDutyCycles, args_.dutyCyclePeriod, args_.boostStrength,
      args_.seed, args_.spVerbosity, args_.wrapAround));
}



void SPRegion::compute() {
  NTA_ASSERT(sp_) << "SP not initialized";

  if (computeCallback_ != nullptr)
    computeCallback_(getName());


  // prepare the input
  Array &inputBuffer  = getInput("bottomUpIn")->getData();
  Array &outputBuffer = getOutput("bottomUpOut")->getData();
  NTA_DEBUG  << "compute " << *getInput("bottomUpIn") << "\n";


  // Call SpatialPooler compute
  sp_->compute(inputBuffer.getSDR(), args_.learningMode, outputBuffer.getSDR());


  NTA_DEBUG << "compute " << *getOutput("bottomUpOut") << "\n";

}

std::string SPRegion::executeCommand(const std::vector<std::string> &args,Int64 index) {
  // The Spatial Pooler does not execute any Commands.
  return "";
}

// This is the per-node output size. This is called by Link to determine how big 
// to create the output buffers during Region::initialization(). It calls this
// only if dimensions were not set on this region, normally from a regionLevel input.
// NOTE: Some outputs are optional, return 0 if not used.
size_t SPRegion::getNodeOutputElementCount(const std::string &outputName) const {
  if (outputName == "bottomUpOut") // This is the only output link we actually use.
  {
      return args_.columnCount; 
  }
  return 0; // an optional output that we don't use.
}



Spec *SPRegion::createSpec() {
  auto ns = new Spec;

  ns->description =
      "SPRegion. This implements the Spatial Pooler algorithm as a plugin "
      "for the Network framework.  The Spatial Pooler manages relationships "
      "between the columns of a region and the inputs bits. The primary "
      "public interface to this function is the \"compute\" method, which "
      "takes in an input vector and returns a list of activeColumns columns.";


  /* ---- parameters ------ */

  /* constructor arguments */
  ns->parameters.add("columnCount",
      ParameterSpec("Total number of columns (coincidences). "
                    "This is the Output Dimension.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "",                   // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "inputWidth",
      ParameterSpec("Maximum size of the 'bottomUpIn' input to the SP.  "
                    "This is the input Dimension. The input buffer width "
                    "is taken from the width of all concatinated output "
                    "buffers that are connected to the input.",
                    NTA_BasicType_UInt32,            // type
                    1,                               // elementCount
                    "",                              // constraints
                    "0",                             // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "potentialRadius",
      ParameterSpec("(int)\n"
                    "This parameter determines the extent of the input that "
                    "each column can potentially be connected to.This can "
                    "be thought of as the input bits that are visible to "
                    "each column, or a 'receptiveField' of the field of "
                    "vision. A large enough value will result in 'global "
                    "coverage', meaning that each column can potentially "
                    "be connected to every input bit. This parameter defines "
                    "a square(or hyper square) area: a column will have a "
                    "max square potential pool with sides of length "
                    "'2 * potentialRadius + 1'. Default '0'. "
                    "If 0, during, Initialization it is set to the value of "
                    "inputWidth parameter.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "",                               // constraints
                    "0",                              // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "potentialPct",
      ParameterSpec(
          " (float)\n"
          "The percent of the inputs, within a column's potential radius, that "
          "a "
          "column can be connected to.If set to 1, the column will be "
          "connected "
          "to every input within its potential radius.This parameter is used "
          "to "
          "give each column a unique potential pool when a large "
          "potentialRadius "
          "causes overlap between the columns.At initialization time we choose "
          "((2 * potentialRadius + 1) ^ (# inputDimensions) * potentialPct) "
          "input bits "
          "to comprise the column's potential pool. Default ``0.5``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.5",                            // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "globalInhibition",
      ParameterSpec(
          "(bool)\n"
          "If true, then during inhibition phase the winning columns are "
          "selected "
          "as the most active columns from the region as a whole.Otherwise, "
          "the "
          "winning columns are selected with respect to their local "
          "neighborhoods. "
          "Using global inhibition boosts performance x60.Default ``False``.",
          NTA_BasicType_Bool,               // type
          1,                                // elementCount
          "bool",                           // constraints
          "false",                          // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "localAreaDensity",
      ParameterSpec(
          "(float)\n"
          "The desired density of active columns within a local inhibition "
          "area (the size of which is set by the internally calculated "
          "inhibitionRadius,  which is in turn determined from the average "
          "size of the connected potential pools of all columns). The "
          "inhibition logic will insure that at most N columns remain ON "
          "within a local inhibition area, where N = localAreaDensity * "
          "(total number of columns in inhibition area). "
	  "Default 0.05 (5%)",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.05",                           // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "stimulusThreshold",
      ParameterSpec("(int)\n"
                    "This is a number specifying the minimum "
                    "number of synapses that must be "
                    "on in order for a columns to turn ON.The "
                    "purpose of this is to prevent "
                    "noise input from activating "
                    "columns.Specified as a percent of a fully "
                    " grown synapse. Default ``0``.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "0",                  // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "synPermInactiveDec",
      ParameterSpec(
          "(float)\n"
          "The amount by which an inactive synapse is decremented in each "
          "round. "
          "Specified as a percent of a fully grown synapse.Default ``0.008``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.008",                          // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "synPermActiveInc",
      ParameterSpec(
          "(float)\n"
          "The amount by which an active synapse is incremented in each round. "
          "Specified as a percent of a fully grown synapse.Default ``0.05``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.05",                           // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "synPermConnected",
      ParameterSpec("(float)\n"
                    "The default connected threshold.Any synapse whose "
                    "permanence value is "
                    "above the connected threshold is a \"connected synapse\", "
                    "meaning it can "
                    "contribute to the cell's firing. Default ``0.1``.",
                    NTA_BasicType_Real32,             // type
                    1,                                // elementCount
                    "",                               // constraints
                    "0.1",                            // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "minPctOverlapDutyCycles",
      ParameterSpec(
          "(float)\n"
          "A number between 0 and 1.0, used to set a floor on how often a "
          "column "
          "should have at least stimulusThreshold active inputs.Periodically, "
          "each "
          "column looks at the overlap duty cycle of all other columns within "
          "its "
          "inhibition radius and sets its own internal minimal acceptable duty "
          "cycle "
          "to : minPctDutyCycleBeforeInh * max(other columns' duty cycles).  "
          "On each "
          "iteration, any column whose overlap duty cycle falls below this "
          "computed "
          "value will  get all of its permanence values boosted up by "
          "synPermActiveInc. Raising all permanences in response to a sub - "
          "par duty "
          "cycle before  inhibition allows a cell to search for new inputs "
          "when "
          "either its previously learned inputs are no longer ever active, or "
          "when "
          "the vast majority of them have been \"hijacked\" by other "
          "columns.Default "
          "``0.001``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.001",                          // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "dutyCyclePeriod",
      ParameterSpec(
          "(uint)\n"
          "The period used to calculate duty cycles.Higher values make it take "
          "longer to respond to changes in boost or synPerConnectedCell. "
          "Shorter "
          "values make it more unstable and likely to oscillate. Default "
          "``1000``.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "1000",                           // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "boostStrength",
      ParameterSpec(
          "(float)\n"
          "A number greater or equal than 0.0, used to control the strength of "
          "boosting.No boosting is applied if it is set to 0. Boosting "
          "strength "
          "increases as a function of boostStrength.Boosting encourages "
          "columns to "
          "have similar activeDutyCycles as their neighbors, which will lead "
          "to more "
          "efficient use of columns.However, too much boosting may also lead "
          "to "
          "instability of SP outputs.Default ``0.0``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.0",                            // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "seed",
      ParameterSpec(
          "(int)\n"
          "Seed for our own pseudo - random number generator. Default ``-1``.",
          NTA_BasicType_Int32,           // type
          1,                             // elementCount
          "",                            // constraints
          "-1",                          // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "spVerbosity",
      ParameterSpec("(uint)\n"
          "spVerbosity level : 0, 1, 2, or 3. Default ``0``.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0",                              // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "wrapAround",
      ParameterSpec("(bool)\n"
          "Determines if inputs at the beginning and "
          "end of an input dimension should "
          "be considered neighbors when mapping "
          "columns to inputs.Default ``True``.",
          NTA_BasicType_Bool, // type
          1,                  // elementCount
          "bool",             // constraints
          "true",             // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  /* ---- other parameters ----- */
  ns->parameters.add(
      "spInputNonZeros",
      ParameterSpec("The indices of the non-zero inputs to the spatial pooler",
          NTA_BasicType_SDR,            // type
          0,                               // elementCount
          "",                              // constraints
          "",                              // defaultValue
          ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "spOutputNonZeros",
      ParameterSpec(
          "The indices of the non-zero outputs from the spatial pooler",
          NTA_BasicType_SDR,            // type
          0,                               // elementCount
          "",                              // constraints
          "",                              // defaultValue
          ParameterSpec::ReadOnlyAccess)); // access


  /* The last group is for parameters that aren't specific to spatial pooler */
  ns->parameters.add("learningMode",
      ParameterSpec("1 if the node is learning (default 1).",
          NTA_BasicType_UInt32, // type
          1,                    // elementCount
          "bool",               // constraints
          "1",                  // defaultValue
          ParameterSpec::ReadWriteAccess)); // access


  ns->parameters.add(
      "activeOutputCount",
      ParameterSpec("Number of active elements in bottomUpOut output.",
          NTA_BasicType_UInt32,            // type
          1,                               // elementCount
          "",                              // constraints
          "0",                             // defaultValue
          ParameterSpec::ReadOnlyAccess)); // access


  ns->parameters.add("spatialImp",
      ParameterSpec("SpatialPooler type or option. not used.",
          NTA_BasicType_Byte,              // type
          0,                               // elementCount
          "",                              // constraints
          "",                              // defaultValue
          ParameterSpec::ReadOnlyAccess)); // access

  /* ----- inputs ------- */
  ns->inputs.add(
      "bottomUpIn",
      InputSpec("The input vector.",  // description
                NTA_BasicType_SDR,    // type
                0,                    // count.
                true,                 // required?
                false,                // isRegionLevel,
                true                  // isDefaultInput
                ));




  /* ----- outputs ------ */
  ns->outputs.add(
      "bottomUpOut",
      OutputSpec("The output signal generated from the bottom-up inputs "
                 "from lower levels.",
                 NTA_BasicType_SDR,    // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 true                  // isDefaultOutput
                 ));


  /* ----- commands ------ */
  // commands TBD

  return ns;
}

////////////////////////////////////////////////////////////////////////
//           Parameters
//
// Most parameters are handled automatically by getParameterFromBuffer().
// The ones that need special treatment are explicitly handled here.
//
////////////////////////////////////////////////////////////////////////

UInt32 SPRegion::getParameterUInt32(const std::string &name, Int64 index) {
  NTA_CHECK(name.size() > 0);
  switch (name[0]) {
  case 'a':
    if (name == "activeOutputCount") {
      return (UInt32)getOutput("bottomUpOut")->getData().getCount();
    }
    break;
  case 'c':
    if (name == "columnCount") {
      if (sp_)
        return sp_->getNumColumns();
      else
        return (Int32)args_.columnCount;
    }
    break;
  case 'd':
    if (name == "dutyCyclePeriod") {
      if (sp_)
        return sp_->getDutyCyclePeriod();
      else
        return args_.dutyCyclePeriod;
    }
    break;
  case 'i':
    if (name == "inputWidth") {
      if (sp_)
        return sp_->getNumInputs();
      else
        return (Int32)args_.inputWidth;
    }
    break;
  case 'l':
    if (name == "learningMode") {
      return args_.learningMode;
    }
    break;
  case 'p':
    if (name == "potentialRadius") {
      if (sp_)
        return sp_->getPotentialRadius();
      return args_.potentialRadius;
    }
    break;
  case 's':
    if (name == "stimulusThreshold") {
      if (sp_)
        return sp_->getStimulusThreshold();
      else
        return args_.stimulusThreshold;
    }
    if (name == "spVerbosity") {
      if (sp_)
        return sp_->getSpVerbosity();
      else
        return args_.spVerbosity;
    }
    break;
  }                                                         // end switch
  return this->RegionImpl::getParameterUInt32(name, index); // default
}

Int32 SPRegion::getParameterInt32(const std::string &name, Int64 index) {
  if (name == "seed") {
    return args_.seed;
  }
  return this->RegionImpl::getParameterInt32(name, index); // default
}

UInt64 SPRegion::getParameterUInt64(const std::string &name, Int64 index) {
  if (name == "computeCallback") {
    return (UInt64)computeCallback_;
  }
  return this->RegionImpl::getParameterUInt64(name, index); // default
}

Real32 SPRegion::getParameterReal32(const std::string &name, Int64 index) {
  switch (name[0]) {
  case 'b':
    if (name == "boostStrength") {
      if (sp_)
        return sp_->getBoostStrength();
      else
        return args_.boostStrength;
    }
    break;
  case 'l':
    if (name == "localAreaDensity") {
      if (sp_)
        return sp_->getLocalAreaDensity();
      else
        return args_.localAreaDensity;
    }
    break;
  case 'm':
    if (name == "minPctOverlapDutyCycles") {
      if (sp_)
        return sp_->getMinPctOverlapDutyCycles();
      else
        return args_.minPctOverlapDutyCycles;
    }
    break;
  case 'p':
    if (name == "potentialPct") {
      if (sp_)
        return sp_->getPotentialPct();
      else
        return args_.potentialPct;
    }
    break;
  case 's':
    if (name == "synPermInactiveDec") {
      if (sp_)
        return sp_->getSynPermInactiveDec();
      else
        return args_.synPermInactiveDec;
    }
    if (name == "synPermActiveInc") {
      if (sp_)
        return sp_->getSynPermActiveInc();
      else
        return args_.synPermActiveInc;
    }
    if (name == "synPermConnected") {
      if (sp_)
        return sp_->getSynPermConnected();
      else
        return args_.synPermConnected;
    }
    break;
  }
  return this->RegionImpl::getParameterReal32(name, index); // default
}

bool SPRegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "globalInhibition") {
    if (sp_)
      return sp_->getGlobalInhibition();
    else
      return args_.globalInhibition;
  }
  if (name == "wrapAround") {
    if (sp_)
      return sp_->getWrapAround();
    else
      return args_.wrapAround;
  }
  return this->RegionImpl::getParameterBool(name, index); // default
}

// copy the contents of the requested array into the caller's array.
// Allocate the buffer if one is not provided.  Convert data types if needed.
void SPRegion::getParameterArray(const std::string &name, Int64 index, Array &array) {
  if (name == "spatialPoolerInput") {
    array = getInput("bottomUpIn")->getData().copy();
  } else if (name == "spatialPoolerOutput") {
    array = getOutput("bottomUpOut")->getData().copy();
  } else if (name == "spInputNonZeros") {
    array = getInput("bottomUpIn")->getData().copy();
  } else if (name == "spOutputNonZeros") {
    array = getOutput("bottomUpOut")->getData().copy();
  }
  else {
    this->RegionImpl::getParameterArray(name, index, array);
  }
}

size_t SPRegion::getParameterArrayCount(const std::string &name, Int64 index) {
  if (name == "spatialPoolerInput") {
    return getInput("bottomUpIn")->getData().getCount();
  } else if (name == "spatialPoolerOutput") {
    return getOutput("bottomUpOut")->getData().getCount();
  } else if (name == "spInputNonZeros") {
    const SDR_sparse_t& v = getInput("bottomUpIn")->getData().getSDR().getSparse();
    return v.size();
  } else if (name == "spOutputNonZeros") {
    const SDR_sparse_t& v = getInput("bottomUpOut")->getData().getSDR().getSparse();
    return v.size();
  }
  return 0;
}

std::string SPRegion::getParameterString(const std::string &name, Int64 index) {
  if (name == "spatialImp") {
    return spatialImp_;
  }
  // "spLearningStatsStr"  not found
  return this->RegionImpl::getParameterString(name, index);
}

void SPRegion::setParameterUInt32(const std::string &name, Int64 index,
                                  UInt32 value) {
  switch (name[0]) {
  case 'd':
    if (name == "dutyCyclePeriod") {
      if (sp_)
        sp_->setDutyCyclePeriod(value);
      args_.dutyCyclePeriod = value;
      return;
    }
    break;
  case 'l':
    if (name == "learningMode") {
      args_.learningMode = (value != 0);
      return;
    }
    break;
  case 'p':
    if (name == "potentialRadius") {
      if (sp_)
        sp_->setPotentialRadius(value);
      args_.potentialRadius = value;
      return;
    }
    break;
  case 's':
    if (name == "stimulusThreshold") {
      if (sp_)
        sp_->setStimulusThreshold(value);
      args_.stimulusThreshold = value;
      return;
    }
    if (name == "spVerbosity") {
      if (sp_)
        sp_->setSpVerbosity(value);
      args_.spVerbosity = value;
      return;
    }
    break;

  } // switch
  // if not handled above, use default handling.
  RegionImpl::setParameterUInt32(name, index, value);
}

void SPRegion::setParameterInt32(const std::string &name, Int64 index, Int32 value) {
  RegionImpl::setParameterInt32(name, index, value);
}

void SPRegion::setParameterUInt64(const std::string &name, Int64 index, UInt64 value) {
  if (name == "computeCallback") {
    computeCallback_ = (computeCallbackFunc)value;
  } else {
	RegionImpl::setParameterUInt64(name, index, value);
  }
}

void SPRegion::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  switch (name[0]) {
  case 'b':
    if (name == "boostStrength") {
      if (sp_)
        sp_->setBoostStrength(value);
      args_.boostStrength = value;
      return;
    }
    break;
  case 'l':
    if (name == "localAreaDensity") {
      if (sp_)
        sp_->setLocalAreaDensity(value);
      args_.localAreaDensity = value;
      return;
    }
    break;
  case 'm':
    if (name == "minPctOverlapDutyCycles") {
      if (sp_)
        sp_->setMinPctOverlapDutyCycles(value);
      args_.minPctOverlapDutyCycles = value;
      return;
    }
    break;
  case 'p':
    if (name == "potentialPct") {
      if (sp_)
        sp_->setPotentialPct(value);
      args_.potentialPct = value;
      return;
    }
    break;

  case 's':
    if (name == "synPermInactiveDec") {
      if (sp_)
        sp_->setSynPermInactiveDec(value);
      args_.synPermInactiveDec = value;
      return;
    }
    if (name == "synPermActiveInc") {
      if (sp_)
        sp_->setSynPermActiveInc(value);
      args_.synPermActiveInc = value;
      return;
    }
    break;
  } // switch
  RegionImpl::setParameterReal32(name, index, value);
}

void SPRegion::setParameterBool(const std::string &name, Int64 index, bool value) {
  if (name == "globalInhibition") {
    if (sp_)
      sp_->setGlobalInhibition(value);
    args_.globalInhibition = value;
    return;
  }
  if (name == "wrapAround") {
    if (sp_)
      sp_->setWrapAround(value);
    args_.wrapAround = value;
    return;
  }

  RegionImpl::setParameterBool(name, index, value);
}



bool SPRegion::operator==(const RegionImpl &o) const {
  if (o.getType() != "SPRegion") return false;
  SPRegion& other = (SPRegion&)o;
  if (args_.inputWidth != other.args_.inputWidth) return false;
  if (args_.columnCount != other.args_.columnCount) return false;
  if (args_.potentialRadius != other.args_.potentialRadius) return false;
  if (args_.potentialPct != other.args_.potentialPct) return false;
  if (args_.globalInhibition != other.args_.globalInhibition) return false;
  if (args_.localAreaDensity != other.args_.localAreaDensity) return false;
  if (args_.stimulusThreshold != other.args_.stimulusThreshold) return false;
  if (args_.synPermInactiveDec != other.args_.synPermInactiveDec) return false;
  if (args_.synPermActiveInc != other.args_.synPermActiveInc) return false;
  if (args_.synPermConnected != other.args_.synPermConnected) return false;
  if (args_.minPctOverlapDutyCycles != other.args_.minPctOverlapDutyCycles) return false;
  if (args_.dutyCyclePeriod != other.args_.dutyCyclePeriod) return false;
  if (args_.boostStrength != other.args_.boostStrength) return false;
  if (args_.seed != other.args_.seed) return false;
  if (args_.spVerbosity != other.args_.spVerbosity) return false;
  if (args_.wrapAround != other.args_.wrapAround) return false;
  if (args_.learningMode != other.args_.learningMode) return false;

  if (dim_ != other.dim_) return false;  // from RegionImpl
  if ((sp_ && !other.sp_) || (other.sp_ && !sp_)) return false;
  if (sp_ && (*sp_ != *other.sp_)) return false;

  return true;
}

} // namespace htm
