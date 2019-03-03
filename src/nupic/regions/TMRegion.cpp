/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2018, Numenta, Inc.  Unless you have an agreement
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
 * Author: David Keeney, June 2018
 * ---------------------------------------------------------------------
 */
#include <fstream>
#include <iomanip> // setprecision() in stream
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <nupic/algorithms/Anomaly.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/regions/TMRegion.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/VectorHelpers.hpp>

#define VERSION 1

using namespace nupic;
using namespace nupic::util;
using namespace nupic::algorithms::temporal_memory;

TMRegion::TMRegion(const ValueMap &params, Region *region)
    : RegionImpl(region), computeCallback_(nullptr) {
  // Note: the ValueMap gets destroyed on return so we need to get all of the
  // parameters
  //       out of the map and set aside so we can pass them to the SpatialPooler
  //       algorithm when we create it during initialization().
  memset((char *)&args_, 0, sizeof(args_));
  args_.numberOfCols = params.getScalarT<UInt32>("numberOfCols", 0);  // normally not passed in.
  args_.cellsPerColumn = params.getScalarT<UInt32>("cellsPerColumn", 32);
  args_.activationThreshold = params.getScalarT<UInt32>("activationThreshold", 13);
  args_.initialPermanence = params.getScalarT<Real32>("initialPermanence", 0.21f);
  args_.connectedPermanence = params.getScalarT<Real32>("connectedPermanence", 0.50f);
  args_.minThreshold = params.getScalarT<UInt32>("minThreshold", 10);
  args_.maxNewSynapseCount = params.getScalarT<UInt32>("maxNewSynapseCount", 20);
  args_.permanenceIncrement = params.getScalarT<Real32>("permanenceIncrement", 0.10f);
  args_.permanenceDecrement = params.getScalarT<Real32>("permanenceDecrement", 0.10f);
  args_.predictedSegmentDecrement = params.getScalarT<Real32>("predictedSegmentDecrement", 0.0f);
  args_.seed = params.getScalarT<Int32>("seed", 42);
  args_.maxSegmentsPerCell = params.getScalarT<UInt32>("maxSegmentsPerCell", 255);
  args_.maxSynapsesPerSegment = params.getScalarT<UInt32>("maxSynapsesPerSegment", 255);
  args_.checkInputs = params.getScalarT<bool>("checkInputs", true);
  args_.extra = 0;  // will be obtained from extra inputs dimensions.

  // variables used by this class and not passed on
  args_.learningMode = params.getScalarT<bool>("learningMode", true);

  args_.iter = 0;
  args_.sequencePos = 0;
  args_.outputWidth = args_.numberOfCols * args_.cellsPerColumn;
  args_.init = false;
  tm_ = nullptr;
}

TMRegion::TMRegion(BundleIO &bundle, Region *region) : RegionImpl(region) {
  tm_ = nullptr;
  deserialize(bundle);
}

TMRegion::~TMRegion() {
}

void TMRegion::initialize() {

  // All input links and buffers should have been initialized during
  // Network.initialize() prior to calling this method.
  //
  // If there are more than one input link, the input buffer will be the
  // concatination of all incomming buffers. This width sets the number
  // columns for the TM.
  Input* in = region_->getInput("bottomUpIn");
  if (!in || !in->hasIncomingLinks())
      NTA_THROW << "TMRegion::initialize - No input was provided.\n";
  NTA_ASSERT(in->getData().getType() == NTA_BasicType_SDR);

  columnDimensions_ = in->getDimensions();
  if (args_.numberOfCols == 0)
    args_.numberOfCols = (UInt32)columnDimensions_.getCount();
  else
    NTA_CHECK(args_.numberOfCols == columnDimensions_.getCount())
    << "The width of the bottomUpIn input buffer (" << columnDimensions_.getCount()
    << ") does not match the configured value for 'numberOfCols' ("
    << args_.numberOfCols << ").";

  // Look for extra data
  // This can come from anywhere and have any size.  The only restriction is
  // that the buffer width of extraWinners must be the same as buffer width of extraActive.
  // The extraWinners on bits should be a subset of on bits in extraWinners.
  args_.extra = 0;
  in = region_->getInput("extraActive");
  if (in && in->hasIncomingLinks()) {
    args_.extra = (UInt32)in->getDimensions().getCount();
    NTA_ASSERT(in->getData().getType() == NTA_BasicType_SDR);

    in = region_->getInput("extraWinners");
    NTA_ASSERT(in->getData().getType() == NTA_BasicType_SDR);
    NTA_CHECK(in 
           && in->hasIncomingLinks() 
           && args_.extra == in->getDimensions().getCount()) 
      << "The input 'extraActive' (width: " << args_.extra 
      << ") is connected but 'extraWinners' input "
      << "is not provided OR it has a different buffer size.";
  }



  nupic::algorithms::temporal_memory::TemporalMemory* tm = 
    new nupic::algorithms::temporal_memory::TemporalMemory(
      columnDimensions_, args_.cellsPerColumn, args_.activationThreshold,
      args_.initialPermanence, args_.connectedPermanence, args_.minThreshold,
      args_.maxNewSynapseCount, args_.permanenceIncrement, args_.permanenceDecrement,
      args_.predictedSegmentDecrement, args_.seed, args_.maxSegmentsPerCell,
      args_.maxSynapsesPerSegment, args_.checkInputs, args_.extra);
  tm_.reset(tm);

  args_.iter = 0;
  args_.sequencePos = 0;
  args_.init = true;

}

void TMRegion::compute() {

  NTA_ASSERT(tm_) << "TM not initialized";

  if (computeCallback_ != nullptr)
    computeCallback_(getName());
  args_.iter++;

  // Handle reset signal
  if (getInput("resetIn")->hasIncomingLinks()) {
    Array &reset = getInput("resetIn")->getData();
    NTA_ASSERT(reset.getType() == NTA_BasicType_Real32);
    if (reset.getCount() == 1 && ((Real32 *)(reset.getBuffer()))[0] != 0) {
      tm_->reset();
      args_.sequencePos = 0; // Position within the current sequence
    }
  }

  // Check the input buffer
  // The buffer width is the number of columns.
  Input *in = getInput("bottomUpIn");
  Array &bottomUpIn = in->getData();
  NTA_ASSERT(bottomUpIn.getType() == NTA_BasicType_SDR);
  SDR& activeColumns = *bottomUpIn.getSDR();

  // Check for 'extra' inputs
  static SDR nullSDR({0});
  Array &extraActive = getInput("extraActive")->getData();
  SDR& extraActiveCells = (args_.extra)?(*extraActive.getSDR()):nullSDR;

  Array &extraWinners = getInput("extraWinners")->getData();
  SDR& extraWinnerCells = (args_.extra)?(*extraWinners.getSDR()):nullSDR;

  NTA_DEBUG << *in << "\n";

  // Perform Bottom up compute()
  
  tm_->compute(activeColumns, args_.learningMode, extraActiveCells, extraWinnerCells);
  tm_->activateDendrites();

  args_.sequencePos++;

  // generate the outputs
  // NOTE: - Output dimensions are set to the region dimensions
  //         plus an additional dimension for 'cellsPerColumn'.
  //       - The region dimensions total elements is the 'numberOfCols'.
  //       - The region dimensions are set by dimensions on incoming "bottomUpIn"
  //       - The region dimensions can be overridden with configuration of "dim" 
  //         or explicitly by calling region->setDimensions().
  //         or explicitly for each output region->setOutputDimensions(output_name).
  //       - The total number of elements in the outputs must be
  //         numberOfCols * cellsPerColumn.
  //
  Output *out;
  out = getOutput("bottomUpOut");
  if (out && out->hasOutgoingLinks()) {  
    tm_->getPredictiveCells(*out->getData().getSDR());
    NTA_DEBUG << *out << "\n";
  }
  out = getOutput("activeCells");
  if (out && out->hasOutgoingLinks()) {  
    tm_->getActiveCells(*out->getData().getSDR());
    NTA_DEBUG << *out << "\n";
  }
  out = getOutput("predictedActiveCells");
  if (out && out->hasOutgoingLinks()) {  
    tm_->getWinnerCells(*out->getData().getSDR());
    NTA_DEBUG << *out << "\n";
  }
}

// Note: - this is called during Region initialization, after configuration
//         is set but prior to calling initialize on this class to create the tm. 
//         The input dimensions should already have been set, normally from 
//         its connected output. This would set the region dimensions if not overridden.
//       - This is not called if output dimensions were explicitly set for this output.
//       - This call determines the dimensions set on the Output buffers.
Dimensions TMRegion::askImplForOutputDimensions(const std::string &name) {
  Dimensions region_dim = getDimensions();
  if (!region_dim.isSpecified()) {
    // we don't have region dimensions, so create some if we know numberOfCols.
    if (args_.numberOfCols == 0) 
      return Dimensions(Dimensions::DONTCARE);  // No info for its size
    region_dim.clear();
    region_dim.push_back(args_.numberOfCols);
    setDimensions(region_dim);
  }
  if (args_.numberOfCols == 0)
    args_.numberOfCols = (UInt32)region_dim.getCount();

  if (name == "bottomUpOut" || name == "activeCells" || name == "predictedActiveCells") {
    // It's size is numberOfCols * args_.cellsPerColumn.
    // So insert a new dimension to what was provided by input.
    Dimensions dim = region_dim;
    dim.insert(dim.begin(), args_.cellsPerColumn);
    return dim;
  }
  return RegionImpl::askImplForOutputDimensions(name);
}



/********************************************************************/

Spec *TMRegion::createSpec() {
  auto ns = new Spec;

  ns->description =
      "TMRegion. Class implementing the temporal memory algorithm as "
      "described in 'BAMI "
      "<https://numenta.com/biological-and-machine-intelligence/>'.  "
      "The implementation here attempts to closely match the pseudocode in "
      "the documentation. This implementation does contain several additional "
      "bells and whistles such as a column confidence measure.";

  ns->singleNodeOnly = true; // this means we don't care about dimensions;

  /* ---- parameters ------ */

  /* constructor arguments */
  ns->parameters.add(
      "numberOfCols",
      ParameterSpec("(int) Number of mini-columns in the region. This values "
                    "needs to be the same as the number of columns in the "
                    "input from SP.  Normally this value is derived from "
                    "the input width but if privided, this parameter must be "
                    "the same total size as the input.",
                    NTA_BasicType_UInt32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0",                           // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "cellsPerColumn",
      ParameterSpec("(int) The number of cells per mini-column.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "32",                 // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "activationThreshold",
      ParameterSpec("(int) Number of synapses that must be active to "
                    "activate a segment.",
                    NTA_BasicType_UInt32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "13",                          // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "initialPermanence",
      ParameterSpec("(float) Initial permanence for newly created synapses.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.21",                        // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "connectedPermanence",
      ParameterSpec("(float) ",
                    NTA_BasicType_Real32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "0.5",                // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "minThreshold",
      ParameterSpec(
          " (int)  Minimum number of active synapses for a segment to "
          "be considered during search for the best-matching segments. ",
          NTA_BasicType_UInt32,          // type
          1,                             // elementCount
          "",                            // constraints
          "8",                           // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "maxNewSynapseCount",
      ParameterSpec("(int) The max number of synapses added "
                    "to a segment during learning.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "20",                 // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "permanenceIncrement", 
      ParameterSpec("(float) Active synapses get their permanence counts "
                    "incremented by this value.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.1",                         // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "permanenceDecrement", 
      ParameterSpec("(float) All other synapses get their permanence counts "
                    "decremented by this value.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.1",                         // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "predictedSegmentDecrement",
      ParameterSpec("(float) Predicted segment decrement  A good value "
                    "is just a bit larger than (the column-level sparsity * "
                    "permanenceIncrement). So, if column-level sparsity is 2% "
                    "and permanenceIncrement is 0.01, this parameter should be "
                    "something like 4% * 0.01 = 0.0004).",
                    NTA_BasicType_Real32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "0.0",                // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
    "maxSegmentsPerCell", 
    ParameterSpec("(int) The maximum number of segments allowed on a "
                  "cell. This is used to turn on 'fixed size CLA' mode. When in "
                  "effect, 'globalDecay' is not applicable and must be set to 0 and "
                  "'maxAge' must be set to 0. When this is used (> 0), "
                  "'maxSynapsesPerSegment' must also be > 0. ",
                  NTA_BasicType_UInt32,             // type
                  1,                                // elementCount
                  "",                               // constraints
                  "255",                            // defaultValue
                  ParameterSpec::ReadOnlyAccess));  // access

  ns->parameters.add(
      "maxSynapsesPerSegment", 
      ParameterSpec("(int) The maximum number of synapses allowed in "
                    "a segment. ",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "",                               // constraints
                    "255",                            // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "seed",
      ParameterSpec("(int)  Random number generator seed. The seed affects the random "
                    "aspects of initialization like the initial permanence values. A "
                    "fixed value ensures a reproducible result.",
                    NTA_BasicType_Int32,              // type
                    1,                                // elementCount
                    "",                               // constraints
                    "42",                             // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ///////// Parameters not part of the calling arguments //////////
  ns->parameters.add(
      "inputWidth",
      ParameterSpec("(int) width of bottomUpIn. Will be 0 if no links.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "",                   // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "learningMode",
      ParameterSpec("1 if the node is learning (default true).",
                    NTA_BasicType_Bool, // type
                    1,                  // elementCount
                    "bool",             // constraints
                    "true",             // defaultValue
                    ParameterSpec::CreateAccess)); // access


  ns->parameters.add(
      "activeOutputCount",
      ParameterSpec("(int)Number of active elements.",
                    NTA_BasicType_UInt32,            // type
                    1,                               // elementCount
                    "",                              // constraints
                    "",                              // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ///////////// Inputs and Outputs ////////////////
  /* ----- inputs ------- */
  ns->inputs.add(
      "bottomUpIn",
      InputSpec(
                "The input signal, conceptually organized as an image pyramid "
                "data structure, but internally organized as a flattened vector. "
                "The width should match the output of SP.  Set numberOfCols to "
                "This value if not configured.  Otherwise the parameter overrides.",
                NTA_BasicType_SDR,   // type
                0,                   // count.
                true,                // required?
                true,                // isRegionLevel,
                true                 // isDefaultInput
                ));
  ns->inputs.add(
      "extraActive",
      InputSpec("External extra active bits from an external source. "
                "These can come from anywhere and be any size. If provided, "
                "the 'extra' flag is set to dense buffer size and both "
                "extraActive and extraWinners must be provided and have the"
                "same dense buffer size.  Dimensions are set by source.",
                NTA_BasicType_SDR,   // type
                0,                   // count.
                false,               // required?
                false,               // isRegionLevel,
                false                // isDefaultInput
                ));
  ns->inputs.add(
      "extraWinners",
      InputSpec("The winning active bits from an external source. "
                "These can come from anywhere and be any size. If provided, "
                "the 'extra' flag is set to dense buffer size and both "
                "extraActive and extraWinners must be provided and have the"
                "same dense buffer size.  Dimensions are set by source.",
               NTA_BasicType_SDR,   // type
                0,                   // count.
                false,               // required?
                false,               // isRegionLevel,
                false                // isDefaultInput
                ));

  ns->inputs.add(
      "resetIn",
      InputSpec("A boolean flag that indicates whether "
                "or not the input vector received in this compute cycle "
                "represents the first training presentation in new temporal "
                "sequence.",
                NTA_BasicType_SDR,    // type
                1,                    // count.
                false,                // required?
                false,                // isRegionLevel,
                false                 // isDefaultInput
                ));


  /* ----- outputs ------ */
  ns->outputs.add(
      "bottomUpOut",
      OutputSpec("The output signal generated from the bottom-up inputs "
                 "from lower levels. The width is 'numberOfCols' "
                 "* 'cellsPerColumn'.",
                 NTA_BasicType_SDR,    // type
                 0,                    // count 0 means is dynamic
                 false,                // isRegionLevel
                 true                  // isDefaultOutput
                 ));

  ns->outputs.add(
      "activeCells",
      OutputSpec("The cells that are active from TM computations. "
                 "The width is 'numberOfCols' * 'cellsPerColumn'.",
                 NTA_BasicType_SDR,    // type
                 0,                    // count 0 means is dynamic
                 false ,               // isRegionLevel
                 false                 // isDefaultOutput
                 ));

  ns->outputs.add(
      "predictedActiveCells",
      OutputSpec("The cells that are active and predicted, the winners. "
                 "The width is 'numberOfCols' * 'cellsPerColumn'.",
                NTA_BasicType_SDR,    // type
                0,                    // count 0 means is dynamic
                false,                // isRegionLevel
                false                 // isDefaultOutput
                ));


  /* ----- commands ------ */
  // commands TBD

  return ns;
}

////////////////////////////////////////////////////////////////////////
//           Parameters
//
// Most parameters are explicitly handled here until initialization.
// After initialization they are passed on to the tm_ for processing.
//
////////////////////////////////////////////////////////////////////////

UInt32 TMRegion::getParameterUInt32(const std::string &name, Int64 index) {

    if (name == "activationThreshold") {
      if (tm_)
        return tm_->getActivationThreshold();
      return args_.activationThreshold;
    }
    if (name == "activeOutputCount") {
      return args_.outputWidth;
    }
    if (name == "cellsPerColumn") {
      if (tm_)
        return (UInt32)tm_->getCellsPerColumn();
      return args_.cellsPerColumn;
    }
    if (name == "inputWidth") {
      NTA_CHECK(getInput("bottomUpIn") != nullptr) << "Unknown Input: 'bottomUpIn'";
      if (!getInput("bottomUpIn")->isInitialized()) {
        return 0; // might not be any links defined.
      }
      return (UInt32)getInput("bottomUpIn")->getData().getCount();
    }
    if (name == "maxNewSynapseCount") {
      if (tm_)
        return tm_->getMaxNewSynapseCount();
      return args_.maxNewSynapseCount;
    }
    if (name == "maxSegmentsPerCell") {
      if (tm_)
        return tm_->getMaxSegmentsPerCell();
      return args_.maxSegmentsPerCell;
    }
    if (name == "maxSynapsesPerSegment") {
      if (tm_)
        return tm_->getMaxSynapsesPerSegment();
      return args_.maxSynapsesPerSegment;
    }
    if (name == "minThreshold") {
      if (tm_)
        return tm_->getMinThreshold();
      return args_.minThreshold;
    }
    if (name == "numberOfCols") {
      if (tm_)
        return (UInt32)tm_->numberOfColumns();
      return args_.numberOfCols;
    }
    if (name == "outputWidth")
      return args_.outputWidth;

  return this->RegionImpl::getParameterUInt32(name, index); // default
}


Int32 TMRegion::getParameterInt32(const std::string &name, Int64 index) {
  if (name == "activationThreshold") {
    if (tm_)
      return tm_->getActivationThreshold();
    return args_.activationThreshold;
  }
  if (name == "seed") {
    return args_.seed;
  }
  return this->RegionImpl::getParameterInt32(name, index); // default
}


Real32 TMRegion::getParameterReal32(const std::string &name, Int64 index) {

    if (name == "connectedPermanence") {
      if (tm_)
        return tm_->getConnectedPermanence();
      return args_.connectedPermanence;
    }
    if (name == "initialPermanence") {
      if (tm_)
        return tm_->getInitialPermanence();
      return args_.initialPermanence;
    }
    if (name == "permanenceIncrement") {
      if (tm_)
        return tm_->getPermanenceIncrement();
      return args_.permanenceIncrement;
    }
    if (name == "permanenceDecrement") {
      if (tm_)
        return tm_->getPermanenceDecrement();
      return args_.permanenceDecrement;
    }
    if (name == "predictedSegmentDecrement") {
      if (tm_)
        return tm_->getPredictedSegmentDecrement();
      return args_.predictedSegmentDecrement;
    }

  return this->RegionImpl::getParameterReal32(name, index); // default
}


bool TMRegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "checkInputs") {
    if (tm_) {
      return tm_->getCheckInputs();
    }
    return args_.checkInputs;
  }
  if (name == "learningMode")
    return args_.learningMode;

  return this->RegionImpl::getParameterBool(name, index); // default
}


std::string TMRegion::getParameterString(const std::string &name, Int64 index) {
  return this->RegionImpl::getParameterString(name, index);
}


void TMRegion::setParameterUInt32(const std::string &name, Int64 index, UInt32 value) {
  if (name == "maxNewSynapseCount") {
    if (tm_) {
      tm_->setMaxNewSynapseCount(value);
    }
    args_.maxNewSynapseCount = value;
    return;
  }
  if (name == "maxSynapsesPerSegment") {
    args_.maxSynapsesPerSegment = value;
    return;
  }
  if (name == "minThreshold") {
    if (tm_) {
      tm_->setMinThreshold(value);
    }
    args_.minThreshold = value;
    return;
  }
  RegionImpl::setParameterUInt32(name, index, value);
}


void TMRegion::setParameterInt32(const std::string &name, Int64 index, Int32 value) {
  if (name == "activationThreshold") {
    if (tm_) {
      tm_->setActivationThreshold(value);
    }
    args_.activationThreshold = value;
    return;
  }
  RegionImpl::setParameterInt32(name, index, value);
}


void TMRegion::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  if (name == "initialPermanence") {
      if (tm_) {
        tm_->setInitialPermanence(value);
      }
      args_.initialPermanence = value;
      return;
  }
  if (name == "connectedPermanence") {
      args_.connectedPermanence = value;
      return;
  }
  if (name == "permanenceIncrement") {
      if (tm_)
        tm_->setPermanenceIncrement(value);
      args_.permanenceIncrement = value;
      return;
  }
  if (name == "permanenceDecrement") {
      if (tm_)
        tm_->setPermanenceDecrement(value);
      args_.permanenceDecrement = value;
      return;
  }
  if (name == "predictedSegmentDecrement") {
      if (tm_)
        tm_->setPredictedSegmentDecrement(value);
      args_.predictedSegmentDecrement = value;
      return;
  }
  
  RegionImpl::setParameterReal32(name, index, value);
}


void TMRegion::setParameterBool(const std::string &name, Int64 index, bool value) 
{

  if (name == "checkInputs") {
    if (tm_)
      tm_->setCheckInputs(value);
    args_.checkInputs = value;
    return;
  }
  if (name == "learningMode") {
    args_.learningMode = value;
    return;
  }

  RegionImpl::setParameterBool(name, index, value);
}


void TMRegion::setParameterString(const std::string &name, Int64 index,
                                  const std::string &value) {
  this->RegionImpl::setParameterString(name, index, value);
}



void TMRegion::serialize(BundleIO &bundle) {
  std::ostream &f = bundle.getOutputStream();
  f.precision(std::numeric_limits<double>::digits10 + 1);
  f.precision(std::numeric_limits<float>::digits10 + 1);

  // There is more than one way to do this. We could serialize to YAML, which
  // would make a readable format, or we could serialize directly to the
  // stream Choose the easier one.
  UInt version = VERSION;
  args_.init = ((tm_) ? true : false);

  f << "TMRegion " << version << std::endl;
  f << sizeof(args_) << " ";
  f.write((const char*)&args_, sizeof(args_));
  f << std::endl;
  //f << cellsSavePath_ << std::endl;
  //f << logPathOutput_ << std::endl;
  if (tm_) {
    // Note: tm_ saves the output buffers
    tm_->save(f);
  }
}


void TMRegion::deserialize(BundleIO &bundle) {
  std::istream &f = bundle.getInputStream();
  // There is more than one way to do this. We could serialize to YAML, which
  // would make a readable format, but that is a bit slow so we try to directly
  // stream binary as much as we can.  
//  char bigbuffer[10000];
  UInt version;
  Size len;
  std::string tag;

  f >> tag;
  if (tag != "TMRegion") {
    NTA_THROW << "Bad serialization for region '" << region_->getName()
              << "' of type TMRegion. Main serialization file must start "
              << "with \"TMRegion\" but instead it starts with '"
              << tag << "'";
  }
  f >> version;
  NTA_CHECK(version >= VERSION) << "TMRegion deserialization, Expecting version 1 or greater.";
  f.ignore(1);
  f >> len;
  NTA_CHECK(len == sizeof(args_)) << "TMRegion deserialization, saved size of "
                                     "structure args_ is wrong: " << len;
  f.ignore(1);
  f.read((char *)&args_, len);
  f.ignore(1);
  //f.getline(bigbuffer, sizeof(bigbuffer));
  //cellsSavePath_ = bigbuffer;
  //f.getline(bigbuffer, sizeof(bigbuffer));
  //logPathOutput_ = bigbuffer;

  if (args_.init) {
    TemporalMemory* tm = new TemporalMemory();
    tm_.reset(tm);

    tm_->load(f);
  } else {
    tm_ = nullptr;
  }
}

