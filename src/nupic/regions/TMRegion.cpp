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
#include <nupic/ntypes/ArrayRef.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/regions/TMRegion.hpp>
#include <nupic/utils/Log.hpp>

#define VERSION 1

namespace nupic {
TMRegion::TMRegion(const ValueMap &params, Region *region)
    : RegionImpl(region), computeCallback_(nullptr) {
  // Note: the ValueMap gets destroyed on return so we need to get all of the
  // parameters
  //       out of the map and set aside so we can pass them to the SpatialPooler
  //       algorithm when we create it during initialization().
  memset((char *)&args_, 0, sizeof(args_));
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
  args_.extra =  params.getScalarT<UInt32>("extra", 0);

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
  // Network.initialize().
  //
  // If there are more than on input link, the input buffer will be the
  // concatination of all incomming buffers.
  UInt32 inputWidth = (UInt32)region_->getInputData("bottomUpIn").getCount();
  if (inputWidth == 0) {
    NTA_THROW << "TMRegion::initialize - No input was provided.\n";
  }

  nupic::algorithms::temporal_memory::TemporalMemory* tm = 
    new nupic::algorithms::temporal_memory::TemporalMemory(
      columnDimensions, args_.cellsPerColumn, args_.activationThreshold,
      args_.initialPermanence, args_.connectedPermanence, args_.minThreshold,
      args_.maxNewSynapseCount, args_.permanenceIncrement, args_.permanenceDecrement,
      args_.predictedSegmentDecrement, args_.seed, args_.maxSegmentsPerCell,
      args_.maxSynapsesPerSegment, args_.checkInputs, args_.extra);
  tm_.reset(tm);

  args_.iter = 0;
  args_.sequencePos = 0;
  args_.init = true;

  // setup the BottomUpOut output buffer in the TM to point to the one 
  // in the Output object.  Trying to avoid a copy.
  // NOTE: the TM must not delete the array. The Output object is owner.
//  Array &tmOutput = region_->getOutput("bottomUpOut")->getData();
//  tm_->setOutputBuffer((Real32*)tmOutput.getBuffer());
}

void TMRegion::compute() {
  // Note: the Python code has a hook at this point to activate profiling with
  // hotshot.
  //       This version does not provide this hook although there are several
  //       C++ profilers that could be used.
  //
  // https://github.com/numenta/nupic/blob/master/src/nupic/algorithms/temporal_memory_shim.py
/***
line 89
def compute(self, activeColumns, learn=True):
    """
    Feeds input record through TM, performing inference and learning.
    Updates member variables with new state.
    @param activeColumns (set) Indices of active columns in `t`
    """
    bottomUpInput = numpy.zeros(self.numberOfCols, dtype=dtype)
    bottomUpInput[list(activeColumns)] = 1
    super(TemporalMemoryShim, self).compute(bottomUpInput,
                                            enableLearn=learn,
                                            enableInference=True)

    predictedState = self.getPredictedState()
    self.predictiveCells = set(numpy.flatnonzero(predictedState))
***/

  NTA_ASSERT(tm_) << "TM not initialized";
  args_.iter++;

  // Handle reset signal
  Array &reset = getInput("resetIn")->getData();
  if (reset.getCount() == 1 && ((Real32 *)(reset.getBuffer()))[0] != 0) {
    tm_->reset();
    args_.sequencePos = 0; // Position within the current sequence
  }


  // Perform inference and / or learning
  Array &bottomUpIn = getInput("bottomUpIn")->getData();  // dense

  // Perform Bottom up compute()
  tm_->compute(bottomUpIn.getCount(),
               (UInt *)bottomUpIn.getBuffer(),
               args_.learningMode
  );
  args_.sequencePos++;

// TODO: this will not compile until I fix Array
//  Output *out;
//  if ((out = getOutput("bottomUpOut"))) {  
//    out->getData().fromVector(tm_->getPredictiveCells());  // sparse
//  }
//  if ((out = getOutput("activeCells"))) {
//    out->getData().fromVector(tm_->getActiveCells());   // sparse
//  }
//  if ((out = getOutput("predictedActiveCells"))) {
//    out->getData().fromVector(tm_->getWinnerCells());  // sparse
//  }
}

std::string TMRegion::executeCommand(const std::vector<std::string> &args,Int64 index) {
  // The TM does not execute any Commands.
  return "";
}

// This is the per-node output size. This determines how big the output
// buffers should be allocated to during Region::initialization(). NOTE: Some
// outputs are optional, return 0 if not used.
size_t TMRegion::getNodeOutputElementCount(const std::string &outputName) {
  if (outputName == "bottomUpOut")
    return args_.outputWidth;
  if (outputName == "activeCells")
    return args_.outputWidth;
  if (outputName == "predictedActiveCells")
    return args_.outputWidth;
  return 0; 
}

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
                    "SP, if one is used.",
                    NTA_BasicType_UInt32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "500",                         // defaultValue
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
                  NTA_BasicType_Int32,              // type
                  1,                                // elementCount
                  "",                               // constraints
                  "255",                             // defaultValue
                  ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxSynapsesPerSegment", 
      ParameterSpec("(int) The maximum number of synapses allowed in "
                    "a segment. This is used to turn on 'fixed size CLA' mode. When in "
                    "effect, 'globalDecay' is not applicable and must be set to 0, and "
                    "'maxAge' must be set to 0. When this is used (> 0), "
                    "'maxSegmentsPerCell' must also be > 0.",
                    NTA_BasicType_Int32,              // type
                    1,                                // elementCount
                    "",                               // constraints
                    "255",                             // defaultValue
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
      ParameterSpec("(int)Number of active elements in bottomUpOut dense output.",
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
                "data structure, but internally organized as a flattened vector.",
                NTA_BasicType_Real32, // type
                0,                    // count.
                true,                 // required?
                false,                // isRegionLevel,
                true,                 // isDefaultInput
                false                 // requireSplitterMap
                ));

  ns->inputs.add(
      "resetIn",
      InputSpec("A boolean flag that indicates whether "
                "or not the input vector received in this compute cycle "
                "represents the first training presentation in new temporal "
                "sequence.",
                NTA_BasicType_Real32, // type
                1,                    // count.
                false,                // required?
                true,                 // isRegionLevel,
                false,                // isDefaultInput
                false                 // requireSplitterMap
                ));


  /* ----- outputs ------ */
  ns->outputs.add(
      "bottomUpOut",
      OutputSpec("The output signal generated from the bottom-up inputs "
                 "from lower levels.",
                 NTA_BasicType_Real32, // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 true                  // isDefaultOutput
                 ));

  ns->outputs.add(
      "activeCells", 
      OutputSpec("The cells that are active",
                NTA_BasicType_Real32, // type
                0,    // count 0 means is dynamic
                true, // isRegionLevel
                false // isDefaultOutput
                ));

  ns->outputs.add(
      "predictedActiveCells",
      OutputSpec("The cells that are active and predicted",
                NTA_BasicType_Real32, // type
                0,                    // count 0 means is dynamic
                true,                 // isRegionLevel
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

  switch (name[0]) {
  case 'a':
    if (name == "activationThreshold") {
      if (tm_)
        return tm_->getActivationThreshold();
      return args_.activationThreshold;
    }
    if (name == "activeOutputCount") {
      return args_.outputWidth;
    }
    break;

  case 'c':
    if (name == "cellsPerColumn") {
      if (tm_)
        return (UInt32)tm_->getCellsPerColumn();
      return args_.cellsPerColumn;
    }
    break;

  case 'i':
    if (name == "inputWidth")
      NTA_CHECK(getInput("bottomUpIn") != nullptr) << "Unknown Input: 'bottomUpIn'";
      if (!getInput("bottomUpIn")->isInitialized())
        return 0; // might not be any links defined.
      return (UInt32)getInput("bottomUpIn")->getData().getCount();
    break;

  case 'm':
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
    break;

  case 'n':
    if (name == "numberOfCols") {
      if (tm_)
        return (UInt32)tm_->numberOfColumns();
      return args_.numberOfCols;
    }
    break;

  case 'o':
    if (name == "outputWidth")
      return args_.outputWidth;

  } // end switch
  return this->RegionImpl::getParameterUInt32(name, index); // default
}

Int32 TMRegion::getParameterInt32(const std::string &name, Int64 index) {
  if (name == "activationThreshold") {
    if (tm_)
      return tm_->getActivationThreshold();
    return args_.activationThreshold;
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
  if (name == "seed") {
    return args_.seed;
  }
  return this->RegionImpl::getParameterInt32(name, index); // default
}

Real32 TMRegion::getParameterReal32(const std::string &name, Int64 index) {

  switch (name[0]) {
  case 'c':
    if (name == "connectedPermanence") {
      if (tm_)
        return tm_->getConnectedPermanence();
      return args_.connectedPermanence;
    }
    break;

  case 'i':
    if (name == "initialPermanence") {
      if (tm_)
        return tm_->getInitialPermanence();
      return args_.initialPermanence;
    }
    break;
  case 'p':
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

    break;
  }
  return this->RegionImpl::getParameterReal32(name, index); // default
}

bool TMRegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "checkInputs") {
    if (tm_)
      return tm_->getCheckInputs();
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
    if (tm_)
      tm_->setMaxNewSynapseCount(value);
    args_.maxNewSynapseCount = value;
    return;
  }
  if (name == "minThreshold") {
    if (tm_)
      tm_->setMinThreshold(value);
    args_.minThreshold = value;
    return;
  }
  RegionImpl::setParameterUInt32(name, index, value);
}

void TMRegion::setParameterInt32(const std::string &name, Int64 index, Int32 value) {
  if (name == "activationThreshold") {
    if (tm_)
      tm_->setActivationThreshold(value);
    args_.activationThreshold = value;
    return;
  }
  if (name == "maxSynapsesPerSegment") {
    args_.maxSynapsesPerSegment = value;
    return;
  }
  RegionImpl::setParameterInt32(name, index, value);
}

void TMRegion::setParameterReal32(const std::string &name, Int64 index, Real32 value) {
  if (name == "initialPermanence") {
      if (tm_)
        tm_->setInitialPermanence(value);
      args_.initialPermanence = value;
      return;
  }
  if (name == "connectedPermanence") {
      if (tm_)
        tm_->setConnectedPermanence(value);
      args_.connectedPermanence = value;
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
  if (tm_)
    // Note: tm_ saves the output buffers
    tm_->save(f);
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
    nupic::algorithms::temporal_memory::TemporalMemory* tm = new nupic::algorithms::temporal_memory::TemporalMemory();
    tm_.reset(tm);

    tm_->load(f);
  } else
    tm_ = nullptr;
}

} // namespace nupic
