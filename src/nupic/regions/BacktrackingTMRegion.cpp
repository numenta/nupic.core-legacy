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
#include <cstring> //strncpy
#include <fstream>
#include <iomanip> // setprecision() in stream
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <nupic/algorithms/Anomaly.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/regions/BacktrackingTMRegion.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/utils/VectorHelpers.hpp>

#define VERSION 1

using namespace nupic;
using namespace nupic::algorithms::anomaly;
using namespace nupic::utils;  // for VectorHelpers
using namespace nupic::algorithms::backtracking_tm;

BacktrackingTMRegion::BacktrackingTMRegion(const ValueMap &params, 
                                           Region *region)
    : RegionImpl(region), 
      computeCallback_(nullptr) {
  // Note: the ValueMap gets destroyed on return so we need to get all of the
  // parameters
  //       out of the map and set aside so we can pass them to the SpatialPooler
  //       algorithm when we create it during initialization().
  memset((char *)&args_, 0, sizeof(args_));
  args_.numberOfCols = params.getScalarT<UInt32>("numberOfCols", 0); // normally from input.
  args_.cellsPerColumn = params.getScalarT<UInt32>("cellsPerColumn", 64);
  args_.initialPerm = params.getScalarT<Real32>("initialPerm", 0.11f);
  args_.connectedPerm = params.getScalarT<Real32>("connectedPerm", 0.50f);
  args_.minThreshold = params.getScalarT<UInt32>("minThreshold", 8);
  args_.newSynapseCount = params.getScalarT<UInt32>("newSynapseCount", 15);
  args_.permanenceInc = params.getScalarT<Real32>("permanenceInc", 0.10f);
  args_.permanenceDec = params.getScalarT<Real32>("permanenceDec", 0.10f);
  args_.permanenceMax = params.getScalarT<Real32>("permanenceMax", 1.0f);
  args_.globalDecay = params.getScalarT<Real32>("globalDecay", 0.10f);
  args_.activationThreshold = params.getScalarT<UInt32>("activationThreshold", 12);
  args_.doPooling = params.getScalarT<bool>("doPooling", false);
  args_.segUpdateValidDuration =  params.getScalarT<UInt32>("segUpdateValidDuration", 5);
  args_.burnIn = params.getScalarT<UInt32>("burnIn", 2);
  args_.collectStats = params.getScalarT<bool>("collectStats", false);
  args_.seed = params.getScalarT<Int32>("seed", 42);
  args_.verbosity = params.getScalarT<UInt32>("verbosity", 0);
  args_.checkSynapseConsistency = params.getScalarT<bool>("checkSynapseConsistency", false);
  args_.pamLength = params.getScalarT<UInt32>("pamLength", 1);
  args_.maxInfBacktrack = params.getScalarT<UInt32>("maxInfBacktrack", 10);
  args_.maxLrnBacktrack = params.getScalarT<UInt32>("maxLrnBacktrack", 5);
  args_.maxAge = params.getScalarT<UInt32>("maxAge", 100000);
  args_.maxSeqLength = params.getScalarT<UInt32>("maxSeqLength", 32);
  args_.maxSegmentsPerCell = params.getScalarT<Int32>("maxSegmentsPerCell", -1);
  args_.maxSynapsesPerSegment = params.getScalarT<Int32>("maxSynapsesPerSegment", -1);

  memset((void *)args_.outputType, 0, sizeof(args_.outputType));
  strncpy(args_.outputType, params.getString("outputType", "normal").c_str(), sizeof(args_.outputType));

  // variables used by this class and not passed on
  args_.learningMode = params.getScalarT<bool>("learningMode", true);
  args_.inferenceMode = params.getScalarT<bool>("inferenceMode", false);
  args_.anomalyMode = params.getScalarT<bool>("anomalyMode", false);
  args_.topDownMode = params.getScalarT<bool>("topDownMode", false);
  args_.storeDenseOutput = params.getScalarT<bool>("storeDenseOutput", false);
  args_.computePredictedActiveCellIndices = params.getScalarT<bool>("computePredictedActiveCellIndices", false);
  args_.orColumnOutputs = params.getScalarT<bool>("orColumnOutputs", false);


  args_.iter = 0;
  args_.sequencePos = 0;
  args_.outputWidth = args_.numberOfCols * args_.cellsPerColumn;
  args_.init = false;
  tm_ = nullptr;
}

BacktrackingTMRegion::BacktrackingTMRegion(BundleIO &bundle, Region *region) 
  : RegionImpl(region) {
  tm_ = nullptr;
  deserialize(bundle);
}

void BacktrackingTMRegion::initialize() {

  // All input links and buffers should have been initialized during
  // Network.initialize().
  //
  // If there are more than on input link, the input buffer will be the
  // concatination of all incomming buffers.
  UInt32 inputWidth = (UInt32)region_->getInputData("bottomUpIn").getCount();
  if (inputWidth == 0) {
    NTA_THROW << "TMRegion::initialize - No input was provided.\n";
  }
  if (args_.numberOfCols == 0) {
    args_.numberOfCols = inputWidth;
  }
  BacktrackingTM* tm(
     new BacktrackingTM(
      args_.numberOfCols, args_.cellsPerColumn, args_.initialPerm,
      args_.connectedPerm, args_.minThreshold, args_.newSynapseCount,
      args_.permanenceInc, args_.permanenceDec, args_.permanenceMax,
      args_.globalDecay, args_.activationThreshold, args_.doPooling,
      args_.segUpdateValidDuration, args_.burnIn, args_.collectStats,
      args_.seed, args_.verbosity, args_.checkSynapseConsistency,
      args_.pamLength, args_.maxInfBacktrack, args_.maxLrnBacktrack,
      args_.maxAge, args_.maxSeqLength, args_.maxSegmentsPerCell,
      args_.maxSynapsesPerSegment, args_.outputType));
  tm_.reset(tm);
  args_.iter = 0;
  args_.sequencePos = 0;
  args_.init = true;

  //   setup the BottomUpOut output buffer in the TM to point to the one 
  //   in the Output object.  Trying to avoid a copy.
  // Array &tmOutput = region_->getOutput("bottomUpOut")->getData();
  // tm_->setOutputBuffer((Real32*)tmOutput.getBuffer());
  //    Let's copy the buffer on each compute for now. Buffer size will be
  //    too small if args_.orColumnOutputs is set.
}

void BacktrackingTMRegion::compute() {
  // Note: the Python code has a hook at this point to activate profiling with
  // hotshot.
  //       This version does not provide this hook although there are several
  //       C++ profilers that could be used.

  NTA_ASSERT(tm_) << "TM not initialized";
  args_.iter++;

  // Handle reset signal
  Array &reset = getInput("resetIn")->getData();
  if (reset.getCount() == 1 && ((Real32 *)(reset.getBuffer()))[0] != 0) {
    tm_->reset();
    args_.sequencePos = 0; // Position within the current sequence
  }

  if (args_.computePredictedActiveCellIndices) {
    prevPredictedState_ = tm_->getPredictedState(); // returns a vector<UInt32>
  }
  if (args_.anomalyMode) {
    Array p(NTA_BasicType_Real32, tm_->topDownCompute(), args_.numberOfCols);
    prevPredictedColumns_ = VectorHelpers::binaryToSparse(p.asVector<Real32>());
  }

  // Perform inference and / or learning
  Array &bottomUpIn = getInput("bottomUpIn")->getData();
  Array &tmOutput = getOutput("bottomUpOut")->getData();

  // Perform Bottom up compute()
  Real *output = tm_->compute((Real32 *)bottomUpIn.getBuffer(),
                              args_.learningMode, args_.inferenceMode);
  args_.sequencePos++;

  Real32 *ptr = (Real32 *)tmOutput.getBuffer();
  if (args_.orColumnOutputs) {
    // OR'ing together the output cells in each column?
    // This reduces the output buffer size to [columnCount] otherwise
    // The size is [columnCount X cellsPerColumn].
    NTA_ASSERT(tmOutput.getCount() == args_.numberOfCols);
    for (size_t i = 0; i < args_.numberOfCols; i++) {
      Real32 sum = 0.0f;
      for (size_t j = 0; j < args_.cellsPerColumn; j++) {
        sum += output[(i * args_.cellsPerColumn) + j];
      }
      ptr[i] = (sum == 0.0f) ? 0.0f : 1.0f;
    }
  } else {
    // copy tm buffer to bottomUpOut buffer. 
    NTA_ASSERT(tmOutput.getCount() == args_.numberOfCols * args_.cellsPerColumn);
    for (size_t i = 0; i < args_.numberOfCols * args_.cellsPerColumn; i++) {
      ptr[i] = output[i];
    }
  }

  if (args_.topDownMode) {
    // Top - down compute
    Real *tdout = tm_->topDownCompute();
    getOutput("topDownOut")->getData() = Array(NTA_BasicType_Real32, 
                                               tdout, 
                                               args_.numberOfCols);
  }

  // Set output for use with anomaly classification region if in anomalyMode
  if (args_.anomalyMode) {
    Byte *lrn = tm_->getLearnActiveStateT();
    Size size = args_.numberOfCols * args_.cellsPerColumn;
    getOutput("lrnActiveStateT")->getData() = Array(NTA_BasicType_Byte, lrn, size);

    auto activeColumns = VectorHelpers::binaryToSparse(bottomUpIn.asVector<Real32>());
    Real32 anomalyScore = algorithms::anomaly::computeRawAnomalyScore(
        activeColumns, prevPredictedColumns_);
    getOutput("anomalyScore")->getData() = Array(NTA_BasicType_Real32, &anomalyScore, 1);
  }

  if (args_.computePredictedActiveCellIndices) {
    Output *activeCells = getOutput("activeCells");
    Output *predictedActiveCells = getOutput("predictedActiveCells");
    Byte *activeState = tm_->getActiveState();
    Size nCells = args_.numberOfCols * args_.cellsPerColumn;
    NTA_ASSERT(activeCells != nullptr);
    NTA_ASSERT(predictedActiveCells != nullptr);
    NTA_ASSERT(args_.outputWidth == nCells);
    NTA_ASSERT(args_.outputWidth == activeCells->getData().getCount());
    NTA_ASSERT(args_.outputWidth == predictedActiveCells->getData().getCount());

    Real32 *activeCellsPtr = (Real32 *)activeCells->getData().getBuffer();
    Real32 *predictedActiveCellsPtr =  (Real32 *)predictedActiveCells->getData().getBuffer();
    for (size_t idx = 0; idx < nCells; idx++) {
      activeCellsPtr[idx] = (activeState[idx]) ? 1.0f : 0.0f;
      predictedActiveCellsPtr[idx] = (prevPredictedState_[idx] && activeState[idx]) ? 1.0f : 0.0f;
    }
  }
}

std::string BacktrackingTMRegion::executeCommand(const std::vector<std::string> &args,
                                     Int64 index) {
  // The TM does not execute any Commands.
  return "";
}

// This is the per-node output size. This determines how big the output
// buffers should be allocated to during Region::initialization(). NOTE: Some
// outputs are optional, return 0 if not used.
size_t BacktrackingTMRegion::getNodeOutputElementCount(const std::string &outputName) const {
  if (outputName == "bottomUpOut")
    return (args_.orColumnOutputs)? args_.numberOfCols: args_.outputWidth;
  if (outputName == "topDownOut")
    return args_.numberOfCols;
  if (outputName == "lrnActiveStateT")
    return args_.outputWidth;
  if (outputName == "activeCells")
    return args_.outputWidth;
  if (outputName == "predictedActiveCells")
    return args_.outputWidth;
  return 0; // an optional output that we don't use.
}
// Note: - this is called during Region initialization, after configuration
//         is set but prior to calling initialize on this class to create the tm. 
//         The input dimensions should already have been set, normally from 
//         its connected output. This would set the region dimensions if not overridden.
//       - This is not called if output dimensions were explicitly set for this output.
//       - This call determines the dimensions set on the Output buffers.
Dimensions BacktrackingTMRegion::askImplForOutputDimensions(const std::string &name) {
  Dimensions region_dim = getDimensions();
  if (!region_dim.isSpecified()) {
    // we don't have region dimensions, so create some if we know numberOfCols.
    if (args_.numberOfCols == 0) 
      return Dimensions(Dimensions::DONTCARE);  // No info about its size
    region_dim.clear();
    region_dim.push_back(args_.numberOfCols);
    setDimensions(region_dim);
  }
  if (args_.numberOfCols == 0)
    args_.numberOfCols = (UInt32)region_dim.getCount();
  else
    NTA_CHECK(args_.numberOfCols == (UInt32)region_dim.getCount())
          << "A configured 'numberOfCols' is not consistant with incoming data dimensions.";


  if (name == "bottomUpOut" && args_.orColumnOutputs) {
    return region_dim;
  } else if (name == "bottomUpOut"  || name == "topDownOut" || name == "lrnActiveStateT"
   || name == "activeCells" || name == "predictedActiveCells") {
    // It's size is numberOfCols * args_.cellsPerColumn.
    // So insert a new dimension to what was provided by input.
    Dimensions dim = region_dim;
    dim.insert(dim.begin(), args_.cellsPerColumn);
    return dim;
  }
  return RegionImpl::askImplForOutputDimensions(name);
}

/********************************************************************/

Spec *BacktrackingTMRegion::createSpec() {
  auto ns = new Spec;

  ns->description =
      "BacktrackingTMRegion. Class implementing the temporal memory algorithm as "
      "described in 'BAMI "
      "<https://numenta.com/biological-and-machine-intelligence/>'.  "
      "The implementation here attempts to closely match the pseudocode in "
      "the documentation. This implementation does contain several additional "
      "bells and whistles such as a column confidence measure.";


  /* ---- parameters ------ */

  /* constructor arguments */
  ns->parameters.add(
      "numberOfCols",
      ParameterSpec("(int) Number of mini-columns in the region. This values "
                    "needs to be the same as the number of columns in the "
                    "SP, if one is "
                    "used.",
                    NTA_BasicType_UInt32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "",                            // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "cellsPerColumn",
      ParameterSpec("(int) The number of cells per mini-column.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "",                   // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "initialPerm",
      ParameterSpec("(float) Initial permanence for newly created synapses.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.11",                        // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "connectedPerm",
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
      "newSynapseCount",
      ParameterSpec("(int) The max number of synapses added "
                    "to a segment during learning.",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "15",                 // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "permanenceInc", // permInc in Cells4
      ParameterSpec("(float) Active synapses get their permanence counts "
                    "incremented by this value.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.1",                         // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "permanenceDec", // permDec in Cells4
      ParameterSpec("(float) All other synapses get their permanence counts "
                    "decremented by this value.",
                    NTA_BasicType_Real32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "0.1",                         // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "permanenceMax", // permMax in Cells4
      ParameterSpec("(float) ",
                    NTA_BasicType_Real32, // type
                    1,                    // elementCount
                    "",                   // constraints
                    "1",                  // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "globalDecay",
      ParameterSpec(
          "(float) Value to decrease permanences when the global "
          "decay process runs. Global decay will remove synapses if their "
          "permanence value reaches 0. It will also remove segments when "
          "they no "
          "longer have synapses. \n"
          "Note:: Global decay is applied after 'maxAge' iterations, after "
          "which it will run every `'maxAge' iterations.",
          NTA_BasicType_Real32,          // type
          1,                             // elementCount
          "",                            // constraints
          "0.10",                        // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "activationThreshold",
      ParameterSpec("(int) Number of synapses that must be active to "
                    "activate a segment.",
                    NTA_BasicType_UInt32,          // type
                    1,                             // elementCount
                    "",                            // constraints
                    "12",                          // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "doPooling",
      ParameterSpec("(bool) If True, pooling is enabled. False is the default.",
                    NTA_BasicType_Bool,            // type
                    1,                             // elementCount
                    "bool",                        // constraints
                    "false",                       // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "segUpdateValidDuration",
      ParameterSpec("(int) ",
                    NTA_BasicType_UInt32, // type
                    1,                    // elementCount
                    "<=5",                   // constraints
                    "5",                  // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "burnIn", // not in Cells4
      ParameterSpec(
          "(int) Used for evaluating the prediction score. Default is 2. ",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "2",                              // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "collectStats", // not in Cells4
      ParameterSpec("(bool) If True, collect training / "
                    "inference stats.  Default is False. ",
                    NTA_BasicType_Bool, // type
                    1,                  // elementCount
                    "bool",             // constraints
                    "false",            // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "seed",
      ParameterSpec(
          "(int)  Random number generator seed. The seed affects the random "
          "aspects of initialization like the initial permanence values. A "
          "fixed value ensures a reproducible result.",
          NTA_BasicType_Int32,              // type
          1,                                // elementCount
          "",                               // constraints
          "42",                             // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "verbosity", // not in Cells4
      ParameterSpec(
          "(int) Controls the verbosity of the TM diagnostic output: \n"
          "- verbosity == 0: silent \n"
          "- verbosity in [1..6]: increasing levels of verbosity.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0",                              // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "checkSynapseConsistency", // not in Cells4
      ParameterSpec("(bool)   Default is False. ",
                    NTA_BasicType_Bool, // type
                    1,                  // elementCount
                    "bool",             // constraints
                    "false",            // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "pamLength", // not in Cells4
      ParameterSpec(
          "(int) Number of time steps to remain in \"Pay Attention Mode\" "
          "after we detect we've reached the end of a learned sequence. "
          "Setting this to 0 disables PAM mode. When we are in PAM mode, we "
          "do not burst unpredicted columns during learning, which in turn "
          "prevents us from falling into a previously learned sequence for a "
          "while (until we run through another 'pamLength' steps). \n"
          "\n"
          "The advantage of PAM mode is that it requires fewer presentations "
          "to learn a set of sequences which share elements. The disadvantage "
          "of PAM mode is that if a learned sequence is immediately followed "
          "by set "
          "set of elements that should be learned as a 2nd sequence, the first "
          "'pamLength' elements of that sequence will not be learned as part "
          "of that 2nd sequence.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "1",                              // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxInfBacktrack", // not in Cells4
      ParameterSpec("(int) How many previous inputs to keep in a buffer for "
                    "inference backtracking.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "",                               // constraints
                    "10",                             // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxLrnBacktrack", // not in Cells4
      ParameterSpec("(int) How many previous inputs to keep in a buffer for "
                    "learning backtracking.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "",                               // constraints
                    "5",                              // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxAge", // not in Cells4
      ParameterSpec(
          "(int) Number of iterations before global decay takes effect. "
          "Also the global decay execution interval. After global decay "
          "starts, it will run again every 'maxAge' iterations. If "
          "'maxAge==1', global decay is applied to every iteration to every "
          "segment. \n"
          "Note: Using 'maxAge > 1' can significantly speed up the TM when "
          "global decay is used. Default=100000.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "100000",                         // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxSeqLength", // not in Cells4
      ParameterSpec(
          "(int) If not 0, we will never learn more than "
          "'maxSeqLength' inputs in a row without starting over at start "
          "cells. This sets an upper bound on the length of learned sequences "
          "and thus is another means (besides `'maxAge' and 'globalDecay') by "
          "which to limit how much the TM tries to learn. ",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "32",                             // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxSegmentsPerCell", // not in Cells4
      ParameterSpec(
          "(int) The maximum number of segments allowed on a "
          "cell. This is used to turn on 'fixed size CLA' mode. When in "
          "effect, 'globalDecay' is not applicable and must be set to 0 and "
          "'maxAge' must be set to 0. When this is used (> 0), "
          "'maxSynapsesPerSegment' must also be > 0. ",
          NTA_BasicType_Int32,              // type
          1,                                // elementCount
          "",                               // constraints
          "-1",                             // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "maxSynapsesPerSegment", // not in Cells4
      ParameterSpec(
          "(int) The maximum number of synapses allowed in "
          "a segment. This is used to turn on 'fixed size CLA' mode. When in "
          "effect, 'globalDecay' is not applicable and must be set to 0, and "
          "'maxAge' must be set to 0. When this is used (> 0), "
          "'maxSegmentsPerCell' must also be > 0.",
          NTA_BasicType_Int32,              // type
          1,                                // elementCount
          "",                               // constraints
          "-1",                             // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "outputType", // not in Cells4
      ParameterSpec(
          "(string) Can be one of the following (default 'normal'):\n"
          " - 'normal': output the OR of the active and predicted state. \n"
          " - 'activeState': output only the active state. \n"
          " - 'activeState1CellPerCol': output only the active state, and at "
          "most 1 cell/column. If more than 1 cell is active in a column, the "
          "one  with the highest confidence is sent up.  ",
          NTA_BasicType_Byte,               // type
          0,                                // elementCount
          "",                               // constraints
          "normal",                         // defaultValue
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
      "orColumnOutputs",
      ParameterSpec(
          "(bool) OR together the cell outputs from each column to produce "
          "the temporal memory output. When this mode is enabled, the number "
          "of cells per column must also be specified and the output size of "
          "the region should be set the same as columnCount",
          NTA_BasicType_Bool,              // type
          1,                               // elementCount
          "bool",                          // constraints
          "false",                         // defaultValue
          ParameterSpec::ReadWriteAccess)); // access


  /* The last group is for parameters that aren't specific to spatial pooler
   */
  ns->parameters.add(
      "learningMode",
      ParameterSpec("1 if the node is learning (default true).",
                    NTA_BasicType_Bool, // type
                    1,                  // elementCount
                    "bool",             // constraints
                    "true",             // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "inferenceMode",
      ParameterSpec("True if the node is inferring (default false).  obsolete.",
                    NTA_BasicType_Bool,            // type
                    1,                             // elementCount
                    "bool",                        // constraints
                    "false",                       // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "anomalyMode",
      ParameterSpec("True if an anomaly score is being computed. obsolete.",
                    NTA_BasicType_Bool,            // type
                    1,                             // elementCount
                    "bool",                        // constraints
                    "false",                       // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "topDownMode",
      ParameterSpec(
          "True if the node should do top down compute on the next call "
          "to compute into topDownOut (default false).",
          NTA_BasicType_Bool,            // type
          1,                             // elementCount
          "bool",                        // constraints
          "false",                       // defaultValue
          ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "computePredictedActiveCellIndices",
      ParameterSpec("True if active and predicted active indices should be "
                    "computed (default false).",
                    NTA_BasicType_Bool,            // type
                    1,                             // elementCount
                    "bool",                        // constraints
                    "false",                       // defaultValue
                    ParameterSpec::CreateAccess)); // access

  ns->parameters.add(
      "activeOutputCount",
      ParameterSpec("(int)Number of active elements in bottomUpOut output.",
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
                NTA_BasicType_Real32,    // type
                0,                    // count.
                true,                 // required?
                true,                 // isRegionLevel,
                true                  // isDefaultInput
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
                false,                // isRegionLevel,
                false                 // isDefaultInput
                ));

  ns->inputs.add("sequenceIdIn", InputSpec("Sequence ID",
                NTA_BasicType_UInt64, // type
                1,                    // count.
                false,                // required?
                false,                // isRegionLevel,
                false                 // isDefaultInput
                ));

  /* ----- outputs ------ */
  ns->outputs.add(
      "bottomUpOut",
      OutputSpec("The output signal generated from the bottom-up inputs "
                 "from lower levels.",
                 NTA_BasicType_Real32,    // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 true                  // isDefaultOutput
                 ));

  ns->outputs.add(
    "topDownOut",
     OutputSpec("The top-down output signal, generated from "
                  "feedback from upper levels.  ",
                  NTA_BasicType_Real32,    // type
                  0,                    // count 0 means is dynamic
                  true,                 // isRegionLevel
                  false                 // isDefaultOutput
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

  ns->outputs.add(
      "anomalyScore",
      OutputSpec("The score for how 'anomalous' (i.e. rare) the current "
                 "sequence is. Higher values are increasingly rare.",
                 NTA_BasicType_Real32, // type
                 1,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 false                 // isDefaultOutput
                 ));

  ns->outputs.add(
      "lrnActiveStateT",
      OutputSpec("Active cells during learn phase at time t.  This is "
                 "used for anomaly classification.",
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

UInt32 BacktrackingTMRegion::getParameterUInt32(const std::string &name, Int64 index) {

  NTA_CHECK(!name.empty()) << "name must not be empty";
    if (name == "activationThreshold") {
      if (tm_) return tm_->getActivationThreshold();
      return args_.activationThreshold;
    }
    else if (name == "activeOutputCount") {
      return args_.outputWidth;
    }
    else if (name == "burnIn") {
      if (tm_) return tm_->getBurnIn();
      return args_.burnIn;
    }
    else if (name == "cellsPerColumn") {
      if (tm_)
        return (UInt32)tm_->getcellsPerCol();
      return args_.cellsPerColumn;
    }
    else if (name == "inputWidth") {
      NTA_CHECK(getInput("bottomUpIn") != nullptr) << "Unknown Input: 'bottomUpIn'";
      if (!getInput("bottomUpIn")->isInitialized()) {
        return 0; // might not be any links defined.
      }
      return (UInt32)getInput("bottomUpIn")->getData().getCount();
    }
    else if (name == "maxAge") {
      if (tm_)
        return tm_->getMaxAge();
      return args_.maxAge;
    }
    else if (name == "maxInfBacktrack") {
      if (tm_)
        return tm_->getMaxInfBacktrack();
      return args_.maxInfBacktrack;
    }
    else if (name == "maxLrnBacktrack") {
      if (tm_)
        return tm_->getMaxLrnBacktrack();
      return args_.maxLrnBacktrack;
    }
    else if (name == "minThreshold") {
      if (tm_)
        return tm_->getMinThreshold();
      return args_.minThreshold;
    }
    else if (name == "maxSeqLength") {
      if (tm_)
        return tm_->getMaxSeqLength();
      return args_.maxSeqLength;
    }
    else if (name == "numberOfCols") {
      if (tm_)
        return (UInt32)tm_->getnumCol();
      return args_.numberOfCols;
    }
    else if (name == "newSynapseCount") {
      if (tm_)
        return tm_->getNewSynapseCount();
      return args_.newSynapseCount;
    }
    else if (name == "outputWidth") {
      return args_.outputWidth;
    }
    else if (name == "pamLength") {
      if (tm_)
        return tm_->getPamLength();
      return args_.pamLength;
    }
    else if (name == "segUpdateValidDuration") {
      if (tm_)
        return tm_->getSegUpdateValidDuration();
      return args_.segUpdateValidDuration;
    }
    else if (name == "verbosity") {
      if (tm_)
        return tm_->getVerbosity();
      return args_.verbosity;
    } else {
      return this->RegionImpl::getParameterUInt32(name, index); // default
    }
}

Int32 BacktrackingTMRegion::getParameterInt32(const std::string &name, Int64 index) {
  if (name == "maxSegmentsPerCell") {
    if (tm_)
      return tm_->getMaxSegmentsPerCell();
    return args_.maxSegmentsPerCell;
  }
  else if (name == "maxSynapsesPerSegment") {
    if (tm_)
      return tm_->getMaxSynapsesPerSegment();
    return args_.maxSynapsesPerSegment;
  }
  else if (name == "seed") {
    if (tm_)
      return tm_->getSeed();
    return args_.seed;
  } else {
    return this->RegionImpl::getParameterInt32(name, index); // default
  }
}

Real32 BacktrackingTMRegion::getParameterReal32(const std::string &name, Int64 index) {

    if (name == "connectedPerm") {
      if (tm_)
        return tm_->getConnectedPerm();
      return args_.connectedPerm;
    }
    else if (name == "globalDecay") {
      if (tm_)
        return tm_->getGlobalDecay();
      return args_.globalDecay;
    }
    else if (name == "initialPerm") {
      if (tm_)
        return tm_->getInitialPerm();
      return args_.initialPerm;
    }
    else if (name == "permanenceInc") {
      if (tm_)
        return tm_->getPermanenceInc();
      return args_.permanenceInc;
    }
    else if (name == "permanenceDec") {
      if (tm_)
        return tm_->getPermanenceDec();
      return args_.permanenceDec;
    }
    else if (name == "permanenceMax") {
      if (tm_)
        return tm_->getPermanenceMax();
      return args_.permanenceMax;
    } else {
      return this->RegionImpl::getParameterReal32(name, index); // default
    }
}


bool BacktrackingTMRegion::getParameterBool(const std::string &name, Int64 index) {
  if (name == "anomalyMode") {
    return args_.anomalyMode;
  }
  else if (name == "collectStats") {
    if (tm_)
      return tm_->getCollectStats();
    return args_.collectStats;
  }
  else if (name == "checkSynapseConsistency") {
    if (tm_)
      return tm_->getCheckSynapseConsistency();
    return args_.checkSynapseConsistency;
  }
  else if (name == "computePredictedActiveCellIndices") {
    return args_.computePredictedActiveCellIndices;
  }
  else if (name == "doPooling") {
    if (tm_)
      return tm_->getDoPooling();
    return args_.doPooling;
  }
  else if (name == "learningMode")
    return args_.learningMode;
  else if (name == "inferenceMode")
    return args_.inferenceMode;
  else if (name == "orColumnOutputs")
    return args_.orColumnOutputs;
  else if (name == "topDownMode")
    return args_.topDownMode;
  else if (name == "storeDenseOutput")
    return args_.storeDenseOutput;
  else {
    return this->RegionImpl::getParameterBool(name, index); // default
  }
}


std::string BacktrackingTMRegion::getParameterString(const std::string &name, Int64 index) {
  if (name == "outputType") {
    if (tm_)
      return tm_->getOutputType();
    return args_.outputType;
  }
  else 
    return this->RegionImpl::getParameterString(name, index);
}

void BacktrackingTMRegion::setParameterUInt32(const std::string &name, Int64 index, UInt32 value) {
    if (name == "burnIn") {
      args_.burnIn = value;
    }
    if (name == "maxInfBacktrack") {
      if (tm_)
        tm_->setMaxInfBacktrack(value);
      args_.maxInfBacktrack = value;
      return;
    }
    if (name == "maxLrnBacktrack") {
      if (tm_)
        tm_->setMaxLrnBacktrack(value);
      args_.maxLrnBacktrack = value;
      return;
    }
    if (name == "maxAge") {
      if (tm_)
        tm_->setMaxAge(value);
      args_.maxAge = value;
      return;
    }
    if (name == "maxSeqLength") {
      if (tm_)
        tm_->setMaxSeqLength(value);
      args_.maxSeqLength = value;
      return;
    }
    if (name == "pamLength") {
      if (tm_)
        tm_->setPamLength( value);
      args_.pamLength = value;
      return;
    }
    if (name == "segUpdateValidDuration") {
      NTA_CHECK(!tm_) << "Cannot set segUpdateValidDuration after initialization.";
      args_.segUpdateValidDuration = value;
      return;
    }
    if (name == "verbosity") {
      if (tm_)
        tm_->setVerbosity(value);
      args_.verbosity = value;
      return;
    }

  RegionImpl::setParameterUInt32(name, index, value); //default
}


void BacktrackingTMRegion::setParameterInt32(const std::string &name, Int64 index,
                                 Int32 value) {
  if (name == "maxSegmentsPerCell") {
    if (tm_)
      tm_->setMaxSegmentsPerCell(value);
    args_.maxSegmentsPerCell = value;
    return;
  }
  if (name == "maxSynapsesPerSegment") {
    if (tm_)
      tm_->setMaxSynapsesPerSegment(value);
    args_.maxSynapsesPerSegment = value;
    return;
  }
  RegionImpl::setParameterInt32(name, index, value);
}


void BacktrackingTMRegion::setParameterBool(const std::string &name, Int64 index, bool value) 
{

  if (name == "checkSynapseConsistency") {
    if (tm_)
      tm_->setCheckSynapseConsistency(value);
    args_.checkSynapseConsistency = value;
    return;
  }
  if (name == "collectStats") {
    if (tm_)
      tm_->setCollectStats(value);
    args_.collectStats = value;
    return;
  }
  if (name == "learningMode") {
    args_.learningMode = value;
    return;
  }
  if (name == "inferenceMode") {
    args_.inferenceMode = value;
    return;
  }
  if (name == "anomalyMode") {
    args_.anomalyMode = value;
    return;
  }
  if (name == "topDownMode") {
    args_.topDownMode = value;
    return;
  }
  if (name == "storeDenseOutput") {
    args_.storeDenseOutput = value;
    return;
  }
  if (name == "computePredictedActiveCellIndices") {
    args_.computePredictedActiveCellIndices = value;
    return;
  }
  if (name == "orColumnOutputs") {
    args_.orColumnOutputs = value;
    return;
  }

  RegionImpl::setParameterBool(name, index, value);
}


void BacktrackingTMRegion::setParameterString(const std::string &name, Int64 index,
                                  const std::string &value) {
    this->RegionImpl::setParameterString(name, index, value);
}


void BacktrackingTMRegion::serialize(BundleIO &bundle) {
  std::ostream &f = bundle.getOutputStream();
  save(f);
}

void BacktrackingTMRegion::save(std::ostream& f) const {
  // There is more than one way to do this. We could serialize to YAML, which
  // would make a readable format, or we could serialize directly to the
  // stream Choose the easier one.
  UInt version = VERSION;

  f << "BacktrackingTMRegion " << version << std::endl;
  f << sizeof(args_) << " ";
  f.write((const char*)&args_, sizeof(args_));
  f << std::endl;
  if (tm_)
    // Note: tm_ saves the output buffers
    tm_->save(f);
}


void BacktrackingTMRegion::deserialize(BundleIO &bundle) {
  std::istream &f = bundle.getInputStream();
  load(f);
}

void BacktrackingTMRegion::load(std::istream &f) {
  UInt version;
  Size len;
  std::string tag;

  f >> tag;
  if (tag != "BacktrackingTMRegion") {
    NTA_THROW << "Bad serialization for region '" << region_->getName()
              << "' of type BacktrackingTMRegion. Main serialization file must start "
              << "with \"BacktrackingTMRegion\" but instead it starts with '"
              << tag << "'";
  }
  f >> version;
  NTA_CHECK(version >= VERSION) << "BacktrackingTMRegion deserialization, Expecting version 1 or greater.";
  f.ignore(1);
  f >> len;
  NTA_CHECK(len == sizeof(args_)) << "BacktrackingTMRegion deserialization, saved size of "
                                     "structure args_ is wrong: " << len;
  f.ignore(1);
  f.read((char *)&args_, len);
  f.ignore(1);

  if (args_.init) {
    auto tm = new nupic::algorithms::backtracking_tm::BacktrackingTM();
    tm_.reset(tm);

    Array &tmOutput = region_->getOutput("bottomUpOut")->getData();
    if (tmOutput.getCount() == 0) {
      Dimensions d = askImplForOutputDimensions("bottomUpOut");
      tmOutput.allocateBuffer(d.getCount());
    }
    //tm_->setOutputBuffer((Real32*)tmOutput.getBuffer());

    tm_->load(f);
  } else
    // it was saved before initialization.
    tm_.reset();
}

