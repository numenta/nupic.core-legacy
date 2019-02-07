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
 * Author: David Keeney, April 2018
 * ---------------------------------------------------------------------
 */
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/engine/Input.hpp>
#include <nupic/engine/Output.hpp>
#include <nupic/engine/Region.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/ntypes/ArrayBase.hpp>
#include <nupic/ntypes/BundleIO.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/regions/SPRegion.hpp>
#include <nupic/utils/Log.hpp>

#define VERSION 1 // version for streaming serialization format

namespace nupic {

SPRegion::SPRegion(const ValueMap &params, Region *region)
    : RegionImpl(region), computeCallback_(nullptr) {
  // Note: the ValueMap gets destroyed on return so we need to get all of the
  // parameters
  //       out of the map and set aside so we can pass them to the SpatialPooler
  //       algorithm when we create it during initialization().
  args_.columnCount = params.getScalarT<UInt32>("columnCount", 0);
  args_.potentialRadius = params.getScalarT<UInt32>("potentialRadius", 0);
  args_.potentialPct = params.getScalarT<Real32>("potentialPct", 0.5);
  args_.globalInhibition = params.getScalarT<bool>("globalInhibition", true);
  args_.localAreaDensity = params.getScalarT<Real32>("localAreaDensity", 0.0f);
  args_.numActiveColumnsPerInhArea =
      params.getScalarT<UInt32>("numActiveColumnsPerInhArea", 10);
  args_.stimulusThreshold = params.getScalarT<UInt32>("stimulusThreshold", 0);
  args_.synPermInactiveDec =
      params.getScalarT<Real32>("synPermInactiveDec", 0.008f);
  args_.synPermActiveInc = params.getScalarT<Real32>("synPermActiveInc", 0.05f);
  args_.synPermConnected = params.getScalarT<Real32>("synPermConnected", 0.1f);
  args_.minPctOverlapDutyCycles =
      params.getScalarT<Real32>("minPctOverlapDutyCycles", 0.001f);
  args_.dutyCyclePeriod = params.getScalarT<UInt32>("dutyCyclePeriod", 1000);
  args_.boostStrength = params.getScalarT<Real32>("boostStrength", 0.0f);
  args_.seed = params.getScalarT<Int32>("seed", 1);
  args_.spVerbosity = params.getScalarT<UInt32>("spVerbosity", 0);
  args_.wrapAround = params.getScalarT<bool>("wrapAround", true);
  spatialImp_ = params.getString("spatialImp", "");

  // variables used by this class and not passed on to the SpatialPooler class
  args_.learningMode = (1 == params.getScalarT<UInt32>("learningMode", true));
  args_.inferenceMode =
      (1 == params.getScalarT<UInt32>("inferenceMode", true)); // obsolete
  args_.anomalyMode =
      (1 == params.getScalarT<UInt32>("anomalyMode", true)); // obsolete
  args_.topDownMode =
      (1 == params.getScalarT<UInt32>("topDownMode", true)); // obsolete
  args_.iter = 0;

  nzInputValid_ = false;
  nzOutputValid_ = false;
}

SPRegion::SPRegion(BundleIO &bundle, Region *region) : RegionImpl(region) {
  nzInputValid_ = false;
  nzOutputValid_ = false;

  deserialize(bundle);
}

SPRegion::~SPRegion() {}

void SPRegion::initialize() {
  // Input and output buffers should already have been created by Network or deserialize.
  Array &outputBuffer = getOutput("bottomUpOut")->getData();
  UInt32 columnCount = (UInt32)outputBuffer.getCount();
  if (columnCount == 0 || outputBuffer.getBuffer() == nullptr) {
    NTA_THROW << "SPRegion::initialize - Output buffer not set.\n";
  }

  // All input links should have been initialized during Network.initialize().
  // However, if nothing is connected it might not be. The SpatialPooler
  // algorithm requires input.
  //
  // If there are more than on input link, the input buffer will be the
  // concatination of all incomming buffers.
  Array &inputBuffer = getInput("bottomUpIn")->getData();
  args_.inputWidth = (UInt32)inputBuffer.getCount();
  if (args_.inputWidth == 0) {
    NTA_THROW << "SPRegion::initialize - No input was provided.\n";
  }

  std::vector<UInt32> inputDimensions = {args_.inputWidth};
  std::vector<UInt32> columnDimensions = {columnCount};
  if (args_.potentialRadius == 0)
    args_.potentialRadius = args_.inputWidth;

  sp_ = std::unique_ptr<SpatialPooler>(new SpatialPooler(
      inputDimensions, columnDimensions, args_.potentialRadius,
      args_.potentialPct, args_.globalInhibition, args_.localAreaDensity,
      args_.numActiveColumnsPerInhArea, args_.stimulusThreshold,
      args_.synPermInactiveDec, args_.synPermActiveInc, args_.synPermConnected,
      args_.minPctOverlapDutyCycles, args_.dutyCyclePeriod, args_.boostStrength,
      args_.seed, args_.spVerbosity, args_.wrapAround));
}

void SPRegion::compute() {
  // Note: the Python code has a hook at this point to activate profiling with
  // hotshot.  This version does not provide this hook although there are
  // several C++ profilers that could be used.

  NTA_ASSERT(sp_) << "SP not initialized";

  if (args_.topDownMode) {
    // TOP-DOWN inference mode
    NTA_THROW << "Top Down Inference mode is not implemented.";
  } else {
    // BOTTOM-UP compute mode
    args_.iter++;

    // Note: The Input and Output objects are UInt32 types (containing 1's and
    // 0's).
    nzInputValid_ = false;

    /**************************************************  Not used.
    // check for reset
    bool resetSignal = false;
    Input* resetIn = region_->getInput("resetIn");
    if (resetIn)
    {
      // if there is a resetIn array and its first element is not 0 then the
    resetSignal is true. const Array& resetArray = resetIn->getData();
      NTA_CHECK(resetArray.getCount() == 1);
      resetSignal = (*((Real32*)resetArray.getBuffer()) != 0.0);
    }
    ***************************************************/

    // Call SpatialPooler
    Array &inputBuffer = getInput("bottomUpIn")->getData();
    Array &outputBuffer = getOutput("bottomUpOut")->getData();

    UInt32 *inputVector = (UInt32 *)inputBuffer.getBuffer();
    UInt32 *outputVector = (UInt32 *)outputBuffer.getBuffer();
    sp_->compute(inputVector, args_.learningMode, outputVector);

    // Prepare the output
    nzOutputValid_ = false;
  }
}

std::string SPRegion::executeCommand(const std::vector<std::string> &args,
                                     Int64 index) {
  // The Spatial Pooler does not execute any Commands.
  return "";
}

// This is the per-node output size. This determines how big the output buffers
// should be allocated to during Region::initialization(). NOTE: Some outputs
// are optional, return 0 if not used.
size_t SPRegion::getNodeOutputElementCount(const std::string &outputName) {
  if (outputName == "bottomUpOut") // This is the only output link we actually use.
  {
    const Array &out = getOutput("bottomUpOut")->getData();
    if (out.getCount())
      return out.getCount();
    else
      return args_.columnCount; // in case it was specified in the args.
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

  ns->singleNodeOnly = true; // this means we don't care about dimensions;

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
          "size of the "
          "connected potential pools of all columns).The inhibition logic will "
          "insure "
          "that at most N columns remain ON within a local inhibition area, "
          "where "
          "N = localAreaDensity * (total number of columns in inhibition "
          "area). "
          "Mutually exclusive with numActiveColumnsPerInhArea. "
          " Default ``0.0``.",
          NTA_BasicType_Real32,             // type
          1,                                // elementCount
          "",                               // constraints
          "0.0",                            // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "numActiveColumnsPerInhArea",
      ParameterSpec(
          "(int)\n"
          "An alternate way to control the density of the active columns.If "
          "numActiveColumnsPerInhArea is specified then localAreaDensity is "
          "set "
          "to 0, and vice versa. When using numActiveColumnsPerInhArea, "
          "the inhibition logic will insure that at most "
          "'numActiveColumnsPerInhArea' "
          "columns remain ON within a local inhibition area(the size of which "
          "is set by the internally calculated inhibitionRadius, which is in "
          "turn determined from the average size of the connected receptive "
          "fields "
          "of all  columns).When using this method, as columns learn and grow "
          "their "
          "effective receptive fields, the inhibitionRadius will grow, and "
          "hence the net density of the active columns will *decrease*.This is "
          "in "
          "contrast to the localAreaDensity method, which keeps the density of "
          "active "
          "columns the same regardless of the size of their receptive fields.\n"
          "Default ``10``.",
          NTA_BasicType_UInt32,             // type
          1,                                // elementCount
          "",                               // constraints
          "10",                             // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add("stimulusThreshold",
      ParameterSpec("(int)\n"
                    "This is a number specifying the minimum "
                    "number of synapses that must be "
                    "on in order for a columns to turn ON.The "
                    "purpose of this is to prevent "
                    "noise input from activating "
                    "columns.Specified as a percent of a fully "
                    " grown synapse.Default ``0``.",
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
                    ParameterSpec::ReadWriteAccess)); // access

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

  ns->parameters.add("wrapAround",
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
                    NTA_BasicType_UInt32,            // type
                    0,                               // elementCount
                    "",                              // constraints
                    "",                              // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "spOutputNonZeros",
      ParameterSpec(
          "The indices of the non-zero outputs from the spatial pooler",
          NTA_BasicType_UInt32,            // type
          0,                               // elementCount
          "",                              // constraints
          "",                              // defaultValue
          ParameterSpec::ReadOnlyAccess)); // access

  /***  not found -- were in python but not implemented in c++ sp.
      ns->parameters.add(
        "spOverlapDistribution",
        ParameterSpec("The overlaps between the active output coincidences "
          "and the input.The overlap amounts for each coincidence are sorted "
          "from highest to lowest. ",
          NTA_BasicType_Real32,                 // type
          0,                                    // elementCount
          "",                                   // constraints
          "",                                   // defaultValue
          ParameterSpec::ReadOnlyAccess));      // access

      ns->parameters.add(
        "sparseCoincidenceMatrix",
        ParameterSpec("The coincidences, as a SparseMatrix",
          NTA_BasicType_Byte,                   // type
          0,                                    // elementCount
          "",                                   // constraints
          "",                                   // defaultValue
          ParameterSpec::ReadOnlyAccess));      // access

      ns->parameters.add(
        "denseOutput",
        ParameterSpec("Score for each coincidence.",
          NTA_BasicType_Real32,                 // type
          0,                                    // elementCount
          "",                                   // constraints
          "",                                   // defaultValue
          ParameterSpec::ReadOnlyAccess));      // access

      ns->parameters.add(
        "spLearningStatsStr",
        ParameterSpec("String representation of dictionary containing a number "
          "of statistics related to learning.",
          NTA_BasicType_Byte,                   // type
          0,                                    // elementCount
          "handle",                             // constraints
          "",                                   // defaultValue
          ParameterSpec::ReadOnlyAccess));      // access
  ****/

  /* The last group is for parameters that aren't specific to spatial pooler */
  ns->parameters.add("learningMode",
                     ParameterSpec("1 if the node is learning (default 1).",
                                   NTA_BasicType_UInt32, // type
                                   1,                    // elementCount
                                   "bool",               // constraints
                                   "1",                  // defaultValue
                                   ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "inferenceMode",
      ParameterSpec("1 if the node is inferring (default 0).  obsolete.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "bool",                           // constraints
                    "",                               // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "anomalyMode",
      ParameterSpec("1 if an anomaly score is being computed. obsolete.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "bool",                           // constraints
                    "0",                              // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "topDownMode",
      ParameterSpec("1 if the node should do top down compute on the next call "
                    "to compute into topDownOut (default 0).  Obsolete.",
                    NTA_BasicType_UInt32,             // type
                    1,                                // elementCount
                    "bool",                           // constraints
                    "0",                              // defaultValue
                    ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "activeOutputCount",
      ParameterSpec("Number of active elements in bottomUpOut output.",
                    NTA_BasicType_UInt32,            // type
                    1,                               // elementCount
                    "",                              // constraints
                    "0",                             // defaultValue
                    ParameterSpec::ReadOnlyAccess)); // access

  ns->parameters.add(
      "logPathInput",
      ParameterSpec(
          "Optional name of input log file. If set, every input vector"
          " will be logged to this file.",
          NTA_BasicType_Byte,               // type
          0,                                // elementCount
          "",                               // constraints
          "",                               // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "logPathOutput",
      ParameterSpec(
          "Optional name of output log file. If set, every output vector"
          " will be logged to this file.",
          NTA_BasicType_Byte,               // type
          0,                                // elementCount
          "",                               // constraints
          "",                               // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add(
      "logPathOutputDense",
      ParameterSpec(
          "Optional name of output log file. If set, every output vector"
          " will be logged to this file as a dense vector.",
          NTA_BasicType_Byte,               // type
          0,                                // elementCount
          "",                               // constraints
          "",                               // defaultValue
          ParameterSpec::ReadWriteAccess)); // access

  ns->parameters.add("spatialImp",
                     ParameterSpec("SpatialPooler type or option. not used.",
                                   NTA_BasicType_Byte, // type
                                   0,                  // elementCount
                                   "",                 // constraints
                                   "",                 // defaultValue
                                   ParameterSpec::ReadOnlyAccess)); // access

  /* ----- inputs ------- */
  ns->inputs.add("bottomUpIn",
                 InputSpec("The input vector.",  // description
                           NTA_BasicType_UInt32, // type
                           0,                    // count.
                           true,                 // required?
                           false,                // isRegionLevel,
                           true,                 // isDefaultInput
                           false                 // requireSplitterMap
                           ));

  ns->inputs.add("topDownIn",
                 InputSpec("The top-down input signal, generated from feedback "
                           "from upper levels. not implemented.",
                           NTA_BasicType_Real32, // type
                           0,                    // count.
                           false,                // required?
                           true,                 // isRegionLevel,
                           false,                // isDefaultInput
                           false                 // requireSplitterMap
                           ));

  ns->inputs.add(
      "resetIn",
      InputSpec("A boolean flag that indicates whether "
                "or not the input vector received in this compute cycle "
                "represents the start of a new temporal sequence.  not used.",
                NTA_BasicType_Real32, // type
                1,                    // count.
                false,                // required?
                true,                 // isRegionLevel,
                false,                // isDefaultInput
                false                 // requireSplitterMap
                ));

  ns->inputs.add("sequenceIdIn", InputSpec("Sequence ID",
                                           NTA_BasicType_UInt64, // type
                                           1,                    // count.
                                           false,                // required?
                                           true,  // isRegionLevel,
                                           false, // isDefaultInput
                                           false  // requireSplitterMap
                                           ));

  /* ----- outputs ------ */
  ns->outputs.add(
      "bottomUpOut",
      OutputSpec("The output signal generated from the bottom-up inputs "
                 "from lower levels.",
                 NTA_BasicType_UInt32, // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 true                  // isDefaultOutput
                 ));

  ns->outputs.add("topDownOut",
                  OutputSpec("The top-down output signal, generated from "
                             "feedback from upper levels. \n"
                             "Not implemented.",
                             NTA_BasicType_Real32, // type
                             0,                    // count 0 means is dynamic
                             true,                 // isRegionLevel
                             false                 // isDefaultOutput
                             ));

  ns->outputs.add(
      "spatialTopDownOut",
      OutputSpec("The top-down output, generated only from the current "
                 "SP output.This can be used to evaluate how well the "
                 "SP is representing the inputs independent of the TM. \n"
                 "Not implemented.",
                 NTA_BasicType_Real32, // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 false                 // isDefaultOutput
                 ));

  ns->outputs.add(
      "temporalTopDownOut",
      OutputSpec("The top-down output, generated only from the current "
                 "TM output feedback down through the SP.\n"
                 "Not implemented.",
                 NTA_BasicType_Real32, // type
                 0,                    // count 0 means is dynamic
                 true,                 // isRegionLevel
                 false                 // isDefaultOutput
                 ));

  ns->outputs.add(
      "anomalyScore",
      OutputSpec("The score for how 'anomalous' (i.e. rare) this spatial "
                 "input pattern is.Higher values are increasingly rare. \n"
                 "Obsolete.",
                 NTA_BasicType_Real32, // type
                 1,                    // count 0 means is dynamic
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
// Most parameters are handled automatically by getParameterFromBuffer().
// The ones that need special treatment are explicitly handled here.
//
////////////////////////////////////////////////////////////////////////

UInt32 SPRegion::getParameterUInt32(const std::string &name, Int64 index) {
  switch (name[0]) {
  case 'a':
    if (name == "activeOutputCount") {
      return (UInt32)getOutput("bottomUpOut")->getData().getCount();
    }
    if (name == "anomalyMode") {
      return args_.anomalyMode;
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
    if (name == "inferenceMode") {
      return args_.inferenceMode;
    }
    break;
  case 'l':
    if (name == "learningMode") {
      return args_.learningMode;
    }
    break;
  case 'n':
    if (name == "numActiveColumnsPerInhArea") {
      if (sp_)
        return sp_->getNumActiveColumnsPerInhArea();
      else
        return args_.numActiveColumnsPerInhArea;
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
  case 't':
    if (name == "topDownMode") {
      return args_.topDownMode;
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
    getInput("bottomUpIn")->getData().convertInto(array);
  } else if (name == "spatialPoolerOutput") {
    getOutput("bottomUpOut")->getData().convertInto(array);
  } else if (name == "spInputNonZeros") {
    if (!nzInputValid_) {
      const Array &incoming = getInput("bottomUpIn")->getData();
      nzInput_ = incoming.nonZero();
      nzInputValid_ = true;
    }
    nzInput_.convertInto(array);
  } else if (name == "spOutputNonZeros") {
    if (!nzOutputValid_) {
      const Array &output = getOutput("bottomUpOut")->getData();
      nzOutput_ = output.nonZero();
      nzOutputValid_ = true;
    }
    nzOutput_.convertInto(array);
  }
  //  spOverlapDistribution not found
  //  sparseCoincidenceMatrix not found
  //  denseOutput not found
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
    if (!nzInputValid_) {
      Array a;
      getParameterArray(name, index, a); // This forces nzInput_ to be valid.
    }
    return nzInput_.getCount();
  } else if (name == "spOutputNonZeros") {
    if (!nzOutputValid_) {
      Array a;
      getParameterArray(name, index, a); // This forces nzOutput_ to be valid.
    }
    return nzOutput_.getCount();
  }
  return 0;
}

std::string SPRegion::getParameterString(const std::string &name, Int64 index) {
  if (name == "logPathInput") {
    return logPathInput_;
  }
  if (name == "logPathOutput") {
    return logPathOutput_;
  }
  if (name == "logPathOutputDense") {
    return logPathOutputDense_;
  }
  if (name == "spatialImp") {
    return spatialImp_;
  }
  // "spLearningStatsStr"  not found
  return this->RegionImpl::getParameterString(name, index);
}

void SPRegion::setParameterUInt32(const std::string &name, Int64 index,
                                  UInt32 value) {
  switch (name[0]) {
  case 'a':
    if (name == "anomalyMode") {
      args_.anomalyMode = (value != 0);
      return;
    }
    break;
  case 'd':
    if (name == "dutyCyclePeriod") {
      if (sp_)
        sp_->setDutyCyclePeriod(value);
      args_.dutyCyclePeriod = value;
      return;
    }
    break;
  case 'i':
    if (name == "inferenceMode") {
      args_.inferenceMode = (value != 0);
      return;
    }
    break;
  case 'l':
    if (name == "learningMode") {
      args_.learningMode = (value != 0);
      return;
    }
    break;
  case 'n':
    if (name == "numActiveColumnsPerInhArea") {
      if (sp_)
        sp_->setNumActiveColumnsPerInhArea(value);
      args_.numActiveColumnsPerInhArea = value;
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
  case 't':
    if (name == "topDownMode") {
      args_.topDownMode = (value != 0);
      return;
    }
    break;

    RegionImpl::setParameterUInt32(name, index, value);
  } // switch
}

void SPRegion::setParameterInt32(const std::string &name, Int64 index,
                                 Int32 value) {
  RegionImpl::setParameterInt32(name, index, value);
}

void SPRegion::setParameterReal32(const std::string &name, Int64 index,
                                  Real32 value) {
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
    if (name == "synPermConnected") {
      if (sp_)
        sp_->setSynPermConnected(value);
      args_.synPermConnected = value;
      return;
    }
    break;
  } // switch
  RegionImpl::setParameterReal32(name, index, value);
}

void SPRegion::setParameterBool(const std::string &name, Int64 index,
                                bool value) {
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

void SPRegion::setParameterString(const std::string &name, Int64 index,
                                  const std::string &s) {
  if (name == "logPathInput") {
    logPathInput_ = s;
  } else if (name == "logPathOutput") {
    logPathOutput_ = s;
  } else if (name == "logPathOutputDense") {
    logPathOutputDense_ = s;
  } else
    this->RegionImpl::setParameterString(name, index, s);
}


void SPRegion::serialize(BundleIO &bundle) {
  std::ostream &f = bundle.getOutputStream();
  // There is more than one way to do this. We could serialize to YAML, which
  // would make a readable format, or we could serialize directly to the stream
  // Choose the fastest executing one.
  f << "SPRegion " << (int)VERSION << std::endl;
  f << "args " << sizeof(args_) << " ";
  f.write((const char *)&args_, sizeof(args_));
  f << std::endl;
  f << spatialImp_ << std::endl;
  f << logPathInput_ << std::endl;
  f << logPathOutput_ << std::endl;
  f << logPathOutputDense_ << std::endl;
  f << "outputs [";
  std::map<std::string, Output *> outputs = region_->getOutputs();
  for (auto iter : outputs) {
    const Array &outputBuffer = iter.second->getData();
    if (outputBuffer.getCount() != 0) {
      f << iter.first << " ";
      outputBuffer.save(f);
    }
  }
  f << "] "; // end of all output buffers

  bool init = ((sp_) ? true : false);
  f << init << " ";
  if (init)
    sp_->save(f);
}

void SPRegion::deserialize(BundleIO &bundle) {
  std::istream &f = bundle.getInputStream();
  // There is more than one way to do this. We could serialize to YAML, which
  // would make a readable format, or we could serialize directly to the stream
  // Choose the easier one.
  char bigbuffer[5000];
  bool init;
  std::string tag;
  Size v;
  f >> tag;
  NTA_CHECK(tag == "SPRegion")
      << "Bad serialization for region '" << region_->getName()
      << "' of type SPRegion. Main serialization file must start "
      << "with \"SPRegion\" but instead it starts with '" << tag << "'";

  f >> v;
  NTA_CHECK(v >= 1)
      << "Unexpected version for SPRegion deserialization stream, "
      << region_->getName();
  f >> tag;
  NTA_CHECK(tag == "args");
  f >> v;
  NTA_CHECK(v == sizeof(args_));
  f.ignore(1);
  f.read((char *)&args_, v);
  f.ignore(1);
  f.getline(bigbuffer, sizeof(bigbuffer));
  spatialImp_ = bigbuffer;
  f.getline(bigbuffer, sizeof(bigbuffer));
  logPathInput_ = bigbuffer;
  f.getline(bigbuffer, sizeof(bigbuffer));
  logPathOutput_ = bigbuffer;
  f.getline(bigbuffer, sizeof(bigbuffer));
  logPathOutputDense_ = bigbuffer;
  f >> tag;
  NTA_CHECK(tag == "outputs");
  f.ignore(1);
  NTA_CHECK(f.get() == '['); // start of outputs
  while (true) {
    f >> tag;
    f.ignore(1);
    if (tag == "]")
      break;
    getOutput(tag)->getData().load(f);
  }
  f >> init;
  f.ignore(1);
  if (init) {
    sp_ = std::unique_ptr<SpatialPooler>(new SpatialPooler());
    sp_->load(f);
  } else
    sp_ = nullptr;
}

} // namespace nupic
