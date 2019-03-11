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
 * Author: David Keeney, June, 2018  (ported from Python)
 * ---------------------------------------------------------------------
 */

/** @file
 * Declarations for TMRegion class
 */

//----------------------------------------------------------------------

#ifndef NTA_BACKTRACKING_TMREGION_HPP
#define NTA_BACKTRACKING_TMREGION_HPP

#include <memory>
#include <nupic/engine/RegionImpl.hpp>
#include <nupic/engine/Spec.hpp>
#include <nupic/ntypes/Value.hpp>
#include <nupic/types/Serializable.hpp>
#include <nupic/algorithms/BacktrackingTM.hpp>
//----------------------------------------------------------------------

namespace nupic {
class BacktrackingTMRegion : public RegionImpl, Serializable {
public:
  typedef void (*computeCallbackFunc)(const std::string &);
  typedef std::map<std::string, Spec> SpecMap;

  BacktrackingTMRegion(const ValueMap &params, Region *region);
  BacktrackingTMRegion(BundleIO &bundle, Region *region);
  virtual ~BacktrackingTMRegion() {}

  /* -----------  Required RegionImpl Interface methods ------- */

  // Used by RegionImplFactory to create and cache
  // a nodespec. Ownership is transferred to the caller.
  static Spec *createSpec();

  std::string getNodeType() { return "BacktrackingTMRegion"; };

  // Compute outputs from inputs and internal state
  void compute() override;
  std::string executeCommand(const std::vector<std::string> &args,
                             Int64 index) override;

  /**
   * Inputs/Outputs are made available in initialize()
   * It is always called after the constructor (or load from serialized state)
   */
  void initialize() override;

  void serialize(BundleIO &bundle) override;
  void deserialize(BundleIO &bundle) override;
  void save(std::ostream& f) const override;
  void load(std::istream& f) override;


  // Per-node size (in elements) of the given output.
  // For per-region outputs, it is the total element count.
  // This method is called only for outputs whose size is not
  // specified in the spec.
  size_t getNodeOutputElementCount(const std::string &outputName) const override;
  Dimensions askImplForOutputDimensions(const std::string &name) override;



  /* -----------  Optional RegionImpl Interface methods ------- */
  UInt32 getParameterUInt32(const std::string &name, Int64 index) override;
  Int32 getParameterInt32(const std::string &name, Int64 index) override;
  Real32 getParameterReal32(const std::string &name, Int64 index) override;
  bool getParameterBool(const std::string &name, Int64 index) override;
  std::string getParameterString(const std::string &name, Int64 index) override;

  void setParameterUInt32(const std::string &name, Int64 index,
                          UInt32 value) override;
  void setParameterInt32(const std::string &name, Int64 index,
                         Int32 value) override;
  void setParameterBool(const std::string &name, Int64 index,
                        bool value) override;
  void setParameterString(const std::string &name, Int64 index,
                          const std::string &s) override;

private:
  //BacktrackingTMRegion() {}
  //BacktrackingTMRegion(const BacktrackingTMRegion &){}

protected:
  struct {
    UInt32 numberOfCols;
    UInt32 cellsPerColumn;
    Real32 initialPerm;
    Real32 connectedPerm;
    UInt32 minThreshold;
    UInt32 newSynapseCount;
    Real32 permanenceInc;
    Real32 permanenceDec;
    Real32 permanenceMax;
    Real32 globalDecay;
    UInt32 activationThreshold;
    bool doPooling;
    UInt32 segUpdateValidDuration;
    UInt32 burnIn;
    bool collectStats;
    Int32 seed;
    UInt32 verbosity;
    bool checkSynapseConsistency;
    UInt32 pamLength;
    UInt32 maxInfBacktrack;
    UInt32 maxLrnBacktrack;
    UInt32 maxAge;
    UInt32 maxSeqLength;
    Int32 maxSegmentsPerCell;
    Int32 maxSynapsesPerSegment;
    char  outputType[25];

    // parameters used by this class and not passed on
    bool learningMode;
    bool inferenceMode;
    bool anomalyMode;
    bool topDownMode;
    bool storeDenseOutput;
    bool computePredictedActiveCellIndices;
    bool orColumnOutputs;

    // some local variables
    UInt32 outputWidth; // columnCount *cellsPerColumn
    bool init;
    Size iter;
    UInt32 sequencePos;
  } args_;


  Byte* prevPredictedState_;
  std::vector<UInt32> prevPredictedColumns_;

  computeCallbackFunc computeCallback_;
  std::unique_ptr<nupic::algorithms::backtracking_tm::BacktrackingTM> tm_;
};

} // namespace nupic

#endif // NTA_BACKTRACKING_TMREGION_HPP
