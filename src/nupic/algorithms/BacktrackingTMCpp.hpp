/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2016, Numenta, Inc.  Unless you have an agreement
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
 * auther: David Keeney, June 2018   -- ported from Python
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for the Temporal Memory in C++
 */

#ifndef NTA_BACKTRACKINGTMCPP_HPP
#define NTA_BACKTRACKINGTMCPP_HPP

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Random.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::Cells4;



namespace nupic {
namespace algorithms {
namespace backtracking_tm {

/////////////////////////////////////////////////////////////////
//  Static function nonzero()
// returns an array of the indexes of the non-zero elements.
// TODO replace with library implementation, or move to utils
template <typename T> static vector<UInt> nonzero(const T *dense_buffer, UInt len) {
  vector<UInt> nz;
  for (UInt idx = 0; idx < len; idx++) {
    if (dense_buffer[idx] != (T)0)
      nz.push_back((UInt)idx);
  }
  return nz;
}

/**
 * Backtracking Temporal Memory implementation in C++.
 * The class:`BacktrackingTMCpp` wraps the C++ algorithms in Cells4.
 *
 * This is a port from the Python module BacktrackingTMCPP.  It was
 * a subclass of BacktrackingTM. This port combined them into one class.
 *
 * Example usage:
 *
 *     SpatialPooler sp(inputDimensions, columnDimensions, <parameters>);
 *     BacktrackingTMCpp tm(columnDimensions, <parameters>);
 *
 *     while (true) {
 *        <get input vector, streaming spatiotemporal information>
 *        sp.compute(inputVector, learn, activeColumns)
 *        tm.compute(number of activeColumns, activeColumns, learn)
 *        <do something with the tm, e.g. classify tm.getActiveCells()>
 *     }
 *
 * The public API uses C arrays, not std::vectors, as inputs. C arrays are
 * a good lowest common denominator. You can get a C array from a vector,
 * but you can't get a vector from a C array without copying it. This is
 * important, for example, when using numpy arrays. The only way to
 * convert a numpy array into a std::vector is to copy it, but you can
 * access a numpy array's internal C array directly.
 *
 * This Class implements the temporal memory algorithm as described in
 * the BAMI <https://numenta.com/biological-and-machine-intelligence/>.
 * The implementation here attempts to closely match the pseudocode in
 * the documentation. This implementation does contain several additional
 * bells and whistles such as a column confidence measure.
 *
 * @:param numberOfCols: (int) Number of mini-columns in the region.
 *                       This values needs to be the same as the number
 *                       of columns in the SP, if one is used.
 *
 * @param cellsPerColumn: (int) The number of cells per mini-column.
 *
 * @param initialPerm: (float) Initial permanence for newly created synapses.
 *
 * @param connectedPerm: TODO: document
 *
 * @param minThreshold: (int)  Minimum number of active synapses for a
 *          segment to be considered during search for the best-matching
 *segments.
 *
 * @param newSynapseCount: (int) The max number of synapses added to a
 *           segment during learning.
 *
 * @param permanenceInc: (float) Active synapses get their permanence counts
 *                    incremented by this value.
 *
 * @param permanenceDec: (float) All other synapses get their permanence
 *                    counts decremented by this value.
 *
 * @param permanenceMax: TODO: document
 *
 * @param maxAge: (int) Number of iterations before global decay takes
 *      effect. Also the global decay execution interval. After global
 *      decay starts, it will will run again every 'maxAge' iterations.
 *      If 'maxAge==1', global decay is applied to every iteration to
 *      every segment.
 *      .. note:: Using 'maxAge > 1' can significantly speed up
 *                the TM when global decay is used.
 *
 * @param globalDecay: (float) Value to decrease permanences when the global
 *      decay process runs. Global decay will remove synapses if
 *      their permanence value reaches 0. It will also remove
 *      segments when they no longer have synapses.
 *      .. note:: Global decay is applied after 'maxAge' iterations,
 *                after which it will run every 'maxAge' iterations.
 *
 * @param activationThreshold: (int) Number of synapses that must be active
 *       to activate a segment.
 *
 * @param doPooling: (bool) If True, pooling is enabled. False is the default.
 *
 * @param segUpdateValidDuration: TODO: document
 *
 * @param burnIn: (int) Used for evaluating the prediction score. Default is 2.
 *
 * @param collectStats: (bool) If True, collect training / inference stats.
 *                     Default is False.
 *
 * @param seed: (int)  Random number generator seed. The seed affects the
 *         random aspects of initialization like the initial permanence values.
 *         A fixed value ensures a reproducible result.
 *
 * @param verbosity: (int) Controls the verbosity of the TM diagnostic
 *         output:
 *                     - verbosity == 0: silent
 *                     - verbosity in [1..6]: increasing levels of verbosity
 *
 * @param pamLength: (int) Number of time steps to remain in "Pay Attention
 *          Mode" after we detect we've reached the end of a learned sequence.
 *          Setting this to 0 disables PAM mode. When we are in PAM mode,
 *          we do not burst unpredicted columns during learning, which in
 *          turn prevents us from falling into a previously learned sequence
 *          for a while (until we run through another 'pamLength' steps).
 *
 *          The advantage of PAM mode is that it requires fewer
 *          presentations to learn a set of sequences which share elements. The
 *          disadvantage of PAM mode is that if a learned sequence is
 *immediately followed by set set of elements that should be learned as a 2nd
 *          sequence, the first 'pamLength' elements of that sequence will not
 *be learned as part of that 2nd sequence.
 *
 * @param maxInfBacktrack: (int) How many previous inputs to keep in a
 *          buffer for inference backtracking.
 *
 * @param maxLrnBacktrack: (int) How many previous inputs to keep in a
 *          buffer for learning backtracking.
 *
 * @param maxSeqLength: (int) If not 0, we will never learn more than
 *          'maxSeqLength' inputs in a row without starting over at
 *          start cells. This sets an upper bound on the length of
 *          learned sequences and thus is another means (besides
 *          'maxAge' and 'globalDecay') by which to limit how much the
 *          TM tries to learn.
 *
 * @param maxSegmentsPerCell: (int) The maximum number of segments allowed
 *          on a cell. This is used to turn on "fixed size CLA" mode. When
 *          in effect, 'globalDecay' is not applicable and must be set to 0 and
 *          'maxAge' must be set to 0. When this is used (> 0),
 *'maxSynapsesPerSegment' must also be > 0.
 *
 * @param maxSynapsesPerSegment: (int) The maximum number of synapses
 *          allowed in a segment. This is used to turn on "fixed size CLA" mode.
 *          When in effect, 'globalDecay' is not applicable and must be set
 *          to 0, and 'maxAge' must be set to 0. When this is used (> 0),
 *          'maxSegmentsPerCell' must also be > 0.
 *
 * @param outputType: (string) Can be one of the following (default 'normal')
 *            - 'normal': output the OR of the active and predicted state.
 *            - 'activeState': output only the active state.
 *            - 'activeState1CellPerCol': output only the active
 *               state, and at most 1 cell/column. If more than 1 cell
 *               is active in a column, the one with the highest confidence
 *               is sent up.
 *
 *************/

class BacktrackingTMCpp {
public:
  BacktrackingTMCpp(); // when restoring from serialization

  BacktrackingTMCpp(UInt32 numberOfCols,
                    UInt32 cellsPerColumn, // first two fields are required
                    Real32 initialPerm = 0.11f, Real32 connectedPerm = 0.50f,
                    UInt32 minThreshold = 8, UInt32 newSynapseCount = 15,
                    Real32 permanenceInc = 0.10f, Real32 permanenceDec = 0.10f,
                    Real32 permanenceMax = 1.0f, Real32 globalDecay = 0.10f,
                    UInt32 activationThreshold = 12, bool doPooling = false,
                    UInt32 segUpdateValidDuration = 5, UInt32 burnIn = 2,
                    bool collectStats = false, Int32 seed = 42,
                    Int32 verbosity = 0, bool checkSynapseConsistency = false,
                    UInt32 pamLength = 1, UInt32 maxInfBacktrack = 10,
                    UInt32 maxLrnBacktrack = 5, UInt32 maxAge = 100000,
                    UInt32 maxSeqLength = 32, Int32 maxSegmentsPerCell = -1,
                    Int32 maxSynapsesPerSegment = -1,
                    const std::string outputType = "normal");

  virtual ~BacktrackingTMCpp();

  //----------------------------------------------------------------------
  //  Main functions
  //----------------------------------------------------------------------

  /**
   * Get the version number of for the TM implementation.
   *
   * @returns Integer version number.
   */
  virtual UInt version() const;

  /**
   * Perform one time step of the Temporal Memory algorithm.
   * This method calls Cells4 to do most of the work.
   *
   * @param bottomUpInput
   * Input array. It only looks to see if values are zero or non-zero
   *              so the type can be UInt32, Int32, or Real32.
   *
   * @param enableLearn
   * Whether or not learning is enabled.
   *
   * @param enableInference
   * Whether or not inference is enabled.
   *
   * @return
   * Pointer to the output buffer.  (The buffer is also in currentOutput_ )
   */
  virtual Real *compute(Real *bottomUpInput, bool enableLearn = true,
                        bool enableInference = false);

  //  Complete learning. Keep weakly formed synapses around because they contain
  //  confidence scores for paths out of learned sequenced and produce a better
  //  prediction than chance.
  virtual void finishLearning() { trimSegments(0.0001f, 0); }
  virtual std::pair<UInt, UInt> trimSegments(Real minPermanence = 0.0,
                                             UInt32 minNumSyns = 0);
  virtual void infer(Real32 *bottomUpInput) {
    compute(bottomUpInput, false, true);
  }
  virtual void learn(Real32 *bottomUpInput, bool enableInference = false) {
    compute(bottomUpInput, true, enableInference);
  }
  virtual std::shared_ptr<Real> predict(Size nSteps);
  virtual void reset();
  virtual Real32 *topDownCompute();
  inline Size getOutputBufferSize() { return nCells; }


  struct score_tuple {
    Real predictionScore;
    Real posPredictionScore;
    Real negPredictionScore;
  };
  struct predictionResults_t {
    Size totalExtras;
    Size totalMissing;
    std::vector<struct score_tuple> conf;
    std::shared_ptr<Real> missing;
  };
  std::shared_ptr<struct BacktrackingTMCpp::predictionResults_t>
    _checkPrediction(std::vector<std::vector<UInt>> patternNZs,
                     const Byte *predicted = nullptr,
                     const Real *colConfidence = nullptr,
                     bool details = false);


  ////////////////// getters/setters ////////////////////////
  inline Size   getNumCells() const { return nCells; }
  inline Size   getnumCol() const  { return loc_.numberOfCols; }
  inline Size   getcellsPerCol() const  { return loc_.cellsPerColumn; }
  inline Real32 getInitialPerm() const  { return cells4_->getPermInitial(); }
  inline Real32 getConnectedPerm() const  { return cells4_->getPermConnected(); }
  inline UInt32 getMinThreshold() const  { return cells4_->getMinThreshold(); }
  inline UInt32 getNewSynapseCount() const  { return cells4_->getNewSynapseCount(); }
  inline Real32 getPermanenceInc() const  { return cells4_->getPermInc(); }
  inline Real32 getPermanenceDec() const  { return cells4_->getPermDec(); }
  inline Real32 getPermanenceMax() const  { return cells4_->getPermMax(); }
  inline Real32 getGlobalDecay() const  { return cells4_->getGlobalDecay(); }
  inline UInt32 getActivationThreshold() const  { return cells4_->getActivationThreshold(); }
  inline bool   getDoPooling() const  { return cells4_->getDoPooling(); }
  inline UInt32 getSegUpdateValidDuration() const { return cells4_->getSegUpdateValidDuration(); }
  inline UInt32 getVerbosity() const  { return cells4_->getVerbosity(); }
  inline bool   getCheckSynapseConsistency() const { return cells4_->getCheckSynapseConsistency(); }
  inline UInt32 getPamLength() const  { return cells4_->getPamLength(); }
  inline UInt32 getMaxInfBacktrack() const  { return cells4_->getMaxInfBacktrack(); }
  inline UInt32 getMaxLrnBacktrack() const  { return cells4_->getMaxLrnBacktrack(); }
  inline UInt32 getMaxAge() const  { return cells4_->getMaxAge(); }
  inline UInt32 getMaxSeqLength() const  { return cells4_->getMaxSeqLength(); }
  inline Int32  getMaxSegmentsPerCell() const  { return cells4_->getMaxSegmentsPerCell(); }
  inline Int32  getMaxSynapsesPerSegment() const  { return cells4_->getMaxSynapsesPerSegment(); }
  inline std::string getOutputType() const  { return outputType_; }
  inline Int32  getBurnIn() const  { return loc_.burnIn; }
  inline bool   getCollectStats() const  { return loc_.collectStats; }
  inline bool   getSeed() const  { return loc_.seed; }

  inline void setVerbosity(UInt val) { cells4_->setVerbosity(val); }
  inline void setCheckSynapseConsistency(bool val) { cells4_->setCheckSynapseConsistency(val); }
  inline void setPamLength(UInt32 val) { cells4_->setPamLength(val); }
  inline void setMaxInfBacktrack(UInt32 val) { cells4_->setMaxInfBacktrack(val); }
  inline void setMaxLrnBacktrack(UInt32 val) { cells4_->setMaxLrnBacktrack(val); }
  inline void setMaxAge(UInt32 val) { cells4_->setMaxAge(val); }
  inline void setMaxSeqLength(UInt32 val) { cells4_->setMaxSeqLength(val); }
  inline void setMaxSegmentsPerCell(Int32 val) { cells4_->setMaxSegmentsPerCell(val); }
  inline void setMaxSynapsesPerSegment(Int32 val) { cells4_->setMaxSynapsesPerSegment(val); }
  inline void setCollectStats(bool val) { loc_.collectStats = val; }
  inline void setBurnIn(UInt32 val) { loc_.burnIn = val; }

  //inline void setRandomState(Int32 random) { loc_.random = random; }
  // inline Int32 getRandomState() { return loc_.random; }

  inline Byte *getActiveState() const { return cells4_->getInfActiveStateT(); }
  inline Byte *getPredictedState() const { return cells4_->getInfPredictedStateT(); }
  inline Byte *getLearnActiveStateT() const {  return cells4_->getLearnActiveStateT(); }
  inline Byte *getLearnPredictedStateT() const {  return cells4_->getLearnPredictedStateT(); }
  inline Real32 getAvgLearnedSeqLength() const { return cells4_->getAvgLearnedSeqLength(); }
  inline UInt getNumSegments() const { return cells4_->nSegments(); }
  inline UInt getNumSegmentsInCell(Size c, Size i) const { return cells4_->nSegmentsOnCell((UInt)c, (UInt)i); }
  inline Size getNumSynapses() const { return cells4_->nSynapses(); }
  inline Real32 getNumSynapsesPerSegmentAvg() const {
    return ((Real32)getNumSynapses() / std::max<Size>(1, getNumSegments()));
  }

  struct SegOnCellInfo_t {
      Size c;  // segment's cell
      Size i;
      Size segIdx;
      bool isSequenceSegment;
      Size positiveActivations;
      Size totalActivations;
      Size lastActiveIteration;
      Real lastPosDutyCycle;
      Size lastPosDutyCycleIteration;
      std::vector<std::tuple<Size,Size,Real>> synapses;
  };
  virtual struct SegOnCellInfo_t getSegmentOnCell(Size c, Size i, Size segIdx) const;

  struct ageinfo_t {
    std::string range;
    Size cnt;
  };
  struct seginfo_t {
    Size nSegments;
    Size nSynapses;
    Size nActiveSegs;
    Size nActiveSynapses;
    std::map<Size, Size> distSegSizes;
    std::map<Size, Size> distNSegsPerCell;
    std::map<Size, Size> distPermValues;
    std::vector<struct ageinfo_t> distAges;
  };
  virtual struct seginfo_t getSegmentInfo(bool collectActiveData = false) const;

  // note: the output buffer size is [param_.numberOfCols X param_.cellsPerColumn]
  virtual Real32 *getOutputBuffer() { return currentOutput_; }
  virtual void setOutputBuffer(Real32 *buf);

  // Allow interfaces a way to take over the buffers
  void setStatePointers(Byte *infActiveT, Byte *infActiveT1, Byte *infPredT,
                        Byte *infPredT1, Real *colConfidenceT,
                        Real *colConfidenceT1, Real *cellConfidenceT,
                        Real *cellConfidenceT1);
  void getStatePointers(Byte *&activeT, Byte *&activeT1, Byte *&predT,
                        Byte *&predT1, Real *&colConfidenceT,
                        Real *&colConfidenceT1, Real *&confidenceT,
                        Real *&confidenceT1) const;



  ///////// printing for Debug /////////////
  virtual void printActiveIndicesByte(const Byte *state, bool andValues = false, std::ostream& out = std::cout)const;
  virtual void printActiveIndicesReal(const Real *confidence, bool andValues = false,std::ostream& out = std::cout) const;
  virtual void printColActiveIndices(const Real *colconfidence,bool andValues = false,std::ostream& out = std::cout) const;
  virtual void printCell(Size c, Size i, bool onlyActiveSegments = false, std::ostream& out = std::cout) const;
  virtual void printCells(bool predictedOnly = false, std::ostream& out = std::cout) const;
  virtual void printColConfidence(const Real *aState, Size maxCols = 20, std::ostream& out = std::cout) const;
  virtual void printComputeEnd(Real *output, bool learn = false, std::ostream& out = std::cout) const;
  virtual void printConfidence(const Real *aState, Size maxCols = 20, std::ostream& out = std::cout) const;
  virtual void printInput(const Real32 *x, std::ostream& out = std::cout) const;
  virtual void printOutput(const Real32 *y, std::ostream& out = std::cout) const;
  virtual void printParameters(std::ostream& out = std::cout) const;
  virtual void printSegment(Segment &s, std::ostream& out = std::cout) const;
  virtual void printState(const Byte *aState, std::ostream& out = std::cout) const;
  virtual void printStates(bool printPrevious = true, bool printLearnState = true, std::ostream& out = std::cout) const;

  //////////// compairing two TM's  ///////////////
/**
 * Given two TM instances, list the difference between them and returns false
 * if there is a difference. This function checks the major parameters. If this
 * passes (and checkLearn is true) it checks the number of segments on each
 * cell. If this passes, checks each synapse on each segment. When comparing C++
 * and Py, the segments are usually in different orders in the cells. tmDiff
 * ignores segment order when comparing TM's.
 *
 * Note that if the TM's states may be different due to the random number generator.
 *
 * @tm1 and tm2 - references to two BacktrackingTM's to be compared.
 * @out where to stream the output
 * @verbosity  - How much to say.
 * @relaxSegmentTests - how strict to be.
 * @checkLearn - if true, will check learn states as well as all the segments
 * @checkStates - If true, will check the various state arrays
 */
  static bool tmDiff2(const BacktrackingTMCpp &tm1, const BacktrackingTMCpp &tm2,
               std::ostream &out = std::cout, Int32 verbosity = 0,
               bool relaxSegmentTests = true, bool checkLearn = true,
               bool checkStates = true);
  static bool sameTMParams(const BacktrackingTMCpp &tm1, const BacktrackingTMCpp &tm2,
                    std::ostream &out = std::cout, Int32 verbosity = 0);
  static bool sameSegment(const struct SegOnCellInfo_t &segVect1, const struct SegOnCellInfo_t &segVect2,
                   std::ostream &out = std::cout, Int32 verbosity = 0);

  // an alternative way to compare two tm's -- by comparing the serialized strings.
  bool diff(const BacktrackingTMCpp& tm1, const BacktrackingTMCpp &tm2) const;

  bool operator==(const BacktrackingTMCpp &tm2) const { return BacktrackingTMCpp::tmDiff2(*this, tm2); } //could also use diff()
  inline bool operator!=(const BacktrackingTMCpp &tm2) const { return !(*this==tm2); }

  /////// statistics ////////
  // Note: These are empty if collectSequenceStats is false.
  virtual std::map<std::string, Real32> getStats(); // backtracking_tm.py line 861
  virtual void resetStats();

  std::shared_ptr<Real>  getPrevSequenceSignature(); // preivous value of confidence Histogram
  std::shared_ptr<Real> getConfHistogram(); // current value of confidence Histogram
  void setRetrieveLearningStates(bool val) { loc_.retrieveLearningStates = val; }

  /**
   * Serialization. save and load the current state of the TM to/from the
   * specified file.
   */
  void saveToFile(std::string filePath);
  void loadFromFile(std::string filePath);
  void save(std::ostream &out) const;
  void load(std::istream &in);


  //----------------------------------------------------------------------
  //  Internal functions and data
  //----------------------------------------------------------------------

protected:
  //////// internal functions ///////////

  void _updateStatsInferEnd(std::map<std::string, Real> internalStats,
                            const UInt32 *bottomUpNZ,
                            const Byte *predictedState,
                            const Real *colConfidence);
  Real *_computeOutput();
  void _inferPhase2();
  inline Size _getCellCol(Size idx) const { return idx / loc_.cellsPerColumn; }
  inline Size _getCellIdx(Size idx) const { return idx % loc_.cellsPerColumn; }
  /*
   * A segment is active if it has >= activationThreshold connected
   *  synapses that are active due to infActiveState. timestep is "t" or "t-1".
   */
  virtual bool _slowIsSegmentActive(Segment &seg, std::string timestep) const;

  // Used by predict() to save/restore current state
  struct ss_t {
      std::shared_ptr<Byte> Byteptr;
      std::shared_ptr<Real> Realptr;
  };
  typedef std::map<std::string, struct ss_t> tmSavedState_t;
  virtual void _getTPDynamicState(tmSavedState_t &ss);
  virtual void _setTPDynamicState(tmSavedState_t &ss);

  void deepcopySave_(tmSavedState_t &ss, std::string name, Byte *buf, Size count);
  void deepcopySave_(tmSavedState_t &ss, std::string name, Real *buf, Size count);

  void deepcopyRestore_(tmSavedState_t &ss, std::string name, Byte *buf, Size count);
  void deepcopyRestore_(tmSavedState_t &ss, std::string name, Real *buf, Size count);

  template <typename T>
  void fastbuffercopy(T *tobuf, T *frombuf, Size size);

  ////// local parameters  ////////////
  // local variables
  struct {
    UInt32 numberOfCols;
    UInt32 cellsPerColumn;

    UInt32 burnIn;
    bool collectStats;
    Int32 seed;
    Size lrnIterationIdx;
    Size iterationIdx;

    // unique segment id, so we can put segments in hashes
    UInt32 segID;

    // pamCounter gets reset to pamLength whenever we detect that the
    // learning state is making good predictions (at least half the
    // columns predicted). Whenever we do not make a good prediction, we
    // decrement pamCounter. When pamCounter reaches 0, we start the learn
    // state over again at start cells.
    UInt32 pamCounter;
    bool collectSequenceStats;
    bool resetCalled;
    Real32 avgInputDensity;
    UInt32 learnedSeqLength;
    Real avgLearnedSeqLength;

   // UInt32 numberOfCells; // numberOfCols * cellsPerColumn

    // If True, let C++ allocate memory for activeState, predictedState, and
    // learnState.  In this case we can retrieve copies of these states but
    // can't set them directly . If False, the interface code can allocate them
    // as external arrays and we can pass pointers to Cells4 using
    // setStatePointers
    bool allocateStatesInCPP;

    // If true, always fetch the learn state pointers after every compute().
    bool retrieveLearningStates;

  } loc_;
  std::string outputType_;

  Size nCells;  // number of cells (numberOfCols * cellsPerColumn)

  // lists indexed by cell index
  Real* currentOutput_; // contains nCells items
  bool currentOutputOwn_; // if true, this class owns the buffer.

  std::map<std::string, Real> internalStats_;

  std::shared_ptr<Real> prevSequenceSignature_; // preivous value of confHistogram
  std::shared_ptr<Real> confHistogram_; // for stats if collectSequenceStats is true


  Cells4::Cells4 *cells4_;
};



} // end namespace backtracking_tm
} // end namespace algorithms
} // end namespace nupic



#endif // NTA_BACKTRACKINGTMCPP_HPP

//################################################################################
//# The following methods are implemented in the base class backtrackingTM
//(Python) # but should never be called in this implementation.
//################################################################################
//   _isSegmentActive(self, seg, timeStep)
//   _getSegmentActivityLevel(self, seg, timeStep, connectedSynapsesOnly =
//   False, learnState = False) isSequenceSegment(self, s)
//   _getBestMatchingSegment(self, c, i, timeStep, learnState = False)
//   _getSegmentActiveSynapses(self, c, i, s, timeStep, newSynapses = False)
//   updateSynapse(self, segment, synapse, delta)
//   _adaptSegment(self, update, positiveReinforcement)
//   debugPrint();

// The following are parts of backtrackingTM.py that is the Python only
// implementation
// class segUpdate
// def _addToSegmentUpdates(self, c, i, segUpdate)   backtracking_tm.py line
// 1434 def _removeSegmentUpdate(self, updateInfo)                    line 1458
// def _inferBacktrack(self, activeColumns)                      line 1666
// def _inferPhase1(self, activeColumns, useStartCells)          line 1921
// def _updateInferenceState(self, activeColumns)                line 2041;
// def _learnBacktrackFrom(self, startOffset, readOnly=True)     line 2086
// def _learnBacktrack(self)                                     line 2184;
// def _learnPhase1(self, activeColumns, readOnly=False)         line 2295
// def _learnPhase2(self, readOnly=False)                        line 2387;
// def _updateLearningState(self, activeColumns)                 line 2436
// def _trimSegmentsInCell(self, colIdx, cellIdx, segList, minPermanence,
// minNumSyns) line 2713 def _cleanUpdatesList(self, col, cellIdx, seg)
// line 2804 def _getBestMatchingCell(self, c, activeState, minThreshold)  line
// 3003 def _getCellForNewSegment(self, colIdx)                       line 3073
// def _getSegmentActiveSynapses(self, c, i, s, activeState, newSynapses=False)
// line 3148 def _chooseCellsToLearnFrom(self, c, i, s, n, activeState)    line
// 3193 def _processSegmentUpdates(self, activeColumns)               line 3238
// def _adaptSegment(self, segUpdate)                            line 3310
// def dutyCycle(self, active=False, readOnly=False)             line 3593
