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

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::Cells4;

namespace nupic {
namespace algorithms {
namespace backtracking_tm {


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
 *          segment to be considered during search for the best-matching segments.
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
 *          disadvantage of PAM mode is that if a learned sequence is immediately 
 *          followed by set set of elements that should be learned as a 2nd 
 *          sequence, the first 'pamLength' elements of that sequence will not be
 *          learned as part of that 2nd sequence.
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
 *          'maxAge' must be set to 0. When this is used (> 0), 'maxSynapsesPerSegment'
 *          must also be > 0.
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
  BacktrackingTMCpp();  // when restoring from serialization

  BacktrackingTMCpp(UInt32 numberOfCols, UInt32 cellsPerColumn,  // first two fields are required
                    Real32 initialPerm = 0.11f, Real32 connectedPerm = 0.50f,
                    UInt32 minThreshold = 8, UInt32 newSynapseCount = 15,
                    Real32 permanenceInc = 0.10f, Real32 permanenceDec = 0.10f,
                    Real32 permanenceMax = 1.0f, Real32 globalDecay = 0.10f,
                    UInt32 activationThreshold = 12, bool doPooling = false,
                    UInt32 segUpdateValidDuration = 5, UInt32 burnIn = 2,
                    bool collectStats = false, Int32 seed = 42,
                    Int32 verbosity = 1, bool checkSynapseConsistency = false,
                    UInt32 pamLength = 1, UInt32 maxInfBacktrack = 10,
                    UInt32 maxLrnBacktrack = 5, UInt32 maxAge = 100000,
                    UInt32 maxSeqLength = 32, Int32 maxSegmentsPerCell = -1,
                    Int32 maxSynapsesPerSegment = -1,
                    const char *outputType = "normal");

  virtual ~BacktrackingTMCpp() {
    delete cells4_;
  }

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
  virtual Real*compute(Real *bottomUpInput, bool enableLearn = true, bool enableInference = false);

  //  Complete learning. Keep weakly formed synapses around because they contain
  //  confidence scores for paths out of learned sequenced and produce a better
  //  prediction than chance.
  virtual void finishLearning() { trimSegments(0.0001f, 0); }
  virtual std::pair<UInt, UInt> trimSegments(Real minPermanence = 0.0,
                                             UInt32 minNumSyns = 0);
  virtual void infer(Real32 *bottomUpInput) {compute(bottomUpInput, false, true); }
  virtual void learn(Real32 *bottomUpInput, bool enableInference) {
    compute(bottomUpInput, true, enableInference);
  }
  virtual std::shared_ptr<Real> predict(Size nSteps);
  virtual void reset();
  virtual Real32 *topDownCompute();
  inline Size getOutputBufferSize() { return nCells; }




  ////////////////// getters/setters ////////////////////////
  virtual void setParameter(const std::string name, Int32 val);
  virtual void setParameter(const std::string name, UInt32 val);
  virtual void setParameter(const std::string name, Real32 val);
  virtual void setParameter(const std::string name, bool val);
  virtual void setParameter(const std::string name, std::string val);
  virtual Int32 getParameterInt32(const std::string name);
  virtual UInt32 getParameterUInt32(const std::string name);
  virtual Real32 getParameterReal32(const std::string name);
  virtual bool getParameterBool(const std::string name);
  virtual std::string getParameterString(const std::string name);
  //inline void setRandomState(Int32 random) { loc_.random = random; }
  //inline Int32 getRandomState() { return loc_.random; }
  inline Size getNumCells() { return nCells; }
  inline Size getnumCol() { return param_.numberOfCols; }
  inline  Size getcellsPerCol() { return param_.cellsPerColumn; }
  inline  Byte *getActiveState() { return infActiveState_["t"]; }
  inline  Byte *getLearnActiveStateT() { return lrnActiveState_["t"]; }
  inline  Real32 getAvgLearnedSeqLength() { return cells4_->getAvgLearnedSeqLength(); }
  inline  Byte *getPredictedState() { return infPredictedState_["t"]; }
  inline  UInt getNumSegments() { return cells4_->nSegments(); }
  inline  UInt getNumSegmentsInCell(Size c, Size i) {  return cells4_->nSegmentsOnCell((UInt)c, (UInt)i); }
  inline  Size getNumSynapses() { return cells4_->nSynapses(); }
  inline  Real32 getNumSynapsesPerSegmentAvg() { return ((Real32)getNumSynapses() / std::max<Size>(1, getNumSegments())); }
  union segoncellinfo_t {
    struct {
      Size segIdx;
      bool isSequenceSegment;
      Size positiveActivations;
      Size totalActivations;
      Size lastActiveIteration;
      Real lastPosDutyCycle;
      Size lastPosDutyCycleIteration;
    } se; // segment info, first item
    struct {
      Size c;
      Size i;
      Real permanence;
    } sy; // synapse info, the rest of the items
  };
  virtual vector<union segoncellinfo_t> getSegmentOnCell(Size c, Size i,
                                                         Size segIdx);

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
  virtual struct seginfo_t getSegmentInfo(bool collectActiveData = false);


  // note: the output buffer size is [param_.numberOfCols X param_.cellsPerColumn]
  virtual Real32 *getOutputBuffer() { return currentOutput_.get(); }
  virtual void setOutputBuffer(Real32 *buf) { currentOutput_.reset(buf); };

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
  virtual void printActiveIndicesByte(const Byte *state, bool andValues = false);
  virtual void printActiveIndicesReal(const Real *confidence, bool andValues = false);
  virtual void printColActiveIndices(const Real *colconfidence,  bool andValues = false);
  virtual void printCell(Size c, Size i, bool onlyActiveSegments = false);
  virtual void printCells(bool predictedOnly = false);
  virtual void printColConfidence(const Real *aState, Size maxCols = 20);
  virtual void printComputeEnd(Real *output, bool learn = false);
  virtual void printConfidence(const Real *aState, Size maxCols = 20);
  virtual void printInput(const Real32 *x);
  virtual void printOutput(const Real32 *y);
  virtual void printParameters();
  virtual void printSegment(Segment &s);
  virtual void printState(const Byte *aState);
  virtual void printStates(bool printPrevious = true, bool printLearnState = true);

  /////// statistics ////////
  // Note: These are empty if collectSequenceStats is false.
  virtual std::map<std::string, Real32> getStats(); // backtracking_tm.py line 861
  virtual void resetStats();

  std::shared_ptr<Real> getPrevSequenceSignature(); // preivous value of confidence Histogram
  std::shared_ptr<Real> getConfHistogram();         // current value of confidence Histogram


  
  /**
   * Serialization. save and load the current state of the TM to/from the
   * specified file.
   */
  void saveToFile(std::string filePath);
  void loadFromFile(std::string filePath);
  void save(std::ofstream& out);
  void load(std::ifstream& in);


  //----------------------------------------------------------------------
  //  Internal functions and data
  //----------------------------------------------------------------------

protected:
  //////// internal functions ///////////

  void _initEphemerals();
  void _updateStatsInferEnd(std::map<std::string, Real> internalStats,
                            const UInt32 *bottomUpNZ,
                            const Byte *predictedState,
                            const Real *colConfidence);
  Real *_computeOutput();
  struct score_tuple {
    Real predictionScore;
    Real posPredictionScore;
    Real negPredictionScore;
  };
  void _checkPrediction(std::vector<const UInt32 *> patternNZs,
                        const Byte *predicted, const Real *colConfidence,
                        bool details, Size &totalExtras, Size &totalMissing,
                        std::vector<struct score_tuple> &conf, Real *missing);
  void _inferPhase2();
  inline Size _getCellCol(Size idx) { return idx / param_.cellsPerColumn; }
  inline Size _getCellIdx(Size idx) { return idx % param_.cellsPerColumn; }
  virtual bool _slowIsSegmentActive(Segment &seg, const char *timestep);


  // Used by predict() to save/restore current state
  typedef struct tmSavedState {
    // Saved state buffers (set aside and restored during predict().
    std::map<std::string, std::shared_ptr<Byte>> infActiveState_;
    std::map<std::string, std::shared_ptr<Byte>> infPredictedState_;
    std::map<std::string, std::shared_ptr<Real>> cellConfidence_;
    std::map<std::string, std::shared_ptr<Real>> colConfidence_;
    std::map<std::string, std::shared_ptr<Byte>> lrnActiveState_;
    std::map<std::string, std::shared_ptr<Byte>> lrnPredictedState_;
  } tmSavedState_t;

  virtual void _getTPDynamicState(tmSavedState_t *ss);
  virtual void _setTPDynamicState(tmSavedState_t *ss);
  template <typename T>
  void deepcopySave_(std::map<std::string, T *> &fromstate,
                     std::map<std::string, std::shared_ptr<T>> &tostate,
                     Size size);
  template <typename T>
  void deepcopyRestore_(std::map<std::string, T *> &tostate,
                        std::map<std::string, std::shared_ptr<T>> &fromstate,
                        Size size);
  template <typename T>
  void fastbuffercopy(T* tobuf, T* frombuf, Size size);


  ////// passed in parameters  ////////////
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
    Int32 verbosity;
    bool checkSynapseConsistency;
    UInt32 pamLength;
    UInt32 maxInfBacktrack;
    UInt32 maxLrnBacktrack;
    UInt32 maxAge;
    UInt32 maxSeqLength;
    Int32 maxSegmentsPerCell;
    Int32 maxSynapsesPerSegment;
    char outputType[25];
  } param_;

  // local state variables
  struct {
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

    // Ephemerals
    UInt32 numberOfCells; // numberOfCols * cellsPerColumn
    bool makeCells4Ephemeral;

    // If True, let C++ allocate memory for activeState, predictedState, and
    // learnState.  In this case we can retrieve copies of these states but
    // can't set them directly . If False, the interface code can allocate them
    // as external arrays and we can pass pointers to Cells4 using
    // setStatePointers
    bool allocateStatesInCPP;

    // If true, always fetch the learn state pointers after every compute().
    bool retrieveLearningStates;

  } loc_;

  Size nCells;
  //std::vector<Size> activeColumns_; // list of indices of active columns
  //std::vector<vector<Size>> prevLrnPatterns_; // list of activeColumns lists.
  //std::vector<vector<Size>> prevInfPatterns_; // list of activeColumns lists.
  //std::map<Size, vector<Size>> segmentUpdates_; // list of activeColumns lists indexed by cell index
  std::shared_ptr<Real> currentOutput_; // contains nCells items

  //std::map<std::string, Real> stats_;
  std::map<std::string, Real> internalStats_;
  std::shared_ptr<Real> prevSequenceSignature_;  // preivous value of confHistogram
  std::shared_ptr<Real> confHistogram_; // for stats if collectSequenceStats is true

  // State buffers from Cells4
  std::map<std::string, Byte *> infActiveState_; // numberOfCols * cellsPerColumn items
  std::map<std::string, Byte *> infPredictedState_; // numberOfCols * cellsPerColumn items
  std::map<std::string, Real *> cellConfidence_; // numberOfCols * cellsPerColumn items
  std::map<std::string, Real *> colConfidence_; // numberOfCols items
  std::map<std::string, Byte *> lrnActiveState_; // numberOfCols * cellsPerColumn items
  std::map<std::string, Byte *> lrnPredictedState_; // numberOfCols * cellsPerColumn items


  // Cells are indexed by column and index in the column
  // Every cells[column][index] contains a list of segments
  // Each segment is a structure of class Segment
  // std::shared_ptr<vector<Segment>> cells_;

  Cells4::Cells4 *cells4_;

};

} // end namespace backtracking_tm
} // end namespace algorithms
} // end namespace nupic

#endif // NTA_BACKTRACKINGTMCPP_HPP




//################################################################################
//# The following methods are implemented in the base class backtrackingTM (Python)
//# but should never be called in this implementation.
//################################################################################
//   _isSegmentActive(self, seg, timeStep)
//   _getSegmentActivityLevel(self, seg, timeStep, connectedSynapsesOnly = False, learnState = False)
//   isSequenceSegment(self, s)
//   _getBestMatchingSegment(self, c, i, timeStep, learnState = False)
//   _getSegmentActiveSynapses(self, c, i, s, timeStep, newSynapses = False)
//   updateSynapse(self, segment, synapse, delta)
//   _adaptSegment(self, update, positiveReinforcement)
//   debugPrint(); 


// The following are parts of backtrackingTM.py that is the Python only implementation 
  //class segUpdate
  // def _addToSegmentUpdates(self, c, i, segUpdate)   backtracking_tm.py line 1434
  // def _removeSegmentUpdate(self, updateInfo)                    line 1458
  // def _inferBacktrack(self, activeColumns)                      line 1666
  // def _inferPhase1(self, activeColumns, useStartCells)          line 1921
  // def _updateInferenceState(self, activeColumns)                line 2041;
  // def _learnBacktrackFrom(self, startOffset, readOnly=True)     line 2086
  // def _learnBacktrack(self)                                     line 2184;
  // def _learnPhase1(self, activeColumns, readOnly=False)         line 2295
  // def _learnPhase2(self, readOnly=False)                        line 2387;
  // def _updateLearningState(self, activeColumns)                 line 2436
  // def _trimSegmentsInCell(self, colIdx, cellIdx, segList, minPermanence, minNumSyns) line 2713
  // def _cleanUpdatesList(self, col, cellIdx, seg)                line 2804
  // def _getBestMatchingCell(self, c, activeState, minThreshold)  line 3003
  // def _getCellForNewSegment(self, colIdx)                       line 3073
  // def _getSegmentActiveSynapses(self, c, i, s, activeState, newSynapses=False) line 3148
  // def _chooseCellsToLearnFrom(self, c, i, s, n, activeState)    line 3193
  // def _processSegmentUpdates(self, activeColumns)               line 3238
  // def _adaptSegment(self, segUpdate)                            line 3310
  // def dutyCycle(self, active=False, readOnly=False)             line 3593
