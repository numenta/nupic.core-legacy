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
 * author: David Keeney, June 2018  -- ported from Python
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of TemporalMemory
 *
 * The functions in this file use the following argument ordering
 * convention:
 *
 * 1. Output / mutated params
 * 2. Traditional parameters to the function, i.e. the ones that would still
 *    exist if this function were a method on a class
 * 3. Model state (marked const)
 * 4. Model parameters (including "learn")
 */

#include <algorithm>
#include <climits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <vector>

#include <nupic/algorithms/BacktrackingTMCpp.hpp>
#include <nupic/algorithms/Segment.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::backtracking_tm;
using namespace nupic::algorithms::Cells4;

static const UInt TM_VERSION = 3; // - 7/14/2018 keeney

BacktrackingTMCpp::BacktrackingTMCpp()
{
    currentOutput_ = nullptr;
    currentOutputOwn_ = true;
}

BacktrackingTMCpp::BacktrackingTMCpp(
    UInt32 numberOfCols, UInt32 cellsPerColumn, Real32 initialPerm,
    Real32 connectedPerm, UInt32 minThreshold, UInt32 newSynapseCount,
    Real32 permanenceInc, Real32 permanenceDec, Real32 permanenceMax,
    Real32 globalDecay, UInt32 activationThreshold, bool doPooling,
    UInt32 segUpdateValidDuration, UInt32 burnIn, bool collectStats, Int32 seed,
    Int32 verbosity, bool checkSynapseConsistency, UInt32 pamLength,
    UInt32 maxInfBacktrack, UInt32 maxLrnBacktrack, UInt32 maxAge,
    UInt32 maxSeqLength, Int32 maxSegmentsPerCell, Int32 maxSynapsesPerSegment,
    std::string outputType)
{
  cells4_ = nullptr;

  // Check arguments
  NTA_ASSERT(pamLength > 0) << "This implementation must have pamLength > 0";
  if (maxSegmentsPerCell != -1 or maxSynapsesPerSegment != -1) {
    NTA_ASSERT(maxSegmentsPerCell > 0 and maxSynapsesPerSegment > 0);
    NTA_ASSERT(globalDecay == 0.0f);
    NTA_ASSERT(maxAge == 0);
    NTA_ASSERT((UInt32)maxSynapsesPerSegment >= newSynapseCount)
        << "TM requires that maxSynapsesPerSegment >= newSynapseCount. "
        << "(Currently " << maxSynapsesPerSegment << ">=" << newSynapseCount
        << ")";
  }

  NTA_ASSERT(outputType == "normal" ||
             outputType == "activeState" ||
             outputType == "activeState1CellPerCol");

  segUpdateValidDuration = (doPooling) ? segUpdateValidDuration : 1;

  // Store creation parameters
  loc_.numberOfCols = numberOfCols;
  loc_.cellsPerColumn = cellsPerColumn;

  loc_.burnIn = burnIn;
  loc_.collectStats = collectStats;
  loc_.seed = seed;

  outputType_ = outputType;

  // Initialize local state data
  loc_.lrnIterationIdx = 0;
  loc_.iterationIdx = 0;
  // unique segment id, so we can put segments in hashes
  loc_.segID = 0;

  // pamCounter gets reset to pamLength whenever we detect that the learning
  // state is making good predictions (at least half the columns predicted).
  // Whenever we do not make a good prediction, we decrement pamCounter.
  // When pamCounter reaches 0, we start the learn state over again at start
  // cells.
  loc_.pamCounter = pamLength;

  // If True, the TM will compute a signature for each sequence
  loc_.collectSequenceStats = false;

  // This gets set when we receive a reset and cleared on the first compute
  // following a reset.
  loc_.resetCalled = false;

  // We keep track of the average input density here
  loc_.avgInputDensity = -1;

  // Keeps track of the length of the sequence currently being learned.
  loc_.learnedSeqLength = 0;

  // Keeps track of the moving average of all learned sequence length.
  loc_.avgLearnedSeqLength = 0.0;

  loc_.allocateStatesInCPP = true;
  loc_.retrieveLearningStates = false;
  nCells = loc_.numberOfCols * loc_.cellsPerColumn;

  cells4_ = new Cells4::Cells4(
      loc_.numberOfCols, loc_.cellsPerColumn, activationThreshold, minThreshold,
      newSynapseCount, segUpdateValidDuration, initialPerm, connectedPerm,
      permanenceMax, permanenceDec, permanenceInc, globalDecay, doPooling,
      loc_.seed, loc_.allocateStatesInCPP, checkSynapseConsistency);

  cells4_->setVerbosity(verbosity);
  cells4_->setPamLength(pamLength);
  cells4_->setMaxAge(maxAge);
  cells4_->setMaxInfBacktrack(maxInfBacktrack);
  cells4_->setMaxLrnBacktrack(maxLrnBacktrack);
  cells4_->setMaxSeqLength(maxSeqLength);
  cells4_->setMaxSegmentsPerCell(maxSegmentsPerCell);
  cells4_->setMaxSynapsesPerSegment(maxSynapsesPerSegment);

  // Note: State buffers are maintained by Cells4.

  // note: setOutputBuffer() may override this buffer.
  currentOutput_ = new Real[nCells];
  memset(currentOutput_, 0, nCells * sizeof(Real));
  currentOutputOwn_ = true;

  // initialize stats
  internalStats_["nPredictions"] = 0.0;
  internalStats_["totalMissing"] = 0.0;
  internalStats_["totalExtra"] = 0.0;
  internalStats_["pctExtraTotal"] = 0.0;
  internalStats_["pctMissingTotal"] = 0.0;
  internalStats_["predictionScoreTotal2"] = 0.0;
  internalStats_["falseNegativeScoreTotal"] = 0.0;
  internalStats_["falsePositiveScoreTotal"] = 0.0;

  confHistogram_.reset(new Real[nCells], std::default_delete<Real[]>());
  memset(confHistogram_.get(), 0, nCells * sizeof(Real));
}

BacktrackingTMCpp::~BacktrackingTMCpp() {
  if (cells4_)
    delete cells4_;
  if (currentOutputOwn_)
    delete[] currentOutput_;
  confHistogram_.reset();
  prevSequenceSignature_.reset();
}


void BacktrackingTMCpp::setOutputBuffer(Real32 *buf) {
    if (currentOutputOwn_)
      delete[] currentOutput_;
    currentOutput_ = buf;
    currentOutputOwn_ = false;
  };


// For backtrackingTMCpp we let Cells4 allocate the buffers.
// However, the language interfaces (Python) might want to take control of the
// buffers. This method can be called after creating the region to set the
// buffers.
void BacktrackingTMCpp::setStatePointers(Byte *infActiveT, Byte *infActiveT1,
                                         Byte *infPredT, Byte *infPredT1,
                                         Real *colConfidenceT,
                                         Real *colConfidenceT1,
                                         Real *cellConfidenceT,
                                         Real *cellConfidenceT1) {
  loc_.allocateStatesInCPP = false;
  cells4_->setStatePointers(infActiveT, infActiveT1, infPredT, infPredT1,
                            colConfidenceT, colConfidenceT1, cellConfidenceT,
                            cellConfidenceT1);
}

// Allows an interface to get the pointers to the buffers that
// are being used by Cells4.
void BacktrackingTMCpp::getStatePointers(Byte *&activeT, Byte *&activeT1,
                                         Byte *&predT, Byte *&predT1,
                                         Real *&colConfidenceT,
                                         Real *&colConfidenceT1,
                                         Real *&confidenceT,
                                         Real *&confidenceT1) const {
  cells4_->getStatePointers(activeT, activeT1, predT, predT1, colConfidenceT,
                            colConfidenceT1, confidenceT, confidenceT1);
}

UInt BacktrackingTMCpp::version() const { return TM_VERSION; }

Real *BacktrackingTMCpp::compute(Real *bottomUpInput, bool enableLearn,
                                 bool enableInference) {
  // Note: the expected width of bottomUpInput[] is number of columns.
  loc_.iterationIdx++;

  //  Run compute and retrieve selected state and member variables
  cells4_->compute(bottomUpInput, currentOutput_, enableInference,
                   enableLearn);
  loc_.avgLearnedSeqLength = cells4_->getAvgLearnedSeqLength();

  Byte *predictedState;
  if (loc_.collectStats) {
    const auto activeColumns = nonzero<Real>(bottomUpInput, (Size)loc_.numberOfCols);
    predictedState = (enableInference) ? cells4_->getInfPredictedStateT1()
                                       : cells4_->getLearnPredictedStateT1();
    _updateStatsInferEnd(internalStats_, activeColumns.data(), predictedState,
                         cells4_->getColConfidenceT1());
  }

  _computeOutput(); // note: modifies currentOutput_

  printComputeEnd(currentOutput_, enableLearn);
  loc_.resetCalled = false;
  return currentOutput_;
}

/********************
 * predict()
 *
 *  This function gives the future predictions for <nSteps> timesteps starting
 *   from the current TM state. The TM is returned to its original state at the
 *   end before returning.
 *
 *   1. We save the TM state.
 *   2. Loop for nSteps
 *
 *      a. Turn-on with lateral support from the current active cells
 *      b. Set the predicted cells as the next step's active cells. This step
 *         in learn and infer methods use input here to correct the predictions.
 *         We don't use any input here.
 *
 *   3. Revert back the TM state to the time before prediction
 *
 *  @param nSteps: (int) The number of future time steps to be predicted
 *  @returns: all the future predictions - an array of type "Real32" and
 *            shape (nSteps, numberOfCols). The ith row gives the tm prediction
 *            for each column at a future timestep (t+i+1).
 *
 ********************************/
std::shared_ptr<Real> BacktrackingTMCpp::predict(Size nSteps) {
  Size nCells = loc_.numberOfCols * loc_.cellsPerColumn;
  Size nCols = loc_.numberOfCols;

  tmSavedState_t pristineTPDynamicState;

  // Save the TM dynamic state, we will use to revert back in the end
  _getTPDynamicState(pristineTPDynamicState);

  NTA_ASSERT(nSteps > 0);

  // Create a buffer to hold all future predictions.
  std::shared_ptr<Real> multiStepColumnPredictions(
      new Real[nSteps * nCols], std::default_delete<Real[]>());

  // This is a(nSteps - 1) + half loop.Phase 2 in both learn and infer methods
  // already predicts for timestep(t + 1).We use that prediction for free and
  // save the half - a - loop of work.
  Size step = 0;
  while (true) {
    // We get the prediction for the columns in the next time step from
    // the topDownCompute() method. It internally uses confidences.
    Real *td = topDownCompute();
    fastbuffercopy<Real>(&multiStepColumnPredictions.get()[step * nCols], td,
                         nCols);

    // Cleanest way handle one and half loops
    if (step == nSteps - 1)
      break;
    step += 1;

    // Copy t into t-1
    fastbuffercopy<Byte>(cells4_->getInfActiveStateT1(),
                         cells4_->getInfActiveStateT(), nCells * sizeof(Byte));
    fastbuffercopy<Byte>(cells4_->getInfPredictedStateT1(),
                         cells4_->getInfPredictedStateT(),
                         nCells * sizeof(Byte));
    fastbuffercopy<Real>(cells4_->getCellConfidenceT1(),
                         cells4_->getCellConfidenceT(), nCells * sizeof(Byte));

    // Predicted state at "t-1" becomes the active state at "t"
    fastbuffercopy<Byte>(cells4_->getInfActiveStateT(),
                         cells4_->getInfPredictedStateT1(),
                         nCells * sizeof(Byte));

    // Predicted state and confidence are set in phase2.
    memset(cells4_->getInfPredictedStateT(), 0, nCells * sizeof(Byte));
    memset(cells4_->getCellConfidenceT(), 0, nCells * sizeof(Real));
    _inferPhase2();
  }

  // Revert the dynamic state to the saved state
  _setTPDynamicState(pristineTPDynamicState);

  return multiStepColumnPredictions;
}

Real *BacktrackingTMCpp::topDownCompute() {
  // For now,  we will assume there is no one above us and that bottomUpOut
  // is simply the output that corresponds to our currently stored column
  // confidences. Simply return the column confidences
  return cells4_->getColConfidenceT();
}

void BacktrackingTMCpp::_inferPhase2() {
  // This calls phase 2 of inference (used in multistep prediction).
  //_setStatePointers();
  cells4_->inferPhase2();
  //_copyAllocatedStates();
}

std::pair<UInt, UInt> BacktrackingTMCpp::trimSegments(Real minPermanence,
                                                      UInt32 minNumSyns) {
  // Print all cells if verbosity says to
  if (cells4_->getVerbosity() >= 5) {
    std::cout << "Cells, all segments:\n";
    printCells(false);
  }

  return cells4_->trimSegments(minPermanence, minNumSyns);
}

// Called at the end of learning and inference,
// this routine will update a number of stats in our _internalStats map,
// including our computed prediction score.
//    @internalStats internal stats dictionary
//    @bottomUpNZ list of the active bottom-up inputs
//         -first element is number of elements.
//    @predictedState The columns we predicted on the last time step
//         -should match the current bottomUpNZ in the best case
//    @colConfidence Column confidences we determined on the last time step.
// from line 945 of backtracking_tm.py
//
void BacktrackingTMCpp::_updateStatsInferEnd(
    std::map<std::string, Real> internalStats, const UInt32 *bottomUpNZ,
    const Byte *predictedState, const Real *colConfidence) {
  if (loc_.collectStats) {
    internalStats["nInfersSinceReset"] += 1;

    // Compute the prediction score, how well the prediction from the last
    // time step predicted the current bottom-up input  //line 945 of
    // backtracking_tm.py
    std::vector<std::vector<UInt>> patternNZs;
    patternNZs.push_back(std::vector<UInt>(*bottomUpNZ));
    std::shared_ptr<struct BacktrackingTMCpp::predictionResults_t> results;
    results = _checkPrediction(patternNZs, predictedState, colConfidence, false);

    // Store the stats that don't depend on burn-in
    internalStats["curPredictionScore2"] = results->conf[0].predictionScore;
    internalStats["curFalseNegativeScore"] =  1.0f - results->conf[0].posPredictionScore;
    internalStats["curFalsePositiveScore"] = results->conf[0].negPredictionScore;
    internalStats["curMissing"] = (Real)results->totalMissing;
    internalStats["curExtra"] = (Real)results->totalExtras;

    // If we are passed the burn-in period, update the accumulated stats
    // Here's what various burn-in values mean:
    //   0 : try to predict the first element of each sequence and all
    //   subsequent 1 : try to predict the second element of each sequence and
    //   all subsequent
    //    etc.
    if (internalStats["nInfersSinceReset"] > loc_.burnIn) {

      // Burn - in related stats
      Real numExpected = std::max<Real>(1.0, (Real)bottomUpNZ[0]); // first element is length.
      internalStats["nPredictions"] += 1.0f;
      internalStats["totalMissing"] += results->totalMissing;
      internalStats["totalExtra"] += results->totalExtras;
      internalStats["pctExtraTotal"] += 100.0f * results->totalExtras / numExpected;
      internalStats["pctMissingTotal"] += 100.0f * results->totalMissing / numExpected;
      internalStats["predictionScoreTotal2"] += results->conf[0].predictionScore;
      internalStats["falseNegativeScoreTotal"] += 1.0f - results->conf[0].posPredictionScore;
      internalStats["falsePositiveScoreTotal"] += results->conf[0].negPredictionScore;

      if (loc_.collectSequenceStats) {
        // Collect cell confidences for every cell that correctly predicted
        // current bottom up input. Normalize confidence across each column
        Real *cc = new Real[nCells];
        Real *cellT1 = cells4_->getCellConfidenceT1();
        Byte *cellState = cells4_->getInfActiveStateT();
        for (Size i = 0; i < (Size)loc_.numberOfCols; i++) {
          Real sconf = 0;
          for (Size j = 0; j < (Size)loc_.cellsPerColumn; j++) {
            // zero out confidence if state bit is zero
            cc[i * j] = cellT1[i * j] * cellState[i * j];
            // add up confidence of a column
            sconf += cc[i * j];
          }
          if (sconf > 0) {
            // Normalize the confidence for each cell in the column
            for (Size j = 0; j < (Size)loc_.cellsPerColumn; j++) {
              cc[i * j] /= sconf;
            }
          }
        }

        // Update cell confidence histogram: add column-normalized confidence
        // scores to the histogram
        Real *ch = confHistogram_.get();
        for (Size i = 0; i < nCells; i++) {
          ch[i] += cc[i];
        }
        delete[] cc;
      }
    }
  }
}

/************************************************
 *   _checkPrediction()
 *   This function produces goodness-of-match scores for a set of input
 *patterns, by checking for their presence in the current and predicted output
 *of the TM. Returns a global count of the number of extra and missing bits,
 *the confidence scores for each input pattern, and (if requested) the bits
 *in each input pattern that were not present in the TM's prediction. (from
 *line 2892 of backtracking_tm.py)
 *
 *   @patternNZs[][] a list of input patterns that we want to check for. Each
 *              element is a list of the indexes of non-zeros in that
 *              pattern. First element of each pattern is length of pattern.
 *   @output    The output of the TM (predictedState).
 *   @colConfidence The column confidences. If not specified, then use the
 *              TM's current colConfidence_. This can be specified
 *              if you are trying to check the prediction metrics
 *              for an output from the past.
 *   @details   if True, also include details of missing bits per pattern.
 *
 *   :returns: struct predictionResults_t
 *               totalExtras,
 *               totalMissing,
 *               [conf_1, conf_2, ...],        a vector of structures
 *               [missing1, missing2, ...]
 *
 *   @retval totalExtras a global count of the number of 'extras', i.e. bits
 *               that are on in the current output but not in the or of all
 *               the passed in patterns.
 *   @retval totalMissing a global count of all the missing bits, i.e. the bits
 *                        that are on in the or of the patterns, but not in the
 *                        current output
 *   @retval conf_[] the confidence score for the i'th pattern inpatternsToCheck
 *                  This consists of 3 items as a tuple:
 *                    Real predictionScore
 *                    Real posPredictionScore
 *                    Real negPredictionScore
 *                  This is an array of structures, one for each pattern.
 *   @retval missing_[] the bits in an OR of All pattern that were missing
 *                     in the output. caller allocates to size of output.
 *                     This list is only returned if details is true.
 **************************************************/
std::shared_ptr<struct BacktrackingTMCpp::predictionResults_t> BacktrackingTMCpp::_checkPrediction(
    std::vector<std::vector<UInt>> patternNZs,
    const Byte *output,
    const Real *colConfidence,
    bool details)
{
  std::shared_ptr<struct predictionResults_t> results(new struct predictionResults_t);

  if (details) {
    std::shared_ptr<Real> sp(new Real[nCells], std::default_delete<Real[]>());
    memset(sp.get(), 0, nCells * sizeof(Real));
    results->missing = sp;
  }


  // Compute the union of all the expected patterns
  std::set<UInt32> orAll;
  for (size_t i = 0; i < patternNZs.size(); i++) {
    for (size_t n = 1; n < patternNZs[i].size(); n++) {
      orAll.insert(patternNZs[i][n]);
    }
  }

  // Get the list of active columns in the output
  std::set<UInt32> outputnz;
  if (output) {
    for (Size i = 0; i < nCells; i++) {
      if (output[i] != 0)
        outputnz.insert((UInt32)i);
    }
  } else {
    NTA_CHECK(currentOutput_);
    for (Size i = 0; i < nCells; i++) {
      if (currentOutput_[i] != 0.0)
        outputnz.insert((UInt32)i);
    }
  }


  // Compute the total extra and missing in the output
  results->totalExtras = 0;
  results->totalMissing = 0;
  std::set<UInt32>::iterator first1 = outputnz.begin();
  std::set<UInt32>::iterator last1 = outputnz.end();
  std::set<UInt32>::iterator first2 = orAll.begin();
  std::set<UInt32>::iterator last2 = orAll.end();
  while (true) {
    if (first1 == last1) {
      if (first2 == last2)
        break;
      results->totalMissing++; // it is in orAll but not in outputnz.
      if (details)
        results->missing.get()[*first2] = 1;
      ++first2;
    } else if (first2 == last2) {
      results->totalExtras++; // it is in outputnz but not in orAll.
      first1++;
    } else {
      if (*first1 < *first2) {
        results->totalExtras++; // it is in outputnz but not in orAll.
        ++first1;
      } else if (*first2 < *first1) {
        results->totalMissing++; // it is in orAll but not in outputnz.
        if (details)
          results->missing.get()[*first2] = 1;
        ++first2;
      } else {
        ++first1;
        ++first2;
      }
    }
  }

  // Get the percent confidence level per column by summing the confidence
  // levels of the cells in the column.During training, each segment's
  // confidence number is computed as a running average of how often it
  // correctly predicted bottom-up activity on that column.  A cell's
  // confidence number is taken from the first active segment found in the cell.
  // Note that confidence will only be non-zero for predicted columns.
  if (colConfidence == nullptr)
    colConfidence = cells4_->getColConfidenceT();
  for (auto pattern : patternNZs) {
    struct score_tuple scores;

    // Sum of the column confidences for this pattern?
    const Size positiveColumnCount = pattern.size();
    Real positivePredictionSum = 0;
    for (UInt i=0; i < pattern.size(); i++) {
	    //debug
	    std::cout << "=====================================\n";
	   cout << "pattern= ";
	   for (auto e : pattern) cout << e << ", ";
	  cout << "confidences= ";
	  for(UInt i=0; i < sizeof(colConfidence)/sizeof(colConfidence[0]); i++) cout << colConfidence[i] << ", ";
//      NTA_ASSERT(pattern[i] < positiveColumnCount) << "BackTM: " << pattern[i] << " of " << positiveColumnCount;
      //end-debug
      positivePredictionSum += colConfidence[i];
    }

    // Sum of all the column confidences
    Real totalPredictionSum = 0;
    for (Size i = 1; i < loc_.numberOfCols; i++) {
      totalPredictionSum += colConfidence[i];
    }

    // Total number of columns
    Size totalColumnCount = loc_.numberOfCols;

    Real negativePredictionSum = totalPredictionSum - positivePredictionSum;
    Size negativeColumnCount = totalColumnCount - positiveColumnCount;

    //  Compute the average confidence score per column for this pattern
    //  Compute the average confidence score per column for the other patterns
    scores.posPredictionScore =
        (positiveColumnCount == 0) ? 0.0f : positivePredictionSum;
    scores.negPredictionScore =
        (negativeColumnCount == 0) ? 0.0f : negativePredictionSum;

    // Scale the positive and negative prediction scores so that they sum to 1.0
    Real currentSum = scores.negPredictionScore + scores.posPredictionScore;
    if (currentSum > 0.0) {
      scores.posPredictionScore *= 1.0f / currentSum;
      scores.negPredictionScore *= 1.0f / currentSum;
    }
    scores.predictionScore =
        scores.posPredictionScore - scores.negPredictionScore;

    results->conf.push_back(scores);
  }
  return results;
}

////////////////////////////////////
//  _computeOutput();
//    Computes output for both learning and inference. In both cases, the
//    output is the boolean OR of 'activeState' and 'predictedState' at 't'.
//    Stores 'currentOutput_'.
Real32 *BacktrackingTMCpp::_computeOutput() {
  if (outputType_ == "activeState1CellPerCol") {
    // Fire only the most confident cell in columns that have 2 or more active
    // cells Don't turn on anything in columns which are not active at all
    Byte *active = cells4_->getInfActiveStateT();
    Real *cc = cells4_->getCellConfidenceT();
    for (Size i = 0; i < (Size)loc_.numberOfCols; i++) {
      Size isColActive = 0;
      Size mostConfidentCell = 0;
      Real c = 0;
      for (Size j = 0; j < loc_.cellsPerColumn; j++) {
        Size cellIdx = i * loc_.cellsPerColumn + j;
        if (cc[cellIdx] > c) {
          c = cc[cellIdx];
          mostConfidentCell = cellIdx;
        }
        currentOutput_[cellIdx] = 0; // zero the output
        isColActive += active[cellIdx];
      }
      // set the most confident cell in this column if active.
      if (c > 0 && isColActive)
        currentOutput_[mostConfidentCell] = 1.0f;
    }

  } else if (outputType_ == "activeState") {
    Byte *active = cells4_->getInfActiveStateT();
    for (Size i = 0; i < nCells; i++) {
      currentOutput_[i] = active[i];
    }

  } else if (outputType_ == "normal") {
    Byte *active = cells4_->getInfActiveStateT();
    Byte *predicted = cells4_->getInfPredictedStateT();
    for (Size i = 0; i < nCells; i++) {
      currentOutput_[i] = (active[i] || predicted[i]) ? 1.0f : 0.0f;
    }

  } else {
    NTA_THROW << "Unimplemented outputType '" << outputType_ << "' ";
  }
  return currentOutput_;
}

void BacktrackingTMCpp::reset() {
  if (cells4_->getVerbosity() >= 3)
    std::cout << "\n==== TM Reset =====" << std::endl;
  //_setStatePointers()
  cells4_->reset();
  // All state buffers have been filled with 0's by the reset.

  internalStats_["nInfersSinceReset"] = 0;

  // To be removed
  internalStats_["curPredictionScore"] = 0;
  // New prediction score
  internalStats_["curPredictionScore2"] = 0;
  internalStats_["curFalseNegativeScore"] = 0;
  internalStats_["curFalsePositiveScore"] = 0;
  internalStats_["curMissing"] = 0;
  internalStats_["curExtra"] = 0;

  // When a reset occurs, set prevSequenceSignature to the signature of the
  // just - completed sequence (confHistogram_) and start accumulating
  // histogram for the next sequence.
  prevSequenceSignature_.reset();
  if (loc_.collectSequenceStats) {
    prevSequenceSignature_ = confHistogram_; // copy smart pointer.
    // initialize a new histogram.
    confHistogram_.reset(new Real[nCells], std::default_delete<Real[]>());
    memset(confHistogram_.get(), 0, nCells * sizeof(Real));
  }
  loc_.resetCalled = true;
}

void BacktrackingTMCpp::resetStats() {
  // Reset the learning and inference stats. This will usually be called by
  // user code at the start of each inference run (for a particular data set).

  internalStats_.clear();

  internalStats_["nInfersSinceReset"] = 0;
  internalStats_["nPredictions"] = 0;

  // New prediction score
  internalStats_["curPredictionScore"] = 0;
  internalStats_["curPredictionScore2"] = 0;
  internalStats_["predictionScoreTotal2"] = 0;
  internalStats_["curFalseNegativeScore"] = 0;
  internalStats_["falseNegativeScoreTotal"] = 0;
  internalStats_["curFalsePositiveScore"] = 0;
  internalStats_["falsePositiveScoreTotal"] = 0;

  internalStats_["pctExtraTotal"] = 0;
  internalStats_["pctMissingTotal"] = 0;
  internalStats_["curMissing"] = 0;
  internalStats_["curExtra"] = 0;
  internalStats_["totalMissing"] = 0;
  internalStats_["totalExtra"] = 0;

  // Sequence signature statistics. Note that we don't reset the sequence
  // signature list itself. Just the histogram.
  prevSequenceSignature_.reset();
  if (loc_.collectSequenceStats) {
    confHistogram_.reset(new Real[nCells], std::default_delete<Real[]>());
    memset(confHistogram_.get(), 0, nCells * sizeof(Real));
  }
}

//   getStats()
//     Return the current learning and inference stats. This returns a map
//    containing all the learning and inference stats we have collected since
//    the last :meth:`resetStats` call. If :class:`BacktrackingTMCpp`
//    ``collectStats`` parameter is False, then an empty map is returned.
//
//    @returns: (map) The following keys are returned in the map when
//        ``collectStats`` is True:
//
//          - 'nPredictions': the number of predictions. This is the total
//                number of inferences excluding burn-in and the last inference.
//          - 'curPredictionScore': the score for predicting the current input
//                (predicted during the previous inference)
//          - 'curMissing': the number of bits in the current input that were
//          not predicted to be on.
//          - 'curExtra': the number of bits in the predicted output that are
//          not in the next input
//          - 'predictionScoreTotal': the sum of every prediction score to date
//          - 'predictionScoreAvg': 'predictionScoreTotal / nPredictions'
//          - 'pctMissingTotal': the total number of bits that were missed over
//          all predictions
//          - 'pctMissingAvg': 'pctMissingTotal / nPredictions``
//      The map is empty if `'collectSequenceStats' is False.

std::map<std::string, Real32> BacktrackingTMCpp::getStats() {
  std::map<std::string, Real32> stats;
  if (!loc_.collectStats) {
    return stats;
  }

  stats["nPredictions"] = internalStats_["nPredictions"];
  stats["curMissing"] = internalStats_["curMissing"];
  stats["curExtra"] = internalStats_["curExtra"];
  stats["totalMissing"] = internalStats_["totalMissing"];
  stats["totalExtra"] = internalStats_["totalExtra"];
  Real nPredictions = std::max<Real>(1.0, stats["nPredictions"]);

  // New prediction score
  stats["curPredictionScore2"] = internalStats_["curPredictionScore2"];
  stats["predictionScoreAvg2"] =
      internalStats_["predictionScoreTotal2"] / nPredictions;
  stats["curFalseNegativeScore"] = internalStats_["curFalseNegativeScore"];
  stats["falseNegativeAvg"] =
      internalStats_["falseNegativeScoreTotal"] / nPredictions;
  stats["curFalsePositiveScore"] = internalStats_["curFalsePositiveScore"];
  stats["falsePositiveAvg"] =
      internalStats_["falsePositiveScoreTotal"] / nPredictions;

  stats["pctExtraAvg"] = internalStats_["pctExtraTotal"] / nPredictions;
  stats["pctMissingAvg"] = internalStats_["pctMissingTotal"] / nPredictions;

  return stats;
}

// returns the preivous value of confidence Histogram
// The signature for the sequence immediately  preceding the last reset.
// This will be empty if collectSequenceStats is False
std::shared_ptr<Real> BacktrackingTMCpp::getPrevSequenceSignature() {
  return prevSequenceSignature_;
}

// current value of confidence Histogram
// This will be empty if collectSequenceStats is False
std::shared_ptr<Real> BacktrackingTMCpp::getConfHistogram() {
  return confHistogram_;
}

/////////////// saving dynamic state for predict() //////////////
void BacktrackingTMCpp::_getTPDynamicState(tmSavedState_t &ss) {
  deepcopySave_(ss, "infActiveStateT", cells4_->getInfActiveStateT(),
                      nCells);
  deepcopySave_(ss, "infActiveStateT1", cells4_->getInfActiveStateT1(),
                      nCells);
  deepcopySave_(ss, "infPredictedStateT",
                      cells4_->getInfPredictedStateT(), nCells);
  deepcopySave_(ss, "infPredictedStateT1",
                      cells4_->getInfPredictedStateT1(), nCells);
  deepcopySave_(ss, "learnActiveStateT", cells4_->getLearnActiveStateT(),
                      nCells);
  deepcopySave_(ss, "learnActiveStateT1",
                      cells4_->getLearnActiveStateT1(), nCells);
  deepcopySave_(ss, "learnPredictedStateT",
                      cells4_->getLearnPredictedStateT(), nCells);
  deepcopySave_(ss, "learnPredictedStateT1",
                      cells4_->getLearnPredictedStateT1(), nCells);
  deepcopySave_(ss, "cellConfidenceT", cells4_->getCellConfidenceT(),
                      nCells);
  deepcopySave_(ss, "cellConfidenceT1", cells4_->getCellConfidenceT1(),
                      nCells);
  deepcopySave_(ss, "colConfidenceT", cells4_->getColConfidenceT(),
                      loc_.numberOfCols);
  deepcopySave_(ss, "colConfidenceT", cells4_->getColConfidenceT(),
                      loc_.numberOfCols);
}

void BacktrackingTMCpp::_setTPDynamicState(tmSavedState_t &ss) {
  deepcopyRestore_(ss, "infActiveStateT", cells4_->getInfActiveStateT(),
                         nCells);
  deepcopyRestore_(ss, "infActiveStateT1", cells4_->getInfActiveStateT1(),
                         nCells);
  deepcopyRestore_(ss, "infPredictedStateT",
                         cells4_->getInfPredictedStateT(), nCells);
  deepcopyRestore_(ss, "infPredictedStateT1",
                         cells4_->getInfPredictedStateT1(), nCells);
  deepcopyRestore_(ss, "learnActiveStateT",
                         cells4_->getLearnActiveStateT(), nCells);
  deepcopyRestore_(ss, "learnActiveStateT1",
                         cells4_->getLearnActiveStateT1(), nCells);
  deepcopyRestore_(ss, "learnPredictedStateT",
                         cells4_->getLearnPredictedStateT(), nCells);
  deepcopyRestore_(ss, "learnPredictedStateT1",
                         cells4_->getLearnPredictedStateT1(), nCells);
  deepcopyRestore_(ss, "cellConfidenceT", cells4_->getCellConfidenceT(),
                         nCells);
  deepcopyRestore_(ss, "cellConfidenceT1", cells4_->getCellConfidenceT1(),
                         nCells);
  deepcopyRestore_(ss, "colConfidenceT", cells4_->getColConfidenceT(),
                         loc_.numberOfCols);
  deepcopyRestore_(ss, "colConfidenceT", cells4_->getColConfidenceT(),
                         loc_.numberOfCols);
}

void BacktrackingTMCpp::deepcopySave_(tmSavedState_t &ss, std::string name,
                                      Byte *buf, Size count) {
  std::shared_ptr<Byte> target(new Byte[count], std::default_delete<Byte[]>());
  fastbuffercopy<Byte>(target.get(), buf, count);
  struct ss_t val;
  val.Byteptr = target;
  ss[name] = val;
}
void BacktrackingTMCpp::deepcopySave_(tmSavedState_t &ss, std::string name,
                                      Real *buf, Size count) {
  std::shared_ptr<Real> target(new Real[count], std::default_delete<Real[]>());
  fastbuffercopy<Real>(target.get(), buf, count);
  struct ss_t val;
  val.Realptr = target;
  ss[name] = val;
}

void BacktrackingTMCpp::deepcopyRestore_(tmSavedState_t &ss, std::string name,
                                         Byte *buf, Size count) {
  struct ss_t val = ss[name];
  fastbuffercopy<Byte>(buf, val.Byteptr.get(), count);
}
void BacktrackingTMCpp::deepcopyRestore_(tmSavedState_t &ss, std::string name,
                                         Real *buf, Size count) {
  struct ss_t val = ss[name];
  fastbuffercopy<Real>(buf, val.Realptr.get(), count);
}

template <typename T>
void BacktrackingTMCpp::fastbuffercopy(T *tobuf, T *frombuf, Size count) {
  memcpy(tobuf, frombuf, count * sizeof(T));
}

///////////////  printing routines for Debug  ///////////////
void BacktrackingTMCpp::printComputeEnd(Real *output, bool learn,
                                        std::ostream &out) const {
  // Called at the end of inference to print out various diagnostic information
  // based on the current verbosity level

  if (cells4_->getVerbosity() >= 3) {
    out << "----- computeEnd summary: \n";
    out << "learn:" << learn << "\n";

    // Inferred Active State
    Size numBurstingCols = 0;
    Size numOn = 0;
    Byte *ptr = cells4_->getInfActiveStateT();
    for (Size i = 0; i < loc_.numberOfCols; i++) {
      bool isbursting = true;
      for (Size j = 0; j < loc_.cellsPerColumn; j++) {
        if (ptr[i * loc_.cellsPerColumn + j] == 0) {
          isbursting = false;
          break;
        } else {
          numOn++;
        }
      }
      if (isbursting)
        numBurstingCols++;
    }
    Real curPredictionScore2 =
        internalStats_.find("curPredictionScore2")->second;
    Real curFalsePositiveScore =
        internalStats_.find("curFalsePositiveScore")->second;
    Real curFalseNegativeScore =
        internalStats_.find("curFalseNegativeScore")->second;
    out << "numBurstingCols:    " << numBurstingCols << "\n";
    out << "curPredScore2:      " << curPredictionScore2 << "\n";
    out << "curFalsePosScore:   " << curFalsePositiveScore << "\n";
    out << "1-curFalseNegScore: " << (1.0f - curFalseNegativeScore) << "\n";
    out << "numSegments:        " << getNumSegments() << "\n";
    out << "avgLearnedSeqLength:" << loc_.avgLearnedSeqLength << "\n";

    out << "\n----- infActiveState (" << numOn << " on) ------\n";
    ptr = cells4_->getInfActiveStateT();
    printActiveIndicesByte(ptr, false, out);
    if (cells4_->getVerbosity() >= 6)
      printState(cells4_->getInfActiveStateT(), out);

    // Inferred Predicted state
    ptr = cells4_->getInfPredictedStateT();
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    out << "\n----- infPredictedState (" << numOn << " on)-----\n";
    printActiveIndicesByte(ptr, false, out);
    if (cells4_->getVerbosity() >= 6)
      printState(ptr, out);

    // Learned Active State
    ptr = cells4_->getLearnActiveStateT();
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    out << "\n----- lrnActiveState (" << numOn << " on) ------\n";
    printActiveIndicesByte(ptr, false, out);
    if (cells4_->getVerbosity() >= 6)
      printState(ptr, out);

    // Learned predicted state
    ptr = cells4_->getLearnPredictedStateT1();
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    out << "\n----- lrnPredictedState (" << numOn << " on)-----\n";
    printActiveIndicesByte(ptr, false, out);
    if (cells4_->getVerbosity() >= 6)
      printState(ptr, out);

    // Cell Confidence
    Real *rptr = cells4_->getCellConfidenceT();
    out << "\n----- cellConfidence -----\n";
    printActiveIndicesReal(rptr, true, out);
    if (cells4_->getVerbosity() >= 6)
      printConfidence(rptr, 20, out);

    // Column Confidence
    rptr = cells4_->getColConfidenceT();
    out << "\n----- colConfidence -----\n";
    printColActiveIndices(rptr, true, out);

    // T-1 cell confidence for active cells
    out << "\n----- cellConfidence[t-1] for currently active cells -----\n";
    ptr = cells4_->getInfActiveStateT();
    Real *c = cells4_->getCellConfidenceT1();
    Real *cc = new Real[nCells];
    for (Size i = 0; i < nCells; i++) {
      cc[i] = (ptr[i]) ? c[i] : 0;
    }
    printActiveIndicesReal(cc, true, out);
    delete[] cc;

    if (cells4_->getVerbosity() >= 4) {
      out << "\nCells, predicted segments only:\n";
      printCells(true, out);
    } else if (cells4_->getVerbosity() >= 5) {
      out << "\nCells, all segments:\n";
      printCells(false, out);
    }
    out << std::endl; // flush buffer

  } else if (cells4_->getVerbosity() >= 1) {
    out << "\nTM: learn:" << learn << "\n";

    const auto outputnz = nonzero<Real>(output, nCells);
    out << "\nTM: active outputs(" << outputnz[0] << ")\n";
    printActiveIndicesReal(output, false, out);
  }
}

// Print the list of '[column, cellIdx]' indices for each of the active cells in
// state.
void BacktrackingTMCpp::printActiveIndicesByte(const Byte *state,
                                               bool andValues,
                                               std::ostream &out) const {
  for (Size i = 0; i < loc_.numberOfCols; i++) {
    out << "Col " << i << ": [";
    bool first = true;
    for (Size j = 0; j < loc_.cellsPerColumn; j++) {
      if (state[i * loc_.cellsPerColumn + j]) {
        if (!first)
          out << ", ";
        if (andValues)
          out << j << ": " << state[i * loc_.cellsPerColumn + j];
        else
          out << j;
        first = false;
      }
    }
    out << "]\n";
  }
}
// Print the list of '[column, cellIdx]' indices for each of the active cells in
// confidence.
void BacktrackingTMCpp::printActiveIndicesReal(const Real *confidence,
                                               bool andValues,
                                               std::ostream &out) const {
  for (Size i = 0; i < loc_.numberOfCols; i++) {
    out << "Col " << i << ": [";
    bool first = true;
    for (Size j = 0; j < loc_.cellsPerColumn; j++) {
      if (confidence[i * loc_.cellsPerColumn + j]) {
        if (!first)
          out << ", ";
        if (andValues)
          out << j << ": " << confidence[i * loc_.cellsPerColumn + j];
        else
          out << j;
        first = false;
      }
    }
    out << "]\n";
  }
}
// Print the list of '[column]' indices for each of the active cells in
// ColConfidence.
void BacktrackingTMCpp::printColActiveIndices(const Real *colconfidence,
                                              bool andValues,
                                              std::ostream &out) const {
  out << "[";
  bool first = true;
  for (Size j = 0; j < loc_.numberOfCols; j++) {
    if (colconfidence[j]) {
      if (!first)
        out << ", ";
      if (andValues)
        out << j << ": " << colconfidence[j];
      else
        out << j;
      first = false;
    }
  }
  out << "]\n";
}

// Print an integer array that is the same shape as activeState.
void BacktrackingTMCpp::printState(const Byte *aState,
                                   std::ostream &out) const {
  for (Size i = 0; i < loc_.cellsPerColumn; i++) {

    for (Size c = 0; c < loc_.numberOfCols; c++) {
      if (c % 10 == 0)
        out << " "; // add a spacer every 10
      out << aState[c * loc_.cellsPerColumn + i] << " ";
    }
    out << "\n";
  }
}

// Print a floating point array that is the same shape as activeState.
void BacktrackingTMCpp::printConfidence(const Real *aState, Size maxCols,
                                        std::ostream &out) const {
  char buf[20];
  for (Size i = 0; i < loc_.cellsPerColumn; i++) {
    for (Size c = 0; c < std::min(maxCols, (Size)loc_.numberOfCols); c++) {
      if (c % 10 == 0)
        out << "  "; // add a spacer every 10
      sprintf(buf, " %5.3f", aState[c * loc_.cellsPerColumn + i]);
      out << buf;
    }
    out << "\n";
  }
}

// Print up to maxCols number from a flat floating point array.
void BacktrackingTMCpp::printColConfidence(const Real *aState, Size maxCols,
                                           std::ostream &out) const {
  char buf[20];
  for (Size c = 0; c < std::min(maxCols, (Size)loc_.numberOfCols); c++) {
    if (c % 10 == 0)
      out << "  "; // add a spacer every 10
    sprintf(buf, " %5.3f", aState[c]);
    out << buf;
  }
  out << "\n";
}

void BacktrackingTMCpp::printCells(bool predictedOnly,
                                   std::ostream &out) const {
  if (predictedOnly)
    out << "--- PREDICTED CELLS ---\n";
  else
    out << "--- ALL CELLS ---\n";
  out << "Activation threshold=" << cells4_->getActivationThreshold() << "\n";
  out << "min threshold=" << cells4_->getMinThreshold() << "\n";
  out << "connected perm=" << cells4_->getPermConnected() << "\n";

  Byte *ptr = cells4_->getInfPredictedStateT();
  for (Size c = 0; c < loc_.numberOfCols; c++) {
    for (Size i = 0; i < loc_.cellsPerColumn; i++) {
      if (!predictedOnly || ptr[c * loc_.cellsPerColumn + i])
        printCell(c, i, predictedOnly, out);
    }
  }
}

void BacktrackingTMCpp::printCell(Size c, Size i, bool onlyActiveSegments,
                                  std::ostream &out) const {
  char buff[1000];
  Size nSegs = (Size)cells4_->nSegmentsOnCell((UInt)c, (UInt)i);
  if (nSegs > 0) {
    vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
    out << "Col " << c << ", Cell " << i << " ("
        << (c * loc_.cellsPerColumn + i) << ") : " << nSegs << " segment(s)\n";
    for (const auto segIdx : segList) {
      Segment &seg = cells4_->getSegment((UInt)c, (UInt)i, segIdx);
      const bool isActive = _slowIsSegmentActive(seg, string("t"));
      if (!onlyActiveSegments || isActive) {
        snprintf(buff, sizeof(buff), "%sSeg #%-3d %d %c %9.7f (%4d/%-4d) %4d ",
                 ((isActive) ? "*" : " "), segIdx, (int)seg.size(),
                 (seg.isSequenceSegment()) ? 'T' : 'F',
                 seg.dutyCycle(cells4_->getNLrnIterations(), false, true),
                 seg.getPositiveActivations(), seg.getTotalActivations(),
                 cells4_->getNLrnIterations() - seg.getLastActiveIteration());
        for (UInt idx = 0; idx < (UInt)seg.size(); idx++) {
          const Size len = sizeof(buff) - strlen(buff);
          const Size srcIdx = seg.getSrcCellIdx(idx);
          snprintf(buff + strlen(buff), len, "[%2d,%-2d]%4.2f ",
                   (int)_getCellCol(srcIdx), (int)_getCellIdx(srcIdx),
                   seg.getPermanence(idx));
        }
        cout << buff << "\n";
      }
    }
  }
}

void BacktrackingTMCpp::printInput(const Real32 *x, std::ostream &out) const {
  out << "Input\n";
  for (Size c = 0; c < loc_.numberOfCols; c++) {
    out << (int)x[c] << " ";
  }
  out << std::endl;
}

void BacktrackingTMCpp::printOutput(const Real32 *y, std::ostream &out) const {
  char buff[100];
  out << "Output\n";
  for (Size i = 0; i < loc_.cellsPerColumn; i++) {
    snprintf(buff, sizeof(buff), "[%3d] ", (int)i);
    out << buff;
    for (Size c = 0; c < loc_.numberOfCols; c++) {
      out << (int)y[c * loc_.numberOfCols + i] << " ";
    }
    out << "\n";
  }
  out << std::endl;
}

//     Print the parameter settings for the TM.
void BacktrackingTMCpp::printParameters(std::ostream &out) const {
  out << "numberOfCols=", loc_.numberOfCols;
  out << "cellsPerColumn=", loc_.cellsPerColumn;
  out << "minThreshold=", cells4_->getMinThreshold();
  out << "newSynapseCount=", cells4_->getNewSynapseCount();
  out << "activationThreshold=", cells4_->getActivationThreshold();
  out << "\n";
  out << "initialPerm=", cells4_->getPermInitial();
  out << "connectedPerm=", cells4_->getPermConnected();
  out << "permanenceInc=", cells4_->getPermInc();
  out << "permanenceDec=", cells4_->getPermDec();
  out << "permanenceMax=", cells4_->getPermMax();
  out << "globalDecay=", cells4_->getGlobalDecay();
  out << "\n";
  out << "doPooling=", cells4_->getDoPooling();
  out << "segUpdateValidDuration=", cells4_->getSegUpdateValidDuration();
  out << "pamLength=", cells4_->getPamLength();
  out << std::endl;
}
void BacktrackingTMCpp::printSegment(Segment &s, std::ostream &out) const {
  s.print(out, loc_.cellsPerColumn);
}

static char *formatRow(char *buffer, Size bufsize, const Byte *val, Size i,
                       Size numberOfCols, Size cellsPerColumn) {
  char *ptr = buffer;
  Size len = bufsize - 4;
  for (Size c = 0; c < numberOfCols; c++) {
    if (c > 0 && c % 10 == 0) {
      *ptr++ = ' ';
      len--;
    }
    *ptr++ = (val[c * cellsPerColumn + i]) ? '1' : '0';
    len--;
    if (len <= 0)
      break; // out of buffer space
  }
  *ptr++ = ' ';
  *ptr++ = '\0';
  return buffer;
}
void BacktrackingTMCpp::printStates(bool printPrevious, bool printLearnState,
                                    std::ostream &out) const {
  char buffer[5000]; // temporary scratch space

  out << "\nInference Active state\n";
  for (Size i = 0; i < loc_.cellsPerColumn; i++) {
    if (printPrevious)
      out << formatRow(buffer, sizeof(buffer), cells4_->getInfActiveStateT1(),
                       i, loc_.numberOfCols, loc_.cellsPerColumn);
    out << formatRow(buffer, sizeof(buffer), cells4_->getInfActiveStateT(), i,
                     loc_.numberOfCols, loc_.cellsPerColumn);
    out << "\n";
  }
  out << "\nInference Predicted state\n";
  for (Size i = 0; i < loc_.cellsPerColumn; i++) {
    if (printPrevious)
      out << formatRow(buffer, sizeof(buffer),
                       cells4_->getInfPredictedStateT1(), i, loc_.numberOfCols,
                       loc_.cellsPerColumn);
    out << formatRow(buffer, sizeof(buffer), cells4_->getInfPredictedStateT(),
                     i, loc_.numberOfCols, loc_.cellsPerColumn);
    out << "\n";
  }

  if (printLearnState) {
    out << "\nLearn Active state\n";
    for (Size i = 0; i < loc_.cellsPerColumn; i++) {
      if (printPrevious)
        out << formatRow(buffer, sizeof(buffer),
                         cells4_->getLearnActiveStateT1(), i, loc_.numberOfCols,
                         loc_.cellsPerColumn);
      out << formatRow(buffer, sizeof(buffer), cells4_->getLearnActiveStateT(),
                       i, loc_.numberOfCols, loc_.cellsPerColumn);
      out << "\n";
    }

    out << "Learn Predicted state\n";
    for (Size i = 0; i < loc_.cellsPerColumn; i++) {
      if (printPrevious)
        out << formatRow(buffer, sizeof(buffer),
                         cells4_->getLearnPredictedStateT1(), i,
                         loc_.numberOfCols, loc_.cellsPerColumn);
      out << formatRow(buffer, sizeof(buffer),
                       cells4_->getLearnPredictedStateT(), i, loc_.numberOfCols,
                       loc_.cellsPerColumn);
      out << "\n";
    }
  }
}

  ////////////////////////////////////////////////////////////////////////
  ////       compare two TM's
  ////////////////////////////////////////////////////////////////////////

  // Given two TM instances, see if any parameters are different.
  // If verbosity > 0 it will also print out the differences.

#define PARAMETER_CHECK(name, call)                                            \
  if (tm1.call != tm2.call) {                                                  \
    result = false;                                                            \
    if (verbosity > 0) {                                                       \
      out << name ":  " << tm1.call << " != " << tm2.call << std::endl;        \
    }                                                                          \
  }

bool BacktrackingTMCpp::sameTMParams(const BacktrackingTMCpp &tm1,
                                     const BacktrackingTMCpp &tm2,
                                     std::ostream &out, Int32 verbosity) {
  bool result = true;
  PARAMETER_CHECK("numberOfCols", getnumCol());
  PARAMETER_CHECK("cellsPerColumn", getcellsPerCol());
  PARAMETER_CHECK("initialPerm", getInitialPerm());
  PARAMETER_CHECK("connectedPerm", getConnectedPerm());
  PARAMETER_CHECK("minThreshold", getMinThreshold());
  PARAMETER_CHECK("newSynapseCount", getNewSynapseCount());
  PARAMETER_CHECK("permanenceInc", getPermanenceInc());
  PARAMETER_CHECK("permanenceDec", getPermanenceDec());
  PARAMETER_CHECK("permanenceMax", getPermanenceMax());
  PARAMETER_CHECK("globalDecay", getGlobalDecay());
  PARAMETER_CHECK("activationThreshold", getActivationThreshold());
  PARAMETER_CHECK("doPooling", getDoPooling());
  PARAMETER_CHECK("segUpdateValidDuration", getSegUpdateValidDuration());
  PARAMETER_CHECK("verbosity", getVerbosity());
  PARAMETER_CHECK("checkSynapseConsistency", getCheckSynapseConsistency());
  PARAMETER_CHECK("pamLength", getPamLength());
  PARAMETER_CHECK("maxInfBacktrack", getMaxInfBacktrack());
  PARAMETER_CHECK("maxLrnBacktrack", getMaxLrnBacktrack());
  PARAMETER_CHECK("maxAge", getMaxAge());
  PARAMETER_CHECK("maxSeqLength", getMaxSeqLength());
  PARAMETER_CHECK("maxSegmentsPerCell", getMaxSegmentsPerCell());
  PARAMETER_CHECK("maxSynapsesPerSegment", getMaxSynapsesPerSegment());
  PARAMETER_CHECK("burnIn", getBurnIn());
  PARAMETER_CHECK("collectStats", getCollectStats());
  PARAMETER_CHECK("seed", getSeed());
  return result;
}

bool BacktrackingTMCpp::sameSegment(
    const struct BacktrackingTMCpp::SegOnCellInfo_t &seg1,
    const struct BacktrackingTMCpp::SegOnCellInfo_t &seg2, std::ostream &out,
    Int32 verbosity) {
  // Return true if segVect1 and segVect2 are identical, ignoring order of
  // synapses
  bool result = true;

  // check sequence segment, total activations etc.
  // For floats, check that they are within 0.001.
  if (seg1.isSequenceSegment != seg2.isSequenceSegment) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].isSequenceSegment does not match: " << seg1.isSequenceSegment
          << ", " << seg2.isSequenceSegment << "\n";
    result = false;
  }
  if (seg1.positiveActivations != seg2.positiveActivations) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].positiveActivations does not match: "
          << seg1.positiveActivations << ", " << seg2.positiveActivations
          << "\n";
    result = false;
  }
  if (seg1.totalActivations != seg2.totalActivations) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].totalActivations does not match: " << seg1.totalActivations
          << ", " << seg2.totalActivations << "\n";
    result = false;
  }
  if (seg1.lastActiveIteration != seg2.lastActiveIteration) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].lastActiveIteration does not match: "
          << seg1.lastActiveIteration << ", " << seg2.lastActiveIteration
          << "\n";
    result = false;
  }
  if (abs(seg1.lastPosDutyCycle - seg2.lastPosDutyCycle) > 0.001) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].lastPosDutyCycle does not match: " << seg1.lastPosDutyCycle
          << ", " << seg2.lastPosDutyCycle << "\n";
    result = false;
  }
  if (seg1.lastPosDutyCycleIteration != seg2.lastPosDutyCycleIteration) {
    if (verbosity > 0)
      out << "Segment[" << seg1.c << "," << seg1.i
          << "].lastPosDutyCycleIteration does not match: "
          << seg1.lastPosDutyCycleIteration << ", "
          << seg2.lastPosDutyCycleIteration << "\n";
    result = false;
  }

  // Compare number of synapses
  if (seg1.synapses.size() != seg2.synapses.size()) {
    if (verbosity > 0)
      out << "Number of synapses does not match: " << seg1.synapses.size()
          << ", " << seg2.synapses.size() << "\n";
    result = false;
  }
  // Now compare synapses, ignoring order of synapses
  for (Size synIdx2 = 0; synIdx2 < seg2.synapses.size(); synIdx2++) {
    Real permanence = std::get<2>(seg2.synapses[synIdx2]);
    if (permanence <= 0.0f) {
      if (verbosity > 0) {
        Size c = std::get<0>(seg2.synapses[synIdx2]);
        Size i = std::get<1>(seg2.synapses[synIdx2]);
        out << "A synapse in TM2 Segment[" << seg2.c << "," << seg2.i << "] "
            << "with zero permanence encountered. [" << c << "," << i << "]\n";
      }
      result = false;
    }
    for (Size synIdx1 = 0; synIdx1 < seg1.synapses.size(); synIdx1++) {
      Size c1 = std::get<0>(seg1.synapses[synIdx1]);
      Size i1 = std::get<1>(seg1.synapses[synIdx1]);
      Real permanence1 = std::get<2>(seg1.synapses[synIdx1]);
      if (permanence1 <= 0.0f) {
        if (verbosity > 0) {
          out << "A synapse in TM2 Segment[" << seg2.c << "," << seg2.i << "] "
              << "with zero permanence encountered. [" << c1 << "," << i1
              << "]\n";
        }
        result = false;
      }

      bool match = false;
      for (Size synIdx2 = 1; synIdx2 < seg2.synapses.size(); synIdx2++) {
        Size c2 = std::get<0>(seg2.synapses[synIdx2]);
        Size i2 = std::get<1>(seg2.synapses[synIdx2]);
        Real permanence2 = std::get<2>(seg2.synapses[synIdx2]);
        if (c1 == c2 && i1 == i2) {
          match = true;
          if (abs(permanence1 - permanence2) > 0.001f) {
            if (verbosity > 0)
              out << "A synapse permanence does not match for Cell[" << c2
                  << "," << i2 << "] " << permanence1 << ", " << permanence2
                  << "\n";
            result = false;
          }
          break;
        }
      }
      if (!match) {
        if (verbosity > 0)
          out << "A synapse in TM1 Segment[" << seg1.c << "," << seg1.i << "]"
              << " did not match with a synapse in TM2. synapse[" << c1 << ","
              << i1 << "] ";
        result = false;
      }
    }
  }
  return result;
}

bool BacktrackingTMCpp::tmDiff2(const BacktrackingTMCpp &tm1,
                                const BacktrackingTMCpp &tm2, std::ostream &out,
                                Int32 verbosity, bool relaxSegmentTests,
                                bool checkLearn, bool checkStates) {
  bool result = true;

  // First check basic parameters. If we fail here, don't continue
  if (!sameTMParams(tm1, tm2)) {
    if (verbosity > 0) out << "Two TM's have different parameters\n";
    return false;
  }
  if (checkStates) {
    // Compare states at T first, they usually diverge before the structure of
    // the cells starts diverging  (infActiveStateT)
    if (memcmp(tm1.getActiveState(), tm2.getActiveState(), tm1.getNumCells())) {
      if (verbosity > 0) out << "Active states diverged (infActiveStateT)\n";
      return false;
    }
    if (memcmp(tm1.getPredictedState(), tm2.getPredictedState(), 
			    tm1.getNumCells())) {
      if (verbosity > 0) out << "Predicted states diverged (infPredictedStateT)\n";
      return false;
    }

    if (checkLearn) {
      if (memcmp(tm1.getLearnActiveStateT(), tm2.getLearnActiveStateT(),
			     tm1.getNumCells())) {
        if (verbosity > 0) out << "Learn Active states diverged (lrnActiveStateT)\n";
        return false;
      }
      if (memcmp(tm1.getLearnPredictedStateT(), tm2.getLearnPredictedStateT(), 
			      tm1.getNumCells())) {
        if (verbosity > 0) out << "Learn Predicted states diverged (lrnPredictedStateT)\n";
        return false;
      }
      Real32 rstate1 = tm1.getAvgLearnedSeqLength();
      Real32 rstate2 = tm2.getAvgLearnedSeqLength();
      if (abs(rstate1 - rstate2) > 0.01f) {
        if (verbosity > 0)
          out << "Average learned sequence lengths differ: " << rstate1 << " "
              << rstate2 << "\n";
        return false;
      }
    }
    if (!result && verbosity > 1) {
      out << "TM1: \n";
      tm1.printStates(false, true, out);
      out << "TM2: \n";
      tm2.printStates(false, true, out);
    }
  }
  // Now check some high level learned parameters
  if (tm1.getNumSegments() != tm2.getNumSegments()) {
    if (verbosity > 0)
      out << "Number of segments are different: " << tm1.getNumSegments()
          << ", " << tm2.getNumSegments();
    return false;
  }

  // Check that each cell has the same number of synapses
  if (tm1.getNumSynapses() != tm2.getNumSynapses()) {
    if (verbosity > 0) {
      out << "Number of synapses are different: " << tm1.getNumSynapses()
          << ", " << tm2.getNumSynapses();
    }
    if (verbosity >= 3) {
      out << "TM1: ";
      tm1.printCells(false, out);
      out << "TM2: ";
      tm2.printCells(false, out);
    }
    return false;
  }

  // Check that each cell has the same number of segments
  for (Size c = 0; c < tm1.getnumCol(); c++) {
    for (Size i = 0; i < tm1.getcellsPerCol(); i++) {
      if (tm1.getNumSegmentsInCell(c, i) != tm2.getNumSegmentsInCell(c, i)) {
        out << "Num segments different in cell: [" << c << "," << i << "] "
            << tm1.getNumSegmentsInCell(c, i)
            << " != " << tm2.getNumSegmentsInCell(c, i) << std::endl;
        return false;
      }
    }
  }

  // If the above tests pass, then check each segment and report differences
  // Note that segments in tm1 can be in a different order than tm2. Here we
  // make sure that, for each segment in tm1, there is an identical segment in
  // tm2.
  if (!relaxSegmentTests && checkLearn) {
    for (Size c = 0; c < tm1.getnumCol(); c++) {
      for (Size i = 0; i < tm1.getcellsPerCol(); i++) {
        UInt32 nSegs = tm1.getNumSegmentsInCell(c, i);
        for (Size segIdx = 0; segIdx < nSegs; segIdx++) {
          auto tm1seg = tm1.getSegmentOnCell(c, i, segIdx);

          // Loop through all segments in tm2seg and see if any of them match
          // tm1seg
          bool res = false;
          for (Size tm2segIdx = 0; tm2segIdx < nSegs; tm2segIdx++) {
            BacktrackingTMCpp::SegOnCellInfo_t tm2seg;
            tm2seg = tm2.getSegmentOnCell(c, i, tm2segIdx);
            if (sameSegment(tm1seg, tm2seg) == true) {
              res = true;
              break;
            }
          }
          if (!res) {
            if (verbosity >= 0) {
              out << "\nSegments are different for cell: [" << c << ", " << i
                  << "]\n";
              out << "TM1: ";
              tm1.printCell(c, i, false, out);
              out << "TM2: ";
              tm2.printCell(c, i, false, out);
            }
	    return false;
          }
        }
      }
    }
  }
  if (verbosity > 1) out << "TM's match" << std::endl;
  return true;
}

// identify the differences between this and another TM.
// compare parameters and everything that is serializeable.
// What does it mean for two tm's to be the same?
// If the two TM's were run in parallel they will have different
// results because the random number generator gives different numbers.
// The only way to get the same numbers is the following:
//    1) Run the first TM to some point.
//    2) save the TM.
//    3) Run the first TM to another point.
//    4) Restore the saved TM.  The restore should restore the Random generator.
//    5) Run the second TM to the same point using the same data.
// Now you can use this function to compare the two tm's.
// The two will be the same.

bool BacktrackingTMCpp::diff(const BacktrackingTMCpp &tm1,
                             const BacktrackingTMCpp &tm2) const {
  std::stringstream ss1;
  std::stringstream ss2;
  ss1.clear();
  ss2.clear();
  tm1.save(ss1);
  tm2.save(ss2);
  bool same = (ss1.str() == ss2.str());
  return same;
}

/////////////////////////////////////////////////////////////

// A segment is active if it has >= activationThreshold connected
// synapses that are active due to infActiveState. timestep is "t" or "t-1".
bool BacktrackingTMCpp::_slowIsSegmentActive(Segment &seg, std::string timestep) const {
  Size numActiveSyns = 0;
  NTA_ASSERT(timestep == "t" || timestep == "t-1") << "Only t, t-1 timesteps expected!";
  const UInt32 threshold = cells4_->getActivationThreshold();
  for (UInt synIdx = 0; synIdx < (UInt)seg.size(); synIdx++) {
    if (seg.getPermanence(synIdx) >= cells4_->getPermConnected()) {
      const Size srcIdx = seg.getSrcCellIdx(synIdx);
      Byte *state = (timestep == "t") ? cells4_->getInfActiveStateT()
                                      : cells4_->getInfActiveStateT1();
      if (state[srcIdx]) {
        numActiveSyns += 1;
        if (numActiveSyns >= threshold)
          return true;
      }
    }
  }
  return false;
}

/**
  struct SegOnCellInfo_t {
      Size c;
      Size i;
      Size segIdx;
      bool isSequenceSegment;
      Size positiveActivations;
      Size totalActivations;
      Size lastActiveIteration;
      Real lastPosDutyCycle;
      Size lastPosDutyCycleIteration;
      std::vector <std::tuple<Size,Size,Real>> synapses;
  };
**/

struct BacktrackingTMCpp::SegOnCellInfo_t
BacktrackingTMCpp::getSegmentOnCell(Size c, Size i, Size segIdx) const {
  std::vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
  Segment &seg = cells4_->getSegment((UInt)c, (UInt)i, segList[segIdx]);
  Size numSyn = seg.size();
  NTA_ASSERT(numSyn != 0);

  // segment info
  struct BacktrackingTMCpp::SegOnCellInfo_t info;
  info.c = c;
  info.i = i;
  info.segIdx = segIdx;
  info.isSequenceSegment = seg.isSequenceSegment();
  info.positiveActivations = seg.getPositiveActivations();
  info.totalActivations = seg.getTotalActivations();
  info.lastActiveIteration = seg.getLastActiveIteration();
  info.lastPosDutyCycle = seg.getLastPosDutyCycle();
  info.lastPosDutyCycleIteration = seg.getLastPosDutyCycleIteration();

  // synapse info
  for (Size s = 0; s < numSyn; numSyn++) {
    UInt idx = seg.getSrcCellIdx((UInt)s);
    Size c = idx / loc_.cellsPerColumn;
    Size i = idx % loc_.cellsPerColumn;
    Real permanence = seg.getPermanence((UInt)s);
    std::tuple<Size, Size, Real> syn = make_tuple(c, i, permanence);
    info.synapses.push_back(syn);
  }
  return info;
}

struct BacktrackingTMCpp::seginfo_t
BacktrackingTMCpp::getSegmentInfo(bool collectActiveData) const {
  NTA_ASSERT(collectActiveData == false) << " Requires appropriate accessors "
                                            "in C++ cells4 (currently "
                                            "unimplemented)";

  struct BacktrackingTMCpp::seginfo_t info;
  info.nSegments = getNumSegments();
  info.nSynapses = getNumSynapses();
  info.nActiveSegs = 0;
  info.nActiveSynapses = 0;
  info.distSegSizes.clear();
  info.distNSegsPerCell.clear();
  info.distPermValues.clear(); // Num synapses with given permanence values
  info.distAges.clear();

  // initialize the histogram buckets
  Size numAgeBuckets = 20;
  Size ageBucketSize = (loc_.iterationIdx + numAgeBuckets) / numAgeBuckets;
  for (Size i = 0; i < numAgeBuckets; i++) {
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%2zd-%-2zd", i * ageBucketSize,
             (i + 1) * ageBucketSize - 1);
    struct ageinfo_t bucket;
    bucket.range = buffer;
    bucket.cnt = 0;
    info.distAges.push_back(bucket);
  }

  for (Size c = 0; c < loc_.numberOfCols; c++) {
    for (Size i = 0; i < loc_.cellsPerColumn; i++) {
      Size nSegmentsThisCell = getNumSegmentsInCell(c, i);
      if (nSegmentsThisCell > 0) {
        // Update histogram counting cell sizes
        if (info.distNSegsPerCell.find(nSegmentsThisCell) !=
            info.distNSegsPerCell.end())
          info.distNSegsPerCell[nSegmentsThisCell] += 1;
        else
          info.distNSegsPerCell[nSegmentsThisCell] = 1;

        // Update histogram counting segment sizes.
        vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
        for (Size segIdx = 0; segIdx < nSegmentsThisCell; segIdx++) {
          struct SegOnCellInfo_t segcellinfo = getSegmentOnCell(c, i, segIdx);
          Size nSynapsesThisSeg = segcellinfo.synapses.size();
          if (nSynapsesThisSeg > 0) {
            if (info.distSegSizes.find(nSynapsesThisSeg) !=
                info.distSegSizes.end())
              info.distSegSizes[nSynapsesThisSeg] += 1;
            else
              info.distSegSizes[nSynapsesThisSeg] = 1;

            // Accumulate permanence value histogram (scaled by 10)
            for (Size synIdx = 1; synIdx < nSynapsesThisSeg; synIdx++) {
              Size permanence =
                  (Size)(std::get<2>(segcellinfo.synapses[synIdx]) * 10.0);
              if (info.distPermValues.find(permanence) !=
                  info.distPermValues.end())
                info.distPermValues[permanence] += 1;
              else
                info.distPermValues[permanence] = 1;
            }
          }
          const Segment &segObj = cells4_->getSegment((UInt)c, (UInt)i, segList[segIdx]);
          const Size age = loc_.iterationIdx - segObj.getLastActiveIteration();
          const Size ageBucket = age / ageBucketSize;
          info.distAges[ageBucket].cnt += 1;
        }
      }
    }
  }
  return info;
}

//////////////////  Serialization ///////////////////////
void BacktrackingTMCpp::saveToFile(std::string filePath) {
  std::ofstream out(filePath.c_str(),
                    std::ios_base::out | std::ios_base::binary);
  out.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  out.precision(std::numeric_limits<double>::digits10 + 1);
  out.precision(std::numeric_limits<float>::digits10 + 1);
  save(out);
  out.close();
}
void BacktrackingTMCpp::loadFromFile(std::string filePath) {
  std::ifstream in(filePath.c_str(), std::ios_base::in | std::ios_base::binary);
  load(in);
  in.close();
}

// Note: Most parts are saved in binary format. The file
//       into which this is written must be opened in binary mode.
//       This also means that it must be restored on a machine
//       of the same architecture and compiled with the same bitness.
void BacktrackingTMCpp::save(std::ostream &out) const {
  cells4_->save(out);
  out << "BacktrackingTMCpp " << TM_VERSION << " ";

  out << "loc " << sizeof(loc_) << " ";
  out.write((const char *)&loc_, sizeof(loc_));
  out << std::endl;
  out << outputType_ << std::endl;

  out << "CurrentOutput " << nCells << "[";
  out.write((const char*)currentOutput_, nCells * sizeof(UInt));
  out << "]" << std::endl;
}

// if caller wants to own the buffers, it must
// set the buffers before loading so they get
// filled with the restored data.
void BacktrackingTMCpp::load(std::istream &in) {
  std::string tag;
  UInt version;
  Size len;

  cells4_ = new Cells4::Cells4();
  cells4_->load(in);

  loc_.numberOfCols = cells4_->nColumns();
  loc_.cellsPerColumn = cells4_->nCellsPerCol();

  // Fields that this class needed to serialize
  in >> tag;
  NTA_ASSERT(tag == "BacktrackingTMCpp");
  in >> version;
  NTA_ASSERT(version >= 3);

  // loc_ variables saved as binary
  in >> tag;
  NTA_ASSERT(tag == "loc");
  in >> len;
  NTA_ASSERT(len == sizeof(loc_));
  in.ignore(1);
  in.read((char *)&loc_, len);
  in >> outputType_;

  NTA_ASSERT(loc_.numberOfCols == cells4_->nColumns());
  NTA_ASSERT(loc_.cellsPerColumn == cells4_->nCellsPerCol());
  nCells = cells4_->nCells();

  // restore the currentOutput_
  if (currentOutputOwn_ && currentOutput_ == nullptr)
      currentOutput_ = new Real[nCells];
  in >> tag;
  NTA_ASSERT(tag == "CurrentOutput");
  in >> len;
  NTA_ASSERT(len == nCells);
  in.ignore(1); // '['
  in.read((char *)currentOutput_, len * sizeof(Real));
  in >> tag;
  NTA_ASSERT(tag == "]");
  in.ignore(1);


  // initialize stats
  internalStats_["nPredictions"] = 0.0;
  internalStats_["totalMissing"] = 0.0;
  internalStats_["totalExtra"] = 0.0;
  internalStats_["pctExtraTotal"] = 0.0;
  internalStats_["pctMissingTotal"] = 0.0;
  internalStats_["predictionScoreTotal2"] = 0.0;
  internalStats_["falseNegativeScoreTotal"] = 0.0;
  internalStats_["falsePositiveScoreTotal"] = 0.0;

  confHistogram_.reset(new Real[nCells], std::default_delete<Real[]>());
  memset(confHistogram_.get(), 0, nCells * sizeof(Real));
}

////////////////////////////////////////////////////////////////
