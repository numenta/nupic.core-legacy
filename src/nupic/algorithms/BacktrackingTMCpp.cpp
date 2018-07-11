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
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <nupic/algorithms/BacktrackingTMCpp.hpp>
#include <nupic/algorithms/Segment.hpp>
template <typename T> static UInt32 *nonzero(const T *dence_buffer, Size len);

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::backtracking_tm;
using namespace nupic::algorithms::Cells4;

/////////////////////////////////////////////////////////////////
//  Static function nonzero()
// returns an array of the indexes of the non-zero elements.
// The first element is the number of elements in the rest of the array.
// Dynamically allocated, caller must delete.
template <typename T> static UInt32 *nonzero(const T *dence_buffer, Size len) {
  UInt32 *nz = new UInt32[len + 1];
  nz[0] = 1;
  for (Size n = 0; n < len; n++) {
    if (dence_buffer[n] != (T)0)
      nz[nz[0]++] = (UInt32)n;
  }
  return nz;
}

static const UInt TM_VERSION = 2;

BacktrackingTMCpp::BacktrackingTMCpp() {}

BacktrackingTMCpp::BacktrackingTMCpp(
    UInt32 numberOfCols, UInt32 cellsPerColumn, Real32 initialPerm,
    Real32 connectedPerm, UInt32 minThreshold, UInt32 newSynapseCount,
    Real32 permanenceInc, Real32 permanenceDec, Real32 permanenceMax,
    Real32 globalDecay, UInt32 activationThreshold, bool doPooling,
    UInt32 segUpdateValidDuration, UInt32 burnIn, bool collectStats, Int32 seed,
    Int32 verbosity, bool checkSynapseConsistency, UInt32 pamLength,
    UInt32 maxInfBacktrack, UInt32 maxLrnBacktrack, UInt32 maxAge,
    UInt32 maxSeqLength, Int32 maxSegmentsPerCell, Int32 maxSynapsesPerSegment,
    const char *outputType) {
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

  NTA_ASSERT(outputType == "normal" || outputType == "activeState" ||
             outputType == "activeState1CellPerCol");

  // Store creation parameters
  param_.numberOfCols = numberOfCols;
  param_.cellsPerColumn = cellsPerColumn;
  param_.initialPerm = initialPerm;
  param_.connectedPerm = connectedPerm;
  param_.minThreshold = minThreshold;
  param_.newSynapseCount = newSynapseCount;
  param_.permanenceInc = permanenceInc;
  param_.permanenceDec = permanenceDec;
  param_.permanenceMax = permanenceMax;
  param_.globalDecay = globalDecay;
  param_.activationThreshold = activationThreshold;
  param_.doPooling = doPooling;
  param_.segUpdateValidDuration = (doPooling) ? segUpdateValidDuration : 1;

  param_.burnIn = burnIn;
  param_.collectStats = collectStats;
  param_.seed = seed;
  param_.verbosity = verbosity;
  param_.checkSynapseConsistency = checkSynapseConsistency;
  param_.pamLength = pamLength;
  param_.maxInfBacktrack = maxInfBacktrack;
  param_.maxLrnBacktrack = maxLrnBacktrack;
  param_.maxAge = maxAge;
  param_.maxSeqLength = maxSeqLength;
  param_.maxSegmentsPerCell = maxSegmentsPerCell;
  param_.maxSynapsesPerSegment = maxSynapsesPerSegment;
  memset(param_.outputType, sizeof(param_.outputType), '0');
  strcpy_s(param_.outputType, sizeof(param_.outputType), outputType);

  // Initialize local state data
  loc_.numberOfCells = numberOfCols * cellsPerColumn;
  loc_.lrnIterationIdx = 0;
  loc_.iterationIdx = 0;
  // unique segment id, so we can put segments in hashes
  loc_.segID = 0;

  // pamCounter gets reset to pamLength whenever we detect that the learning
  // state is making good predictions (at least half the columns predicted).
  // Whenever we do not make a good prediction, we decrement pamCounter.
  // When pamCounter reaches 0, we start the learn state over again at start
  // cells.
  loc_.pamCounter = param_.pamLength;

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

  loc_.makeCells4Ephemeral = false;

  _initEphemerals();
}

// These are the members that do not need to be saved during serialization.
// However, to get a better resolution on the restore point we are serializing
// these.
void BacktrackingTMCpp::_initEphemerals() {
  loc_.allocateStatesInCPP = true;
  loc_.retrieveLearningStates = false;
  nCells = param_.numberOfCols * param_.cellsPerColumn;

  cells4_ = new Cells4::Cells4(
      param_.numberOfCols, param_.cellsPerColumn, param_.activationThreshold,
      param_.minThreshold, param_.newSynapseCount,
      param_.segUpdateValidDuration, param_.initialPerm, param_.connectedPerm,
      param_.permanenceMax, param_.permanenceDec, param_.permanenceInc,
      param_.globalDecay, param_.doPooling, param_.seed,
      loc_.allocateStatesInCPP, param_.checkSynapseConsistency);

  cells4_->setVerbosity(param_.verbosity);
  cells4_->setPamLength(param_.pamLength);
  cells4_->setMaxAge(param_.maxAge);
  cells4_->setMaxInfBacktrack(param_.maxInfBacktrack);
  cells4_->setMaxLrnBacktrack(param_.maxLrnBacktrack);
  cells4_->setMaxSeqLength(param_.maxSeqLength);
  cells4_->setMaxSegmentsPerCell(param_.maxSegmentsPerCell);
  cells4_->setMaxSynapsesPerCell(param_.maxSynapsesPerSegment);

  // get the buffers
  Byte *activeT;
  Byte *activeT1;
  Byte *predT;
  Byte *predT1;
  Real *colConfidenceT;
  Real *colConfidenceT1;
  Real *confidenceT;
  Real *confidenceT1;

  cells4_->getStatePointers(activeT, activeT1, predT, predT1, colConfidenceT,
                            colConfidenceT1, confidenceT, confidenceT1);
  infActiveState_["t"] = activeT;
  infActiveState_["t-1"] = activeT1;
  infPredictedState_["t"] = predT;
  infPredictedState_["t-1"] = predT1;
  cellConfidence_["t"] = colConfidenceT;
  colConfidence_["t-1"] = colConfidenceT1;
  cellConfidence_["t"] = confidenceT;
  cellConfidence_["t-1"] = confidenceT1;

  // note: setOutputBuffer() may override this buffer.
  currentOutput_.reset(new Real[nCells], std::default_delete<Real[]>());
  memset(currentOutput_.get(), 0, nCells * sizeof(Real));

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

// For backtrackingTMCpp we can let Cells4 allocate the buffers.
// However, the interfaces might want to take control of the buffers.
// This method can be called after creating the region to set the buffers.
void BacktrackingTMCpp::setStatePointers(Byte *infActiveT, Byte *infActiveT1,
                                         Byte *infPredT, Byte *infPredT1,
                                         Real *colConfidenceT,
                                         Real *colConfidenceT1,
                                         Real *cellConfidenceT,
                                         Real *cellConfidenceT1) {
  loc_.allocateStatesInCPP = false;
  infActiveState_["t"] = infActiveT;
  infActiveState_["t-1"] = infActiveT1;
  infPredictedState_["t"] = infPredT;
  infPredictedState_["t-1"] = infPredT1;
  cellConfidence_["t"] = colConfidenceT;
  colConfidence_["t-1"] = colConfidenceT1;
  cellConfidence_["t"] = cellConfidenceT;
  cellConfidence_["t-1"] = cellConfidenceT1;

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
  loc_.iterationIdx++;

  //  Run compute and retrieve selected state and member variables
  cells4_->compute(bottomUpInput, currentOutput_.get(), enableInference,
                   enableLearn);
  loc_.avgLearnedSeqLength = cells4_->getAvgLearnedSeqLength();

  // Get learn states if we need to print them out
  if (param_.verbosity > 1 || loc_.retrieveLearningStates) {
    Byte *activeT;
    Byte *activeT1;
    Byte *predT;
    Byte *predT1;
    cells4_->getLearnStatePointers(activeT, activeT1, predT, predT1);

    lrnActiveState_["t-1"] = activeT1;
    lrnActiveState_["t"] = activeT;
    lrnPredictedState_["t-1"] = predT1;
    lrnPredictedState_["t"] = predT;
  }

  Byte *predictedState;
  if (param_.collectStats) {
    UInt32 *activeColumns =
        nonzero<Real>(bottomUpInput, (Size)param_.numberOfCols);
    predictedState = (enableInference) ? infPredictedState_["t-1"]
                                       : lrnPredictedState_["t-1"];
    _updateStatsInferEnd(internalStats_, activeColumns, predictedState,
                         colConfidence_["t-1"]);
    delete activeColumns;
  }

  _computeOutput(); // note: modifies currentOutput_

  printComputeEnd(currentOutput_.get(), enableLearn);
  loc_.resetCalled = false;
  return currentOutput_.get();
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
  Size nCells = param_.numberOfCols * param_.cellsPerColumn;
  Size nCols = param_.numberOfCols;

  tmSavedState_t pristineTPDynamicState;

  // Save the TM dynamic state, we will use to revert back in the end
  _getTPDynamicState(&pristineTPDynamicState);

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

    // Copy t - 1 into t
    fastbuffercopy<Byte>(infActiveState_["t-1"], infActiveState_["t"], nCells);
    fastbuffercopy<Byte>(infPredictedState_["t-1"], infPredictedState_["t"],
                         nCells);
    fastbuffercopy<Real>(cellConfidence_["t-1"], cellConfidence_["t"], nCells);

    // Predicted state at "t-1" becomes the active state at "t"
    fastbuffercopy<Byte>(infActiveState_["t"], infPredictedState_["t-1"],
                         nCells);

    // Predicted state and confidence are set in phase2.
    memset(infPredictedState_["t"], 0, nCells);
    memset(cellConfidence_["t"], 0, nCells * sizeof(Real));
    _inferPhase2();
  }

  // Revert the dynamic state to the saved state
  _setTPDynamicState(&pristineTPDynamicState);

  return multiStepColumnPredictions;
}

Real *BacktrackingTMCpp::topDownCompute() {
  // For now,  we will assume there is no one above us and that bottomUpOut
  // is simply the output that corresponds to our currently stored column
  // confidences. Simply return the column confidences
  return colConfidence_["t"];
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
  if (param_.verbosity >= 5) {
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
  if (param_.collectStats) {
    internalStats["nInfersSinceReset"] += 1;

    // Compute the prediction score, how well the prediction from the last
    // time step predicted the current bottom-up input  //line 945 of
    // backtracking_tm.py
    std::vector<const UInt32 *> patternNZs;
    patternNZs.push_back(bottomUpNZ);
    Size numExtra2;
    Size numMissing2;
    std::vector<struct score_tuple> confidences2;
    _checkPrediction(patternNZs, predictedState, colConfidence, false,
                     numExtra2, numMissing2, confidences2, nullptr);

    // Store the stats that don't depend on burn-in
    internalStats["curPredictionScore2"] = confidences2[0].predictionScore;
    internalStats["curFalseNegativeScore"] =  1.0f - confidences2[0].posPredictionScore;
    internalStats["curFalsePositiveScore"] = confidences2[0].negPredictionScore;
    internalStats["curMissing"] = (Real)numMissing2;
    internalStats["curExtra"] = (Real)numExtra2;

    // If we are passed the burn-in period, update the accumulated stats
    // Here's what various burn-in values mean:
    //   0 : try to predict the first element of each sequence and all
    //   subsequent 1 : try to predict the second element of each sequence and
    //   all subsequent
    //    etc.
    if (internalStats["nInfersSinceReset"] > param_.burnIn) {

      // Burn - in related stats
      Real numExpected =
          std::max<Real>(1.0, (Real)bottomUpNZ[0]); // first element is length.
      internalStats["nPredictions"] += 1.0f;
      internalStats["totalMissing"] += numMissing2;
      internalStats["totalExtra"] += numExtra2;
      internalStats["pctExtraTotal"] += 100.0f * numExtra2 / numExpected;
      internalStats["pctMissingTotal"] += 100.0f * numMissing2 / numExpected;
      internalStats["predictionScoreTotal2"] += confidences2[0].predictionScore;
      internalStats["falseNegativeScoreTotal"] += 1.0f - confidences2[0].posPredictionScore;
      internalStats["falsePositiveScoreTotal"] += confidences2[0].negPredictionScore;

      if (loc_.collectSequenceStats) {
        // Collect cell confidences for every cell that correctly predicted
        // current bottom up input. Normalize confidence across each column
        Real *cc = new Real[nCells];
        Real *cellT1 = cellConfidence_["t-1"];
        Byte *cellState = infActiveState_["t"];
        for (Size i = 0; i < (Size)param_.numberOfCols; i++) {
          Real sconf = 0;
          for (Size j = 0; j < (Size)param_.cellsPerColumn; j++) {
            // zero out confidence if state bit is zero
            cc[i * j] = cellT1[i * j] * cellState[i * j];
            // add up confidence of a column
            sconf += cc[i * j];
          }
          if (sconf > 0) {
            // Normalize the confidence for each cell in the column
            for (Size j = 0; j < (Size)param_.cellsPerColumn; j++) {
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
 *   @predicted The output of the TM (predictedState).
 *   @colConfidence The column confidences. If not specified, then use the
 *              TM's current colConfidence_. This can be specified
 *              if you are trying to check the prediction metrics
 *              for an output from the past.
 *   @details   if True, also include details of missing bits per pattern.
 *
 *   :returns:
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
void BacktrackingTMCpp::_checkPrediction(
    std::vector<const UInt32 *> patternNZs, const Byte *predicted,
    const Real *colConfidence, bool details, Size &totalExtras,
    Size &totalMissing, std::vector<struct score_tuple> &conf, Real *missing) {
  if (details) {
    NTA_ASSERT(missing != nullptr);
  }
  // Compute the union of all the expected patterns
  std::set<UInt32> orAll;
  for (size_t i = 0; i < patternNZs.size(); i++) {
    for (size_t n = 1; n < patternNZs[i][0]; n++) {
      orAll.insert(patternNZs[i][n]);
    }
  }
  // Get the list of active columns in the output
  std::set<UInt32> outputnz;
  for (Size i = 0; i < nCells; i++) {
    if (predicted[i] != 0)
      outputnz.insert((UInt32)i);
    if (details)
      missing[i] = 0;
  }

  // Compute the total extra and missing in the output
  totalExtras = 0;
  totalMissing = 0;
  std::set<UInt32>::iterator first1 = outputnz.begin();
  std::set<UInt32>::iterator last1 = outputnz.end();
  std::set<UInt32>::iterator first2 = orAll.begin();
  std::set<UInt32>::iterator last2 = orAll.end();
  while (true) {
    if (first1 == last1) {
      if (first2 == last2)
        break;
      totalMissing++; // it is in orAll but not in outputnz.
      if (details)
        missing[*first2] = 1;
      ++first2;
    } else if (first2 == last2) {
      totalExtras++; // it is in outputnz but not in orAll.
      first1++;
    } else {
      if (*first1 < *first2) {
        totalExtras++; // it is in outputnz but not in orAll.
        ++first1;
      } else if (*first2 < *first1) {
        totalMissing++; // it is in orAll but not in outputnz.
        if (details)
          missing[*first2] = 1;
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
    colConfidence = colConfidence_["t"];
  for (Size p = 0; p < patternNZs.size(); p++) {
    const UInt32 *pattern = patternNZs[p];
    struct score_tuple scores;

    // Sum of the column confidences for this pattern?
    Size positiveColumnCount =
        pattern[0]; // first element is number of elements.
    Real positivePredictionSum = 0;
    for (Size i = 1; i < positiveColumnCount; i++) {
      positivePredictionSum += colConfidence[pattern[i]];
    }

    // Sum of all the column confidences
    Real totalPredictionSum = 0;
    for (Size i = 1; i < param_.numberOfCols; i++) {
      totalPredictionSum += colConfidence[i];
    }

    // Total number of columns
    Size totalColumnCount = param_.numberOfCols;

    Real negativePredictionSum = totalPredictionSum - positivePredictionSum;
    Size negativeColumnCount = totalColumnCount - positiveColumnCount;

    //  Compute the average confidence score per column for this pattern
    //  Compute the average confidence score per column for the other patterns
    scores.posPredictionScore = (positiveColumnCount == 0) ? 0.0f : positivePredictionSum;
    scores.negPredictionScore = (negativeColumnCount == 0) ? 0.0f : negativePredictionSum;

    // Scale the positive and negative prediction scores so that they sum to 1.0
    Real currentSum = scores.negPredictionScore + scores.posPredictionScore;
    if (currentSum > 0.0) {
      scores.posPredictionScore *= 1.0f / currentSum;
      scores.negPredictionScore *= 1.0f / currentSum;
    }
    scores.predictionScore =
        scores.posPredictionScore - scores.negPredictionScore;

    conf.push_back(scores);
  }
}

////////////////////////////////////
//  _computeOutput();
//    Computes output for both learning and inference. In both cases, the
//    output is the boolean OR of 'activeState' and 'predictedState' at 't'.
//    Stores 'currentOutput_'.
Real32 *BacktrackingTMCpp::_computeOutput() {
  auto currentOutput = currentOutput_.get();
  if (param_.outputType == "activeState1CellPerCol") {
    // Fire only the most confident cell in columns that have 2 or more active
    // cells Don't turn on anything in columns which are not active at all
    Byte *active = infActiveState_["t"];
    Real *cc = cellConfidence_["t"];
    for (Size i = 0; i < (Size)param_.numberOfCols; i++) {
      Size isColActive = 0;
      Size mostConfidentCell = 0;
      Real c = 0;
      for (Size j = 0; j < param_.cellsPerColumn; j++) {
        Size cellIdx = i * param_.cellsPerColumn + j;
        if (cc[cellIdx] > c) {
          c = cc[cellIdx];
          mostConfidentCell = cellIdx;
        }
        currentOutput[cellIdx] = 0; // zero the output
        isColActive += active[cellIdx];
      }
      if (c > 0 && isColActive) // set the most confident cell in this column
                                // if active.
        currentOutput[mostConfidentCell] = 1;
    }

  } else if (param_.outputType == "activeState") {
    Byte *active = infActiveState_["t"];
    for (Size i = 0; i < nCells; i++) {
      currentOutput[i] = active[i];
    }

  } else if (param_.outputType == "normal") {
    Byte *active = infActiveState_["t"];
    Byte *predicted = infPredictedState_["t"];
    for (Size i = 0; i < nCells; i++) {
      currentOutput[i] = (active[i] || predicted[i]) ? 1.0f : 0.0f;
    }

  } else {
    NTA_THROW << "Unimplemented outputType '" << param_.outputType << "' ";
  }
  return currentOutput;
}

void BacktrackingTMCpp::reset() {
  if (param_.verbosity >= 3)
    std::cout << "\n==== TM Reset =====" << std::endl;
  //_setStatePointers()
  cells4_->reset();

  // ---from line 769 of backtracking_tm.py
  // Reset the state of all cells.
  // This is normally used between sequences while training.
  // All internal states are reset to 0.
  memset(lrnActiveState_["t-1"], 0, nCells);
  memset(lrnActiveState_["t"], 0, nCells);
  memset(lrnPredictedState_["t-1"], 0, nCells);
  memset(lrnPredictedState_["t"], 0, nCells);
  memset(infActiveState_["t-1"], 0, nCells);
  memset(infActiveState_["t"], 0, nCells);
  memset(infPredictedState_["t-1"], 0, nCells);
  memset(infPredictedState_["t"], 0, nCells);
  memset(cellConfidence_["t-1"], 0, nCells * sizeof(Real));
  memset(cellConfidence_["t"], 0, nCells * sizeof(Real));


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
//          - ``nPredictions``: the number of predictions. This is the total
//                number of inferences excluding burn-in and the last inference.
//          - ``curPredictionScore``: the score for predicting the current input
//                (predicted during the previous inference)
//          - ``curMissing``: the number of bits in the current input that were not predicted to be on.
//          - ``curExtra``: the number of bits in the predicted output that are not in the next input
//          - ``predictionScoreTotal``: the sum of every prediction score to date
//          - ``predictionScoreAvg``: ``predictionScoreTotal / nPredictions``
//          - ``pctMissingTotal``: the total number of bits that were missed over all predictions
//          - ``pctMissingAvg``: ``pctMissingTotal / nPredictions``
//      The map is empty if ``collectSequenceStats``
//                is False.

std::map<std::string, Real32> BacktrackingTMCpp::getStats() {
  std::map<std::string, Real32> stats;
  if (!param_.collectStats) {
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
  stats["predictionScoreAvg2"] = internalStats_["predictionScoreTotal2"] / nPredictions;
  stats["curFalseNegativeScore"] = internalStats_["curFalseNegativeScore"];
  stats["falseNegativeAvg"] = internalStats_["falseNegativeScoreTotal"] / nPredictions;
  stats["curFalsePositiveScore"] = internalStats_["curFalsePositiveScore"];
  stats["falsePositiveAvg"] = internalStats_["falsePositiveScoreTotal"] / nPredictions;

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
void BacktrackingTMCpp::_getTPDynamicState(tmSavedState_t *ss) {

  deepcopySave_<Byte>(infActiveState_, ss->infActiveState_, nCells);
  deepcopySave_<Byte>(infPredictedState_, ss->infPredictedState_, nCells);
  deepcopySave_<Real>(cellConfidence_, ss->cellConfidence_, nCells);
  deepcopySave_<Real>(colConfidence_, ss->colConfidence_, param_.numberOfCols);
  deepcopySave_<Byte>(lrnActiveState_, ss->lrnActiveState_, nCells);
  deepcopySave_<Byte>(lrnPredictedState_, ss->lrnPredictedState_, nCells);
}

void BacktrackingTMCpp::_setTPDynamicState(tmSavedState_t *ss) {
  deepcopyRestore_<Byte>(infActiveState_, ss->infActiveState_, nCells);
  deepcopyRestore_<Byte>(infPredictedState_, ss->infPredictedState_, nCells);
  deepcopyRestore_<Real>(cellConfidence_, ss->cellConfidence_, nCells);
  deepcopyRestore_<Real>(colConfidence_, ss->colConfidence_,
                         param_.numberOfCols);
  deepcopyRestore_<Byte>(lrnActiveState_, ss->lrnActiveState_, nCells);
  deepcopyRestore_<Byte>(lrnPredictedState_, ss->lrnPredictedState_, nCells);
}

template <typename T>
void BacktrackingTMCpp::deepcopySave_(
    std::map<std::string, T*> &fromstate,
    std::map<std::string, std::shared_ptr<T>> &tostate, 
    Size count) 
{
  tostate.clear();
  typename std::map<std::string, T *>::iterator it;
  for (it = fromstate.begin(); it != fromstate.end(); it++) {
    std::shared_ptr<T> target(new T[count], std::default_delete<T[]>());
    tostate[it->first] = target;
    fastbuffercopy<T>(target.get(), it->second, count);
  }
}

template <typename T>
void BacktrackingTMCpp::deepcopyRestore_(
    std::map<std::string, T*> &tostate,
    std::map<std::string, std::shared_ptr<T>> &fromstate, Size count) 
{
  typename std::map<std::string, std::shared_ptr<T>>::iterator it;
  typename std::map<std::string, T *>::iterator target;
  for (it = fromstate.begin(); it != fromstate.end(); it++) {
    target = tostate.find(it->first);
    if (target != tostate.end()) {
      fastbuffercopy<T>(target->second, it->second.get(), count);
    }
  }
}

template <typename T>
void BacktrackingTMCpp::fastbuffercopy(T* tobuf, T* frombuf, Size count) 
{
  memcpy(tobuf, frombuf, count * sizeof(T));
}



/////////////// getter/setter methods //////////////////////
UInt32 BacktrackingTMCpp::getParameterUInt32(std::string name) {
  switch (name[0]) {
  case 'a':
    if (name == "activationThreshold") {
      return param_.activationThreshold;
    }
    break;
  case 'b':
    if (name == "burnIn")
      return param_.burnIn;
    break;

  case 'c':
    if (name == "cellsPerColumn")
      return param_.cellsPerColumn;
    if (name == "numberOfCols")
      return param_.numberOfCols;
    break;

  case 'm':
    if (name == "maxAge")
      return param_.maxAge;
    if (name == "maxInfBacktrack")
      return param_.maxInfBacktrack;
    if (name == "maxLrnBacktrack")
      return param_.maxLrnBacktrack;
    if (name == "minThreshold")
      return param_.minThreshold;
    if (name == "maxSeqLength")
      return param_.maxSeqLength;
    break;

  case 'n':
    if (name == "newSynapseCount")
      return param_.newSynapseCount;
    break;

  case 'o':
    if (name == "outputWidth")
      return (UInt32)nCells;

  case 'p':
    if (name == "pamLength")
      return param_.pamLength;
    break;

  case 's':
    if (name == "segUpdateValidDuration")
      return param_.segUpdateValidDuration;
    break;
  } // end switch
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

Int32 BacktrackingTMCpp::getParameterInt32(std::string name) {
  if (name == "maxSegmentsPerCell")
    return param_.maxSegmentsPerCell;
  if (name == "maxSynapsesPerSegment")
    return param_.maxSynapsesPerSegment;
  if (name == "seed") {
    return param_.seed;
    if (name == "verbosity")
      return param_.verbosity;
  }
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

Real32 BacktrackingTMCpp::getParameterReal32(std::string name) {
  switch (name[0]) {
  case 'c':
    if (name == "initialPerm")
      return param_.initialPerm;
    break;
  case 'g':
    if (name == "globalDecay")
      return param_.globalDecay;
    break;

  case 'i':
    if (name == "initialPerm")
      return param_.initialPerm;
    break;
  case 'p':
    if (name == "permanenceInc")
      return param_.permanenceInc;
    if (name == "permanenceDec")
      return param_.permanenceDec;
    if (name == "connectedPerm")
      return param_.connectedPerm;
    if (name == "permanenceMax")
      return param_.permanenceMax;
    break;
  }
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

bool BacktrackingTMCpp::getParameterBool(std::string name) {
  if (name == "collectStats")
    return param_.collectStats;
  if (name == "checkSynapseConsistency")
    return param_.checkSynapseConsistency;
  if (name == "doPooling")
    return param_.doPooling;
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

std::string BacktrackingTMCpp::getParameterString(std::string name) {
  if (name == "outputType") {
    return param_.outputType;
  }
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

void BacktrackingTMCpp::setParameter(std::string name, UInt32 value) {
  switch (name[0]) {
  case 'a':
    if (name == "activationThreshold") {
      param_.activationThreshold = value;
      return;
    }
    break;
  case 'b':
    if (name == "burnIn") {
      param_.burnIn = value;
      return;
    }
    break;
  case 'm':
    if (name == "minThreshold") {
      param_.minThreshold = value;
      return;
    }
    if (name == "maxInfBacktrack") {
      param_.maxInfBacktrack = value;
      return;
    }
    if (name == "maxLrnBacktrack") {
      param_.maxLrnBacktrack = value;
      return;
    }
    if (name == "maxAge") {
      param_.maxAge = value;
      return;
    }
    if (name == "maxSeqLength") {
      param_.maxSeqLength = value;
      return;
    }
    break;
  case 'n':
    if (name == "newSynapseCount") {
      param_.newSynapseCount = value;
      return;
    }
    break;
  case 'p':
    if (name == "pamLength") {
      param_.pamLength = value;
      return;
    }
    break;
  case 's':
    if (name == "segUpdateValidDuration") {
      param_.segUpdateValidDuration = value;
      return;
    }
    break;
  } // switch
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

void BacktrackingTMCpp::setParameter(std::string name, Int32 value) {
  if (name == "maxSegmentsPerCell") {
    param_.maxSegmentsPerCell = value;
    return;
  }
  if (name == "maxSynapsesPerSegment") {
    param_.maxSynapsesPerSegment = value;
    return;
  }
  if (name == "verbosity") {
    param_.verbosity = value;
    return;
  }
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

void BacktrackingTMCpp::setParameter(std::string name, Real32 value) {
  switch (name[0]) {
  case 'c':
    if (name == "connectedPerm") {
      param_.connectedPerm = value;
      return;
    }
    break;
  case 'g':
    if (name == "globalDecay") {
      param_.globalDecay = value;
      return;
    }
    break;
  case 'i':
    if (name == "initialPerm") {
      param_.initialPerm = value;
      return;
    }
    break;
  case 'p':
    if (name == "permanenceInc") {
      param_.permanenceInc = value;
      return;
    }
    if (name == "permanenceDec") {
      param_.permanenceDec = value;
      return;
    }
    if (name == "permanenceMax") {
      param_.permanenceMax = value;
      return;
    }
    break;
  } // switch
  NTA_THROW << "parameter name '" << name << "' unknown.";
}

void BacktrackingTMCpp::setParameter(std::string name, bool value) {
  if (name == "doPooling") {
    param_.doPooling = value;
    return;
  }
  if (name == "collectStats") {
    param_.collectStats = value;
    return;
  }

  if (name == "checkSynapseConsistency") {
    param_.checkSynapseConsistency = value;
    return;
  }

  NTA_THROW << "parameter name '" << name << "' unknown.";
}
void BacktrackingTMCpp::setParameter(std::string name, std::string val) {
  if (name == "outputType") {
    strncpy(param_.outputType, val.c_str(), sizeof(param_.outputType));
  } else
    NTA_THROW << "parameter name '" << name << "' unknown.";
}

///////////////  printing routines for Debug  ///////////////
void BacktrackingTMCpp::printComputeEnd(Real *output, bool learn) {
  // Called at the end of inference to print out various diagnostic information
  // based on the current verbosity level

  if (param_.verbosity >= 3) {
    std::cout << "----- computeEnd summary: \n";
    std::cout << "learn:" << learn << "\n";

    Size numBurstingCols = 0;
    Size numOn = 0;
    Byte *ptr = infActiveState_["t"];
    for (Size i = 0; i < param_.numberOfCols; i++) {
      bool isbursting = true;
      for (Size j = 0; j < param_.cellsPerColumn; j++) {
        if (ptr[i * param_.cellsPerColumn + j] == 0) {
          isbursting = false;
          break;
        } else {
          numOn++;
        }
      }
      if (isbursting)
        numBurstingCols++;
    }
    std::cout << "numBurstingCols:    " << numBurstingCols << "\n";
    std::cout << "curPredScore2:      " << internalStats_["curPredictionScore2"]
              << "\n";
    std::cout << "curFalsePosScore:   "
              << internalStats_["curFalsePositiveScore"] << "\n";
    std::cout << "1-curFalseNegScore: "
              << (1 - internalStats_["curFalseNegativeScore"]) << "\n";
    std::cout << "numSegments:        " << getNumSegments() << "\n";
    std::cout << "avgLearnedSeqLength:" << loc_.avgLearnedSeqLength << "\n";

    std::cout << "\n----- infActiveState (" << numOn << " on) ------\n";
    printActiveIndicesByte(infActiveState_["t"]);
    if (param_.verbosity >= 6)
      printState(infActiveState_["t"]);

    ptr = infPredictedState_["t"];
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    std::cout << "\n----- infPredictedState (" << numOn << " on)-----\n";
    printActiveIndicesByte(infPredictedState_["t"]);
    if (param_.verbosity >= 6)
      printState(infPredictedState_["t"]);

    ptr = lrnActiveState_["t"];
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    std::cout << "\n----- lrnActiveState (" << numOn << " on) ------\n";
    printActiveIndicesByte(lrnActiveState_["t"]);
    if (param_.verbosity >= 6)
      printState(lrnActiveState_["t"]);

    ptr = lrnPredictedState_["t"];
    numOn = 0;
    for (Size i = 0; i < nCells; i++) {
      if (ptr[i])
        numOn++;
    }
    std::cout << "\n----- lrnPredictedState (%d on)-----\n";
    printActiveIndicesByte(lrnPredictedState_["t"]);
    if (param_.verbosity >= 6)
      printState(lrnPredictedState_["t"]);

    std::cout << "\n----- cellConfidence -----\n";
    printActiveIndicesReal(cellConfidence_["t"], true);
    if (param_.verbosity >= 6)
      printConfidence(cellConfidence_["t"]);

    std::cout << "\n----- colConfidence -----\n";
    printColActiveIndices(colConfidence_["t"], true);

    std::cout
        << "\n----- cellConfidence[t-1] for currently active cells -----\n";
    ptr = infActiveState_["t"];
    Real *c = cellConfidence_["t-1"];
    Real *cc = new Real[nCells];
    for (Size i = 0; i < nCells; i++) {
      cc[i] = (ptr[i]) ? c[i] : 0;
    }
    printActiveIndicesReal(cc, true);
    delete[] cc;

    if (param_.verbosity >= 4) {
      std::cout << "\nCells, predicted segments only:\n";
      printCells(true);
    } else if (param_.verbosity >= 5) {
      std::cout << "\nCells, all segments:\n";
      printCells(false);
    }
    std::cout << std::endl; // flush buffer

  } else if (param_.verbosity >= 1) {
    std::cout << "\nTM: learn:" << learn << "\n";

    UInt32 *outputnz = nonzero<Real>(output, nCells);
    std::cout << "\nTM: active outputs(" << outputnz[0] << ")\n";
    printActiveIndicesReal(output);
    delete[] outputnz;
  }
}

// Print the list of '[column, cellIdx]' indices for each of the active cells in
// state.
void BacktrackingTMCpp::printActiveIndicesByte(const Byte *state, bool andValues) {
  for (Size i = 0; i < param_.numberOfCols; i++) {
    std::cout << "Col " << i << ": [";
    bool first = true;
    for (Size j = 0; j < param_.cellsPerColumn; j++) {
      if (state[i * param_.cellsPerColumn + j]) {
        if (!first)
          std::cout << ", ";
        if (andValues)
          std::cout << j << ": " << state[i * param_.cellsPerColumn + j];
        else
          std::cout << j;
        first = false;
      }
    }
    std::cout << "]\n";
  }
}
// Print the list of '[column, cellIdx]' indices for each of the active cells in
// confidence.
void BacktrackingTMCpp::printActiveIndicesReal(const Real *confidence, bool andValues) {
  for (Size i = 0; i < param_.numberOfCols; i++) {
    std::cout << "Col " << i << ": [";
    bool first = true;
    for (Size j = 0; j < param_.cellsPerColumn; j++) {
      if (confidence[i * param_.cellsPerColumn + j]) {
        if (!first)
          std::cout << ", ";
        if (andValues)
          std::cout << j << ": " << confidence[i * param_.cellsPerColumn + j];
        else
          std::cout << j;
        first = false;
      }
    }
    std::cout << "]\n";
  }
}
// Print the list of '[column]' indices for each of the active cells in
// ColConfidence.
void BacktrackingTMCpp::printColActiveIndices(const Real *colconfidence, bool andValues) {
  std::cout << "[";
  bool first = true;
  for (Size j = 0; j < param_.numberOfCols; j++) {
    if (colconfidence[j]) {
      if (!first)
        std::cout << ", ";
      if (andValues)
        std::cout << j << ": " << colconfidence[j];
      else
        std::cout << j;
      first = false;
    }
  }
  std::cout << "]\n";
}

// Print an integer array that is the same shape as activeState.
void BacktrackingTMCpp::printState(const Byte *aState) {
  for (Size i = 0; i < param_.cellsPerColumn; i++) {

    for (Size c = 0; c < param_.numberOfCols; c++) {
      if (c % 10 == 0)
        std::cout << " "; // add a spacer every 10
      std::cout << aState[c * param_.cellsPerColumn + i] << " ";
    }
    std::cout << "\n";
  }
}

// Print a floating point array that is the same shape as activeState.
void BacktrackingTMCpp::printConfidence(const Real *aState, Size maxCols) 
{
  char buf[20];
  for (Size i = 0; i < param_.cellsPerColumn; i++) {
    for (Size c = 0; c < std::min(maxCols, (Size)param_.numberOfCols); c++) {
      if (c % 10 == 0)
        std::cout << "  "; // add a spacer every 10
      sprintf(buf, " %5.3f", aState[c * param_.cellsPerColumn + i]);
      std::cout << buf;
    }
    std::cout << "\n";
  }
}

// Print up to maxCols number from a flat floating point array.
void BacktrackingTMCpp::printColConfidence(const Real *aState, Size maxCols) 
{
  char buf[20];
  for (Size c = 0; c < std::min(maxCols, (Size)param_.numberOfCols); c++) {
    if (c % 10 == 0)
      std::cout << "  "; // add a spacer every 10
    sprintf(buf, " %5.3f", aState[c]);
    std::cout << buf;
  }
  std::cout << "\n";
}

void BacktrackingTMCpp::printCells(bool predictedOnly) {
  if (predictedOnly)
    std::cout << "--- PREDICTED CELLS ---\n";
  else
    std::cout << "--- ALL CELLS ---\n";
  std::cout << "Activation threshold=" << param_.activationThreshold << "\n";
  std::cout << "min threshold=" << param_.minThreshold << "\n";
  std::cout << "connected perm=" << param_.connectedPerm << "\n";

  for (Size c = 0; c < param_.numberOfCols; c++) {
    for (Size i = 0; i < param_.cellsPerColumn; i++) {
      if (!predictedOnly ||
          infPredictedState_["t"][c * param_.cellsPerColumn + i])
        printCell(c, i, predictedOnly);
    }
  }
}

void BacktrackingTMCpp::printCell(Size c, Size i,  bool onlyActiveSegments) 
{
  char buff[1000];
  Size nSegs = (Size)cells4_->nSegmentsOnCell((UInt)c, (UInt)i);
  if (nSegs > 0) {
    vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
    std::cout << "Col " << c << ", Cell " << i << " ("
              << (c * param_.cellsPerColumn + i) << ") : " << nSegs
              << " segment(s)\n";
    std::vector<UInt32>::iterator it;
    for (it = segList.begin(); it != segList.end(); ++it) {
      Segment &seg = cells4_->getSegment((UInt)c, (UInt)i, *it);
      bool isActive = _slowIsSegmentActive(seg, "t");
      if (!onlyActiveSegments || isActive) {
        snprintf(buff, sizeof(buff), "%sSeg #%-3d %d %c %9.7f (%4d/%-4d) %4d ",
                 ((isActive) ? "*" : " "), *it, (int)seg.size(),
                 (seg.isSequenceSegment()) ? 'T' : 'F',
                 seg.dutyCycle(cells4_->getNLrnIterations(), false, true),
                 seg.getPositiveActivations(), seg.getTotalActivations(),
                 cells4_->getNLrnIterations() - seg.getLastActiveIteration());
        for (UInt idx = 0; idx < (UInt)seg.size(); idx++) {
          Size len = sizeof(buff) - strlen(buff);
          Size srcIdx = seg.getSrcCellIdx(idx);
          snprintf(buff + strlen(buff), len, "[%2d,%-2d]%4.2f ",
                   (int)_getCellCol(srcIdx), (int)_getCellIdx(srcIdx),
                   seg.getPermanence(idx));
        }
        std::cout << buff << "\n";
      }
    }
  }
}

void BacktrackingTMCpp::printInput(const Real32 *x) {
  std::cout << "Input\n";
  for (Size c = 0; c < param_.numberOfCols; c++) {
    std::cout << (int)x[c] << " ";
  }
  std::cout << std::endl;
}

void BacktrackingTMCpp::printOutput(const Real32 *y) {
  char buff[100];
  std::cout << "Output\n";
  for (Size i = 0; i < param_.cellsPerColumn; i++) {
    snprintf(buff, sizeof(buff), "[%3d] ", (int)i);
    std::cout << buff;
    for (Size c = 0; c < param_.numberOfCols; c++) {
      std::cout << (int)y[c * param_.numberOfCols + i] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

//     Print the parameter settings for the TM.
void BacktrackingTMCpp::printParameters() {
  std::cout << "numberOfCols=", param_.numberOfCols;
  std::cout << "cellsPerColumn=", param_.cellsPerColumn;
  std::cout << "minThreshold=", param_.minThreshold;
  std::cout << "newSynapseCount=", param_.newSynapseCount;
  std::cout << "activationThreshold=", param_.activationThreshold;
  std::cout << "\n";
  std::cout << "initialPerm=", param_.initialPerm;
  std::cout << "connectedPerm=", param_.connectedPerm;
  std::cout << "permanenceInc=", param_.permanenceInc;
  std::cout << "permanenceDec=", param_.permanenceDec;
  std::cout << "permanenceMax=", param_.permanenceMax;
  std::cout << "globalDecay=", param_.globalDecay;
  std::cout << "\n";
  std::cout << "doPooling=", param_.doPooling;
  std::cout << "segUpdateValidDuration=", param_.segUpdateValidDuration;
  std::cout << "pamLength=", param_.pamLength;
  std::cout << std::endl;
}
void BacktrackingTMCpp::printSegment(Segment &s) {
  s.print(std::cout, param_.cellsPerColumn);
}

//void BacktrackingTMCpp::printSegmentUpdates() {
//  std::map<Size, vector<Size>>::iterator it;
//  std::cout << "=== SEGMENT UPDATES ===, Num = " << segmentUpdates_.size()
//            << "\n";
//  for (it = segmentUpdates_.begin(); it != segmentUpdates_.end(); it++) {
//    Size c = it->first / param_.cellsPerColumn;
//    Size i = it->first % param_.cellsPerColumn;
//    vector<Size> &list = it->second;
//    std::cout << "[c=" << c << " i=" << i << "] updateList: ";
//    for (Size j = 0; j < list.size(); j++) {
//      std::cout << list[j] << " ";
//    }
//    std::cout << "\n";
//  }
//  std::cout << std::endl;
//}

static char *formatRow(char *buffer, Size bufsize, const Byte *val, Size i,
                       Size numberOfCols, Size cellsPerColumn) 
{
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
void BacktrackingTMCpp::printStates(bool printPrevious, bool printLearnState) 
{
  char buffer[5000]; // temporary scratch space

  std::cout << "\nInference Active state\n";
  for (Size i = 0; i < param_.cellsPerColumn; i++) {
    if (printPrevious)
      std::cout << formatRow(buffer, sizeof(buffer), infActiveState_["t-1"], i,
                             param_.numberOfCols, param_.cellsPerColumn);
    std::cout << formatRow(buffer, sizeof(buffer), infActiveState_["t"], i,
                           param_.numberOfCols, param_.cellsPerColumn);
    std::cout << "\n";
  }
  std::cout << "\nInference Predicted state\n";
  for (Size i = 0; i < param_.cellsPerColumn; i++) {
    if (printPrevious)
      std::cout << formatRow(buffer, sizeof(buffer), infPredictedState_["t-1"],
                             i, param_.numberOfCols, param_.cellsPerColumn);
    std::cout << formatRow(buffer, sizeof(buffer), infPredictedState_["t"], i,
                           param_.numberOfCols, param_.cellsPerColumn);
    std::cout << "\n";
  }

  if (printLearnState) {
    std::cout << "\nLearn Active state\n";
    for (Size i = 0; i < param_.cellsPerColumn; i++) {
      if (printPrevious)
        std::cout << formatRow(buffer, sizeof(buffer), lrnActiveState_["t-1"],
                               i, param_.numberOfCols, param_.cellsPerColumn);
      std::cout << formatRow(buffer, sizeof(buffer), lrnActiveState_["t"], i,
                             param_.numberOfCols, param_.cellsPerColumn);
      std::cout << "\n";
    }

    std::cout << "Learn Predicted state\n";
    for (Size i = 0; i < param_.cellsPerColumn; i++) {
      if (printPrevious)
        std::cout << formatRow(buffer, sizeof(buffer),
                               lrnPredictedState_["t-1"], i,
                               param_.numberOfCols, param_.cellsPerColumn);
      std::cout << formatRow(buffer, sizeof(buffer), lrnPredictedState_["t"], i,
                             param_.numberOfCols, param_.cellsPerColumn);
      std::cout << "\n";
    }
  }
}

/////////////////////////////////////////////////////////////

// A segment is active if it has >= activationThreshold connected
// synapses that are active due to infActiveState.
bool BacktrackingTMCpp::_slowIsSegmentActive(Segment &seg, const char *timestep) 
{
  Size numActiveSyns = 0;
  for (UInt synIdx = 0; synIdx < (UInt)seg.size(); synIdx++) {
    if (seg.getPermanence(synIdx) >= param_.connectedPerm) {
      Size srcIdx = seg.getSrcCellIdx(synIdx);
      Byte *state = infActiveState_[timestep];
      if (state[srcIdx]) {
        numActiveSyns += 1;
        if (numActiveSyns >= param_.activationThreshold)
          return true;
      }
    }
  }
  return (numActiveSyns >= param_.activationThreshold);
}

vector<union BacktrackingTMCpp::segoncellinfo_t>
BacktrackingTMCpp::getSegmentOnCell(Size c, Size i, Size segIdx) 
{
  std::vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
  Segment &seg = cells4_->getSegment((UInt)c, (UInt)i, segList[segIdx]);
  Size numSyn = seg.size();
  NTA_ASSERT(numSyn != 0);

  // Accumulate segment information
  std::vector<union BacktrackingTMCpp::segoncellinfo_t> result;

  // first element is the segment info
  union BacktrackingTMCpp::segoncellinfo_t info;
  info.se.segIdx = segIdx;
  info.se.isSequenceSegment = seg.isSequenceSegment();
  info.se.positiveActivations = seg.getPositiveActivations();
  info.se.totalActivations = seg.getTotalActivations();
  info.se.lastActiveIteration = seg.getLastActiveIteration();
  info.se.lastPosDutyCycle = seg.getLastPosDutyCycle();
  info.se.lastPosDutyCycleIteration = seg.getLastPosDutyCycleIteration();
  result.push_back(info);

  // remaing elements are synapse info
  for (Size s = 0; s < numSyn; numSyn++) {
    UInt idx = seg.getSrcCellIdx((UInt)s);
    info.sy.c = idx / param_.cellsPerColumn;
    info.sy.i = idx % param_.cellsPerColumn;
    info.sy.permanence = seg.getPermanence((UInt)s);
    result.push_back(info);
  }
  return result;
}

struct BacktrackingTMCpp::seginfo_t
BacktrackingTMCpp::getSegmentInfo(bool collectActiveData) 
{
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

  for (Size c = 0; c < param_.numberOfCols; c++) {
    for (Size i = 0; i < param_.cellsPerColumn; i++) {
      Size nSegmentsThisCell = getNumSegmentsInCell(c, i);
      if (nSegmentsThisCell > 0) {
        // Update histogram counting cell sizes
        if (info.distNSegsPerCell.find(nSegmentsThisCell) != info.distNSegsPerCell.end())
          info.distNSegsPerCell[nSegmentsThisCell] += 1;
        else
          info.distNSegsPerCell[nSegmentsThisCell] = 1;

        // Update histogram counting segment sizes.
        vector<UInt32> segList = cells4_->getNonEmptySegList((UInt)c, (UInt)i);
        for (Size segIdx = 0; segIdx < nSegmentsThisCell; segIdx++) {
          vector<union BacktrackingTMCpp::segoncellinfo_t> segcellinfo = getSegmentOnCell(c, i, segIdx);
          Size nSynapsesThisSeg = segcellinfo.size() - 1;
          if (nSynapsesThisSeg > 0) {
            if (info.distSegSizes.find(nSynapsesThisSeg) !=  info.distSegSizes.end())
              info.distSegSizes[nSynapsesThisSeg] += 1;
            else
              info.distSegSizes[nSynapsesThisSeg] = 1;

            // Accumulate permanence value histogram (scaled by 10)
            for (Size synIdx = 1; synIdx < nSynapsesThisSeg; synIdx++) {
              Size p = (Size)(segcellinfo[synIdx].sy.permanence * 10.0);
              if (info.distPermValues.find(p) != info.distPermValues.end())
                info.distPermValues[p] += 1;
              else
                info.distPermValues[p] = 1;
            }
          }
          Segment &segObj =cells4_->getSegment((UInt)c, (UInt)i, segList[segIdx]);
          Size age = loc_.iterationIdx - segObj.getLastActiveIteration();
          Size ageBucket = age / ageBucketSize;
          info.distAges[ageBucket].cnt += 1;
        }
      }
    }
  }
  return info;
}

//////////////////  Serialization ///////////////////////
void BacktrackingTMCpp::saveToFile(std::string filePath) 
{
  std::ofstream out(filePath.c_str(),  std::ios_base::out | std::ios_base::binary);
  out.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  out.precision(std::numeric_limits<double>::digits10 + 1);
  save(out);
  out.close();
}
void BacktrackingTMCpp::loadFromFile(std::string filePath) 
{
  std::ifstream in(filePath.c_str(), std::ios_base::in | std::ios_base::binary);
  load(in);
  in.close();
}


void BacktrackingTMCpp::save(std::ofstream& out) 
{ 
  cells4_->save(out); 
  out << param_.burnIn;
  out << param_.collectStats;
  out << param_.seed;
  std::string outputType(param_.outputType);
  out << outputType;

  out << loc_.lrnIterationIdx;
  out << loc_.iterationIdx;
  out << loc_.segID;
  out << loc_.pamCounter;
  out << loc_.collectSequenceStats;
  out << loc_.resetCalled;
  out << loc_.avgInputDensity;
  out << loc_.learnedSeqLength;
  out << loc_.avgLearnedSeqLength;
  out << loc_.retrieveLearningStates;
  out << std::endl;

  UInt* output = nonzero(currentOutput_.get(), nCells);
  out << " CurrentOutput [ " << output[0];
  for (Size i = 1; i < output[0]; i++) {
    out << output[i] << " ";
  }
  out << "] ";
  delete[] output;
}


void BacktrackingTMCpp::load(std::ifstream& in) 
{
  std::string tag;

  cells4_ = new Cells4::Cells4();
  cells4_->load(in);

  // Fields restored by Cells4 that are needed here
  // So this class does not need to serialize them.
  nCells = cells4_->nCells();
  param_.numberOfCols = cells4_->nColumns();
  param_.cellsPerColumn = cells4_->nCellsPerCol();
  param_.initialPerm = cells4_->getPermInitial();
  param_.connectedPerm = cells4_->getPermConnected();
  param_.minThreshold = cells4_->getMinThreshold();
  param_.newSynapseCount = cells4_->getNewSynapseCount();
  param_.permanenceInc = cells4_->getPermInc();
  param_.permanenceDec = cells4_->getPermDec();
  param_.permanenceMax = cells4_->getPermMax();
  param_.globalDecay = cells4_->getGlobalDecay();
  param_.activationThreshold = cells4_->getActivationThreshold();
  param_.doPooling = cells4_->getDoPooling();
  param_.segUpdateValidDuration = cells4_->getSegUpdateValidDuration();
  param_.verbosity = cells4_->getVerbosity();
  param_.checkSynapseConsistency = cells4_->getCheckSynapseConsistency();
  param_.pamLength = cells4_->getPamLength();
  param_.maxInfBacktrack = cells4_->getMaxInfBacktrack();
  param_.maxLrnBacktrack = cells4_->getMaxLrnBacktrack();
  param_.maxAge = cells4_->getMaxAge();
  param_.maxSeqLength = cells4_->getMaxSeqLength();
  param_.maxSegmentsPerCell = cells4_->getMaxSegmentsPerCell();
  param_.maxSynapsesPerSegment = cells4_->getMaxSynapsesPerSegment();

  // Fields that this class needed to serialize
  in >> param_.burnIn;
  in >> param_.collectStats;
  in >> param_.seed;
  in >> tag;
  strncpy(param_.outputType, tag.c_str(), sizeof(param_.outputType));

  in >> loc_.lrnIterationIdx;
  in >> loc_.iterationIdx;
  in >> loc_.segID;
  in >> loc_.pamCounter;
  in >> loc_.collectSequenceStats;
  in >> loc_.resetCalled;
  in >> loc_.avgInputDensity;
  in >> loc_.learnedSeqLength;
  in >> loc_.avgLearnedSeqLength;
  in >> loc_.retrieveLearningStates;

  // restore the currentOutput_
  // saved as non-zero indexes with first element the count of elements.
  currentOutput_.reset(new Real[nCells], std::default_delete<Real[]>());
  Real *ptr = currentOutput_.get();
  memset(ptr, 0, nCells * sizeof(Real));
  Size buflen;
  UInt32 i;
  in >> tag;
  NTA_ASSERT(tag == "CurrentOutput");
  in >> tag;
  NTA_ASSERT(tag == "[");
  in >> buflen;
  for (Size idx = 1; idx < buflen; idx++) {
    in >> i;
    ptr[i] = 1;
  }
  in >> tag;
  NTA_ASSERT(tag == "]");

  loc_.allocateStatesInCPP = true;
  loc_.makeCells4Ephemeral = false;

  // get the buffers from Cells4.
  Byte *activeT;
  Byte *activeT1;
  Byte *predT;
  Byte *predT1;
  Real *colConfidenceT;
  Real *colConfidenceT1;
  Real *confidenceT;
  Real *confidenceT1;

  cells4_->getStatePointers(activeT, activeT1, predT, predT1, colConfidenceT,
                            colConfidenceT1, confidenceT, confidenceT1);
  infActiveState_["t"] = activeT;
  infActiveState_["t-1"] = activeT1;
  infPredictedState_["t"] = predT;
  infPredictedState_["t-1"] = predT1;
  cellConfidence_["t"] = colConfidenceT;
  colConfidence_["t-1"] = colConfidenceT1;
  cellConfidence_["t"] = confidenceT;
  cellConfidence_["t-1"] = confidenceT1;


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
