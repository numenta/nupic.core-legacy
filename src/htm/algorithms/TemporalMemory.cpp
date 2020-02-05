/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013-2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

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

#include <algorithm> //is_sorted
#include <climits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <set>


#include <htm/algorithms/TemporalMemory.hpp>

#include <htm/utils/GroupBy.hpp>
#include <htm/algorithms/Anomaly.hpp>

using namespace std;
using namespace htm;


static const UInt TM_VERSION = 2;

TemporalMemory::TemporalMemory() {}

TemporalMemory::TemporalMemory(
    vector<CellIdx> columnDimensions, 
    CellIdx cellsPerColumn,
    SynapseIdx activationThreshold, 
    Permanence initialPermanence,
    Permanence connectedPermanence, 
    SynapseIdx minThreshold, 
    SynapseIdx maxNewSynapseCount,
    Permanence permanenceIncrement, 
    Permanence permanenceDecrement,
    Permanence predictedSegmentDecrement, 
    Int seed, 
    SegmentIdx maxSegmentsPerCell,
    SynapseIdx maxSynapsesPerSegment, 
    bool checkInputs, 
    UInt externalPredictiveInputs,
    ANMode anomalyMode) {

  initialize(columnDimensions, cellsPerColumn, activationThreshold,
             initialPermanence, connectedPermanence, minThreshold,
             maxNewSynapseCount, permanenceIncrement, permanenceDecrement,
             predictedSegmentDecrement, seed, maxSegmentsPerCell,
             maxSynapsesPerSegment, checkInputs, externalPredictiveInputs, anomalyMode);
}

TemporalMemory::~TemporalMemory() {}

void TemporalMemory::initialize(
    vector<CellIdx> columnDimensions, 
    CellIdx cellsPerColumn,
    SynapseIdx activationThreshold, 
    Permanence initialPermanence,
    Permanence connectedPermanence, 
    SynapseIdx minThreshold, 
    SynapseIdx maxNewSynapseCount,
    Permanence permanenceIncrement, 
    Permanence permanenceDecrement,
    Permanence predictedSegmentDecrement, 
    Int seed, 
    SegmentIdx maxSegmentsPerCell,
    SynapseIdx maxSynapsesPerSegment, 
    bool checkInputs, 
    UInt externalPredictiveInputs,
    ANMode anomalyMode) {

  // Validate all input parameters
  NTA_CHECK(columnDimensions.size() > 0) << "Number of column dimensions must be greater than 0";
  NTA_CHECK(cellsPerColumn > 0) << "Number of cells per column must be greater than 0";

  NTA_CHECK(initialPermanence >= 0.0 && initialPermanence <= 1.0);
  NTA_CHECK(connectedPermanence >= 0.0 && connectedPermanence <= 1.0);
  NTA_CHECK(permanenceIncrement >= 0.0 && permanenceIncrement <= 1.0);
  NTA_CHECK(permanenceDecrement >= 0.0 && permanenceDecrement <= 1.0);
  NTA_CHECK(minThreshold <= activationThreshold);

  // Save member variables

  numColumns_ = 1;
  columnDimensions_.clear();
  for (auto &columnDimension : columnDimensions) {
    numColumns_ *= columnDimension;
    columnDimensions_.push_back(columnDimension);
  }

  
  cellsPerColumn_ = cellsPerColumn; //TODO add checks
  activationThreshold_ = activationThreshold;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  checkInputs_ = checkInputs;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;
  predictedSegmentDecrement_ = predictedSegmentDecrement;
  externalPredictiveInputs_ = externalPredictiveInputs;

  // Initialize member variables
  connections_ = Connections(static_cast<CellIdx>(numberOfColumns() * cellsPerColumn_), connectedPermanence_);
  rng_ = Random(seed);

  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;

  tmAnomaly_.mode_ = anomalyMode;

  reset();
}

CellIdx TemporalMemory::getLeastUsedCell_(const CellIdx column) {
  if(cellsPerColumn_ == 1) return column;

  vector<CellIdx> cells = cellsForColumn(column);

  //TODO: decide if we need to choose randomly from the "least used" cells, or if 1st is fine. 
  //In that case the line below is not needed, and this method can become const, deterministic results in tests need to be updated
  //un/comment line below: 
  rng_.shuffle(cells.begin(), cells.end()); //as min_element selects 1st minimal element, and we want to randomly choose 1 from the minimals.

  const auto compareByNumSegments = [&](const CellIdx a, const CellIdx b) {
    if(connections.numSegments(a) == connections.numSegments(b)) 
      return a < b; //TODO rm? 
    else return connections.numSegments(a) < connections.numSegments(b);
  };
  return *std::min_element(cells.begin(), cells.end(), compareByNumSegments);
}


void TemporalMemory::growSynapses_(
			 const Segment& segment,
                         const SynapseIdx nDesiredNewSynapses,
                         const vector<CellIdx> &prevWinnerCells) {
  
  vector<CellIdx> candidates(prevWinnerCells.begin(), prevWinnerCells.end());
  NTA_ASSERT(std::is_sorted(candidates.begin(), candidates.end()));

  //figure the number of new synapses to grow
  const size_t nActual = std::min(static_cast<size_t>(nDesiredNewSynapses), candidates.size());
  // ..Check if we're going to surpass the maximum number of synapses.
  Int overrun = static_cast<Int>(connections.numSynapses(segment) + nActual - maxSynapsesPerSegment_);
  if (overrun > 0) {
    connections_.destroyMinPermanenceSynapses(segment, static_cast<Int>(overrun), prevWinnerCells);
  }
  // ..Recalculate in case we weren't able to destroy as many synapses as needed.
  const size_t nActualWithMax = std::min(nActual, static_cast<size_t>(maxSynapsesPerSegment_) - connections.numSynapses(segment));

  // Pick nActual cells randomly.
  rng_.shuffle(candidates.begin(), candidates.end());
  const size_t nDesired = connections.numSynapses(segment) + nActualWithMax; //num synapses on seg after this function (+-), see #COND
  for (const auto syn : candidates) {
    // #COND: this loop finishes two folds: a) we ran out of candidates (above), b) we grew the desired number of new synapses (below)
    if(connections.numSynapses(segment) == nDesired) break;
    connections_.createSynapse(segment, syn, initialPermanence_); //TODO createSynapse consider creating a vector of new synapses at once?
  }
}


void TemporalMemory::activatePredictedColumn_(
    vector<Segment>::const_iterator columnActiveSegmentsBegin,
    vector<Segment>::const_iterator columnActiveSegmentsEnd,
    const SDR &prevActiveCells,
    const vector<CellIdx> &prevWinnerCells,
    const bool learn) {

  auto activeSegment = columnActiveSegmentsBegin;
  do {
    const CellIdx cell = connections.cellForSegment(*activeSegment);
    activeCells_.push_back(cell);
    winnerCells_.push_back(cell);

    // This cell might have multiple active segments.
    do {
      if (learn) { 
        connections_.adaptSegment(*activeSegment, prevActiveCells,
                     permanenceIncrement_, permanenceDecrement_, true);

        const Int32 nGrowDesired =
            static_cast<Int32>(maxNewSynapseCount_) -
            numActivePotentialSynapsesForSegment_[*activeSegment];
        if (nGrowDesired > 0) {
          growSynapses_(*activeSegment, nGrowDesired, prevWinnerCells);
        }
      }
    } while (++activeSegment != columnActiveSegmentsEnd &&
             connections.cellForSegment(*activeSegment) == cell);
  } while (activeSegment != columnActiveSegmentsEnd);
}


void TemporalMemory::burstColumn_(
	    const UInt column,
            vector<Segment>::const_iterator columnMatchingSegmentsBegin,
            vector<Segment>::const_iterator columnMatchingSegmentsEnd,
            const SDR &prevActiveCells,
            const vector<CellIdx> &prevWinnerCells,
            const bool learn) {

  // Calculate the active cells: active become ALL the cells in this mini-column
  const auto newCells = cellsForColumn(column);
  activeCells_.insert(activeCells_.end(), newCells.begin(), newCells.end());

  const auto bestMatchingSegment =
      std::max_element(columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
                       [&](Segment a, Segment b) {
                         return (numActivePotentialSynapsesForSegment_[a] <
                                 numActivePotentialSynapsesForSegment_[b]);
                       });

  const CellIdx winnerCell =
      (bestMatchingSegment != columnMatchingSegmentsEnd)
          ? connections.cellForSegment(*bestMatchingSegment)
          : getLeastUsedCell_(column); //TODO replace (with random?) this is extremely costly, removing makes TM 6x faster!

  winnerCells_.push_back(winnerCell);

  // Learn.
  if (learn) {
    if (bestMatchingSegment != columnMatchingSegmentsEnd) {
      // Learn on the best matching segment.
      connections_.adaptSegment(*bestMatchingSegment, prevActiveCells,
                   permanenceIncrement_, permanenceDecrement_, true);

      const Int32 nGrowDesired = maxNewSynapseCount_ - numActivePotentialSynapsesForSegment_[*bestMatchingSegment];
      if (nGrowDesired > 0) {
        growSynapses_(*bestMatchingSegment, nGrowDesired, prevWinnerCells);
      }
    } else {
      // No matching segments.
      // Grow a new segment and learn on it.

      // Don't grow a segment that will never match.
      const UInt32 nGrowExact =
          std::min(static_cast<UInt32>(maxNewSynapseCount_), static_cast<UInt32>(prevWinnerCells.size()));
      if (nGrowExact > 0) {
        const Segment segment =
            connections_.createSegment(winnerCell, maxSegmentsPerCell_);

        growSynapses_(segment, nGrowExact, prevWinnerCells);
        NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
      }
    }
  }
}


void TemporalMemory::punishPredictedColumn_(
    vector<Segment>::const_iterator columnMatchingSegmentsBegin,
    vector<Segment>::const_iterator columnMatchingSegmentsEnd,
    const SDR &prevActiveCells) {
  if (predictedSegmentDecrement_ > 0.0) {
    for (auto matchingSegment = columnMatchingSegmentsBegin;
         matchingSegment != columnMatchingSegmentsEnd; matchingSegment++) {
      connections_.adaptSegment(*matchingSegment, prevActiveCells,
                   -predictedSegmentDecrement_, 0.0, true);
    }
  }
}

void TemporalMemory::activateCells(const SDR &activeColumns, const bool learn) {
    NTA_CHECK(columnDimensions_.size() > 0) << "TM constructed using the default TM() constructor, which may only be used for serialization. "
	    << "Use TM constructor where you provide at least column dimensions, eg: TM tm({32});";

    NTA_CHECK( activeColumns.dimensions.size() == columnDimensions_.size() )  //this "hack" because columnDimensions_, and SDR.dimensions are vectors
	    //of different type, so we cannot directly compare
	    << "TM invalid input dimensions: " << activeColumns.dimensions.size() << " vs. " << columnDimensions_.size();
    for(size_t i=0; i< columnDimensions_.size(); i++) {
      NTA_CHECK(static_cast<size_t>(activeColumns.dimensions[i]) == static_cast<size_t>(columnDimensions_[i])) << "Dimensions must be the same.";
    }
    auto &sparse = activeColumns.getSparse();

  SDR prevActiveCells({static_cast<CellIdx>(numberOfCells() + externalPredictiveInputs_)});
  prevActiveCells.setSparse(activeCells_);
  activeCells_.clear();

  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  //maps segment S to a new segment that is at start of a column where
  //S belongs. 
  //for 3 cells per columns: 
  //s1_1, s1_2, s1_3, s2_1, s2_2, s2_3, ...
  //columnForSegment (for short here CFS)
  //CFS(s1_1) = s1_1 = "start of column 1"
  //CFS(s1_2) = s1_1
  //CFS(s1_3) = s1_1
  //CFS(s2_1) = s2_1 = "column 2"
  //CFS(s2_2) = s2_1
  //...
  const auto toColumns = [&](const Segment segment) {
    return connections.cellForSegment(segment) / cellsPerColumn_;
  };
  const auto identity = [](const ElemSparse a) {return a;}; //TODO use std::identity when c++20

  for (auto &&columnData : groupBy( //group by columns, and convert activeSegments & matchingSegments to cols. 
           sparse, identity,
           activeSegments_,   toColumns,
           matchingSegments_, toColumns)) {

    Segment column; //we say "column", but it's the first segment of n-segments/cells that belong to the column
    vector<Segment>::const_iterator activeColumnsBegin, activeColumnsEnd, 
	                            columnActiveSegmentsBegin, columnActiveSegmentsEnd, 
                                    columnMatchingSegmentsBegin, columnMatchingSegmentsEnd;

    // for column in activeColumns (the 'sparse' above):
    //   get its active segments ( >= connectedThr)
    //   get its matching segs   ( >= TODO
    std::tie(column, 
             activeColumnsBegin, activeColumnsEnd, 
             columnActiveSegmentsBegin, columnActiveSegmentsEnd, 
             columnMatchingSegmentsBegin, columnMatchingSegmentsEnd
	) = columnData;

    const bool isActiveColumn = activeColumnsBegin != activeColumnsEnd;
    if (isActiveColumn) { //current active column...
      if (columnActiveSegmentsBegin != columnActiveSegmentsEnd) {
	//...was also predicted -> learn :o)
        activatePredictedColumn_(
            columnActiveSegmentsBegin, columnActiveSegmentsEnd,
            prevActiveCells, prevWinnerCells, learn);
      } else {
	//...has not been predicted -> 
        burstColumn_(column,
                     columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
                     prevActiveCells, prevWinnerCells, 
		     learn);
      }

    } else { // predicted but not active column -> unlearn
      if (learn) {
        punishPredictedColumn_(columnMatchingSegmentsBegin, columnMatchingSegmentsEnd, prevActiveCells);
      }
    } //else: not predicted & not active -> no activity -> does not show up at all
  }
  segmentsValid_ = false;
}


void TemporalMemory::activateDendrites(const bool learn,
                                       const SDR &externalPredictiveInputsActive,
                                       const SDR &externalPredictiveInputsWinners)
{
    if( externalPredictiveInputs_ > 0 )
    {
        NTA_CHECK( externalPredictiveInputsActive.size  == externalPredictiveInputs_ );
        NTA_CHECK( externalPredictiveInputsWinners.size == externalPredictiveInputs_ );
	NTA_CHECK( externalPredictiveInputsActive.dimensions == externalPredictiveInputsWinners.dimensions);
#ifdef NTA_ASSERTIONS_ON
  SDR both(externalPredictiveInputsActive.dimensions);
  both.intersection(externalPredictiveInputsActive, externalPredictiveInputsWinners);
  NTA_ASSERT(both == externalPredictiveInputsWinners) << "externalPredictiveInputsWinners must be a subset of externalPredictiveInputsActive";
#endif
    }
    else
    {
        NTA_CHECK( externalPredictiveInputsActive.getSum() == 0u && externalPredictiveInputsWinners.getSum() == 0u )
            << "External predictive inputs must be declared to TM constructor!";
    }


  if( segmentsValid_ )
    return;

  for(const auto &active : externalPredictiveInputsActive.getSparse()) {
      NTA_ASSERT( active < externalPredictiveInputs_ );
      activeCells_.push_back( static_cast<CellIdx>(active + numberOfCells()) ); 
  }
  for(const auto &winner : externalPredictiveInputsWinners.getSparse()) {
      NTA_ASSERT( winner < externalPredictiveInputs_ );
      winnerCells_.push_back( static_cast<CellIdx>(winner + numberOfCells()) );
  }

  const size_t length = connections.segmentFlatListLength();

  numActivePotentialSynapsesForSegment_.assign(length, 0);
  numActiveConnectedSynapsesForSegment_ = connections_.computeActivity(
                              numActivePotentialSynapsesForSegment_,
                              activeCells_,
			      learn);

  // Active segments, connected synapses.
  activeSegments_.clear();
  for (Segment segment = 0; segment < numActiveConnectedSynapsesForSegment_.size(); segment++) {
    if (numActiveConnectedSynapsesForSegment_[segment] >= activationThreshold_) { //TODO move to SegmentData.numConnected?
      activeSegments_.push_back(segment);
    }
  }
  const auto compareSegments = [&](const Segment a, const Segment b) { return connections.compareSegments(a, b); };
  std::sort( activeSegments_.begin(), activeSegments_.end(), compareSegments); //SDR requires sorted when constructed from activeSegments_
  // Update segment bookkeeping.
  if (learn) {
    for (const auto segment : activeSegments_) {
      connections_.dataForSegment(segment).lastUsed = connections.iteration(); //TODO the destroySegments based on LRU is expensive. Better random? or "energy" based on sum permanences?
    }
  }

  // Matching segments, potential synapses.
  matchingSegments_.clear();
  for (Segment segment = 0; segment < numActivePotentialSynapsesForSegment_.size(); segment++) {
    if (numActivePotentialSynapsesForSegment_[segment] >= minThreshold_) {
      matchingSegments_.push_back(segment);
    }
  }
  std::sort( matchingSegments_.begin(), matchingSegments_.end(), compareSegments);

  segmentsValid_ = true;
}


void TemporalMemory::compute(const SDR &activeColumns, 
                             const bool learn,
                             const SDR &externalPredictiveInputsActive,
                             const SDR &externalPredictiveInputsWinners)
{
  activateDendrites(learn, externalPredictiveInputsActive, externalPredictiveInputsWinners);

  calculateAnomalyScore_(activeColumns);

  activateCells(activeColumns, learn);
}

void TemporalMemory::calculateAnomalyScore_(const SDR &activeColumns){

  // Update Anomaly Metric.  The anomaly is the percent of active columns that
  // were not predicted. 
  // Must be computed here, between `activateDendrites()` and `activateCells()`.
  switch(tmAnomaly_.mode_) {

	case ANMode::DISABLED: {
	  tmAnomaly_.anomaly_ = 0.5f;
			   } break;

	case ANMode::RAW: {
	  tmAnomaly_.anomaly_ = computeRawAnomalyScore(
							 activeColumns,
							 cellsToColumns( getPredictiveCells() ));
			  } break;

	case ANMode::LIKELIHOOD: {
	  const Real raw = computeRawAnomalyScore(
						 activeColumns,
						 cellsToColumns( getPredictiveCells() ));
	  tmAnomaly_.anomaly_ = tmAnomaly_.anomalyLikelihood_.anomalyProbability(raw);
				 } break;

	case ANMode::LOGLIKELIHOOD: {
	  const Real raw = computeRawAnomalyScore(
						 activeColumns,
						 cellsToColumns( getPredictiveCells() ));
	  const Real like = tmAnomaly_.anomalyLikelihood_.anomalyProbability(raw);
	  const Real log  = tmAnomaly_.anomalyLikelihood_.computeLogLikelihood(like);
	  tmAnomaly_.anomaly_ = log;
				} break;
  // TODO: Update mean & standard deviation of anomaly here.
  };
  NTA_ASSERT(tmAnomaly_.anomaly_ >= 0.0f and tmAnomaly_.anomaly_ <= 1.0f) << "TM.anomaly is out-of-bounds!";


}

void TemporalMemory::compute(const SDR &activeColumns, const bool learn) {
  SDR externalPredictiveInputsActive({ externalPredictiveInputs_ });
  SDR externalPredictiveInputsWinners({ externalPredictiveInputs_ });
  compute( activeColumns, learn, externalPredictiveInputsActive, externalPredictiveInputsWinners );
}

void TemporalMemory::reset(void) {
  activeCells_.clear();
  winnerCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
  segmentsValid_ = false;
  tmAnomaly_.anomaly_ = -1.0f; //TODO reset rather to 0.5 as default (undecided) anomaly
}

// ==============================
//  Helper functions
// ==============================
UInt TemporalMemory::columnForCell(const CellIdx cell) const {
  NTA_ASSERT(cell < numberOfCells());
  return cell / cellsPerColumn_;
}


SDR TemporalMemory::cellsToColumns(const SDR& cells) const {
  auto correctDims = getColumnDimensions(); //nD column dimensions (eg 10x100)
  correctDims.push_back(static_cast<CellIdx>(getCellsPerColumn())); //add n+1-th dimension for cellsPerColumn (eg. 10x100x8)

  NTA_CHECK(cells.dimensions.size() == correctDims.size()) 
	  << "cells.dimensions must match TM's (column dims x cellsPerColumn) ";

  for(size_t i = 0; i<correctDims.size(); i++) 
	  NTA_CHECK(correctDims[i] == cells.dimensions[i]);

  SDR cols(getColumnDimensions());
  auto& dense = cols.getDense();
  for(const auto cell : cells.getSparse()) {
    const auto col = columnForCell(cell);
    dense[col] = static_cast<ElemDense>(1);
  }
  cols.setDense(dense);

  NTA_ASSERT(cols.size == numColumns_); 
  return cols;
}


vector<CellIdx> TemporalMemory::cellsForColumn(const CellIdx column) const { 
  const CellIdx start = cellsPerColumn_ * column;
  const CellIdx end = start + cellsPerColumn_;

  vector<CellIdx> cellsInColumn;
  cellsInColumn.reserve(cellsPerColumn_);
  for (CellIdx i = start; i < end; i++) {
    cellsInColumn.push_back(i);
  }

  return cellsInColumn;
}

vector<CellIdx> TemporalMemory::getActiveCells() const { return activeCells_; }

void TemporalMemory::getActiveCells(SDR &activeCells) const
{
  NTA_CHECK( activeCells.size == numberOfCells() );
  activeCells.setSparse( getActiveCells() );
}


SDR TemporalMemory::getPredictiveCells() const {

  NTA_CHECK( segmentsValid_ )
    << "Call TM.activateDendrites() before TM.getPredictiveCells()!";

  auto correctDims = getColumnDimensions();
  correctDims.push_back(static_cast<CellIdx>(getCellsPerColumn()));
  SDR predictive(correctDims);

  std::set<CellIdx> uniqueCells;
  //uniqueCells.reserve(activeSegments_.size());

  for (const auto segment : activeSegments_) {
    const CellIdx cell = connections.cellForSegment(segment);
    uniqueCells.insert(cell); //set keeps the cells unique
  }

  vector<CellIdx> predictiveCells(uniqueCells.begin(), uniqueCells.end());
  predictive.setSparse(predictiveCells);
  return predictive;
}


vector<CellIdx> TemporalMemory::getWinnerCells() const { return winnerCells_; }

void TemporalMemory::getWinnerCells(SDR &winnerCells) const
{
  NTA_CHECK( winnerCells.size == numberOfCells() );
  winnerCells.setSparse( getWinnerCells() );
}

vector<Segment> TemporalMemory::getActiveSegments() const
{
  NTA_CHECK( segmentsValid_ )
    << "Call TM.activateDendrites() before TM.getActiveSegments()!";

  return activeSegments_;
}

vector<Segment> TemporalMemory::getMatchingSegments() const
{
  NTA_CHECK( segmentsValid_ )
    << "Call TM.activateDendrites() before TM.getActiveSegments()!";

  return matchingSegments_;
}


SynapseIdx TemporalMemory::getActivationThreshold() const {
  return activationThreshold_;
}

void TemporalMemory::setActivationThreshold(const SynapseIdx activationThreshold) {
  activationThreshold_ = activationThreshold;
}

Permanence TemporalMemory::getInitialPermanence() const {
  return initialPermanence_;
}

void TemporalMemory::setInitialPermanence(const Permanence initialPermanence) {
  initialPermanence_ = initialPermanence;
}

Permanence TemporalMemory::getConnectedPermanence() const {
  return connectedPermanence_;
}

SynapseIdx TemporalMemory::getMinThreshold() const { return minThreshold_; }

void TemporalMemory::setMinThreshold(const SynapseIdx minThreshold) {
  minThreshold_ = minThreshold;
}

SynapseIdx TemporalMemory::getMaxNewSynapseCount() const {
  return maxNewSynapseCount_;
}

void TemporalMemory::setMaxNewSynapseCount(const SynapseIdx maxNewSynapseCount) {
  maxNewSynapseCount_ = maxNewSynapseCount;
}

bool TemporalMemory::getCheckInputs() const { return checkInputs_; }

void TemporalMemory::setCheckInputs(bool checkInputs) {
  checkInputs_ = checkInputs;
}

Permanence TemporalMemory::getPermanenceIncrement() const {
  return permanenceIncrement_;
}

void TemporalMemory::setPermanenceIncrement(Permanence permanenceIncrement) {
  permanenceIncrement_ = permanenceIncrement;
}

Permanence TemporalMemory::getPermanenceDecrement() const {
  return permanenceDecrement_;
}

void TemporalMemory::setPermanenceDecrement(Permanence permanenceDecrement) {
  permanenceDecrement_ = permanenceDecrement;
}

Permanence TemporalMemory::getPredictedSegmentDecrement() const {
  return predictedSegmentDecrement_;
}

void TemporalMemory::setPredictedSegmentDecrement(
    Permanence predictedSegmentDecrement) {
  predictedSegmentDecrement_ = predictedSegmentDecrement;
}

SegmentIdx TemporalMemory::getMaxSegmentsPerCell() const {
  return maxSegmentsPerCell_;
}

SynapseIdx TemporalMemory::getMaxSynapsesPerSegment() const {
  return maxSynapsesPerSegment_;
}

UInt TemporalMemory::version() const { return TM_VERSION; }


static set<pair<CellIdx, SynapseIdx>>
getComparableSegmentSet(const Connections &connections,
                        const vector<Segment> &segments) {
  set<pair<CellIdx, SynapseIdx>> segmentSet;
  for (Segment segment : segments) {
    segmentSet.emplace(connections.cellForSegment(segment),
                       connections.idxOnCellForSegment(segment));
  }
  return segmentSet;
}

bool TemporalMemory::operator==(const TemporalMemory &other) const {
  if (numColumns_ != other.numColumns_ ||
      columnDimensions_ != other.columnDimensions_ ||
      cellsPerColumn_ != other.cellsPerColumn_ ||
      activationThreshold_ != other.activationThreshold_ ||
      minThreshold_ != other.minThreshold_ ||
      maxNewSynapseCount_ != other.maxNewSynapseCount_ ||
      initialPermanence_ != other.initialPermanence_ ||
      connectedPermanence_ != other.connectedPermanence_ ||
      permanenceIncrement_ != other.permanenceIncrement_ ||
      permanenceDecrement_ != other.permanenceDecrement_ ||
      predictedSegmentDecrement_ != other.predictedSegmentDecrement_ ||
      activeCells_ != other.activeCells_ ||
      winnerCells_ != other.winnerCells_ ||
      maxSegmentsPerCell_ != other.maxSegmentsPerCell_ ||
      maxSynapsesPerSegment_ != other.maxSynapsesPerSegment_ ||
      tmAnomaly_.anomaly_ != other.tmAnomaly_.anomaly_ ||
      tmAnomaly_.mode_ != other.tmAnomaly_.mode_ || 
      tmAnomaly_.anomalyLikelihood_ != other.tmAnomaly_.anomalyLikelihood_ ) {
    return false;
  }

  if (connections != other.connections) {
    return false;
  }

  if (getComparableSegmentSet(connections, activeSegments_) !=
          getComparableSegmentSet(other.connections, other.activeSegments_) ||
      getComparableSegmentSet(connections, matchingSegments_) !=
          getComparableSegmentSet(other.connections, other.matchingSegments_)) {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------


namespace htm {
std::ostream& operator<< (std::ostream& stream, const TemporalMemory& self)
{
  stream << "Temporal Memory " << self.connections;
  return stream;
}
}


// Print the main TM creation parameters
void TemporalMemory::printParameters(std::ostream& out) const {
  out << "Temporal Memory Parameters\n";
  out << "version                   = " << TM_VERSION << std::endl
      << "numColumns                = " << numberOfColumns() << std::endl
      << "cellsPerColumn            = " << getCellsPerColumn() << std::endl
      << "activationThreshold       = " << getActivationThreshold() << std::endl
      << "initialPermanence         = " << getInitialPermanence() << std::endl
      << "connectedPermanence       = " << getConnectedPermanence() << std::endl
      << "minThreshold              = " << getMinThreshold() << std::endl
      << "maxNewSynapseCount        = " << getMaxNewSynapseCount() << std::endl
      << "permanenceIncrement       = " << getPermanenceIncrement() << std::endl
      << "permanenceDecrement       = " << getPermanenceDecrement() << std::endl
      << "predictedSegmentDecrement = " << getPredictedSegmentDecrement()
      << std::endl
      << "maxSegmentsPerCell        = " << getMaxSegmentsPerCell() << std::endl
      << "maxSynapsesPerSegment     = " << getMaxSynapsesPerSegment()
      << std::endl;
}
