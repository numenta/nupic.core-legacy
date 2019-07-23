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
    UInt externalPredictiveInputs) {
  initialize(columnDimensions, cellsPerColumn, activationThreshold,
             initialPermanence, connectedPermanence, minThreshold,
             maxNewSynapseCount, permanenceIncrement, permanenceDecrement,
             predictedSegmentDecrement, seed, maxSegmentsPerCell,
             maxSynapsesPerSegment, checkInputs, externalPredictiveInputs);
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
    UInt externalPredictiveInputs) {

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
  connections = Connections(static_cast<CellIdx>(numberOfColumns() * cellsPerColumn_), connectedPermanence_);
  rng_ = Random(seed);

  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;

  reset();
}

///*
static CellIdx getLeastUsedCell(Random &rng, 
		                const UInt column, //TODO remove static methods, use private instead
                                const Connections &connections,
                                const UInt cellsPerColumn) {
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;

  size_t minNumSegments = std::numeric_limits<CellIdx>::max();
  UInt32 numTiedCells = 0u;
  //for all cells in a mini-column
  for (CellIdx cell = start; cell < end; cell++) {
    const size_t numSegments = connections.numSegments(cell);
    //..find a cell with least segments
    if (numSegments < minNumSegments) {
      minNumSegments = numSegments;
      numTiedCells = 1u;
    //..and how many of the cells have only these min segments? number of weakest
    } else if (numSegments == minNumSegments) {
      numTiedCells++;
    }
  }

  //randomly select one of the tie-d cells from the losers
  const UInt32 tieWinnerIndex = rng.getUInt32(numTiedCells);
  UInt32 tieIndex = 0;
  for (CellIdx cell = start; cell < end; cell++) {
    if (connections.numSegments(cell) == minNumSegments) {
      if (tieIndex == tieWinnerIndex) {
        return cell;
      } else {
        tieIndex++;
      }
    }
  }

  NTA_THROW << "getLeastUsedCell failed to find a cell";
}
//*/

static void growSynapses(Connections &connections, 
		         Random &rng, 
			 const Segment& segment,
                         const SynapseIdx nDesiredNewSynapses,
                         const vector<CellIdx> &prevWinnerCells,
                         const Permanence initialPermanence,
                         const SynapseIdx maxSynapsesPerSegment) {
  // It's possible to optimize this, swapping candidates to the end as
  // they're used. But this is awkward to mimic in other
  // implementations, especially because it requires iterating over
  // the existing synapses in a particular order.

  vector<CellIdx> candidates(prevWinnerCells.begin(), prevWinnerCells.end());
  NTA_ASSERT(std::is_sorted(candidates.begin(), candidates.end()));

  // Skip cells that are already synapsed on by this segment
  // Biological motivation (?):
  // There are structural constraints on the shapes of axons & synapses 
  // which prevent a large number duplicate of connections.
  //
  // It's important to prevent cells from growing duplicate synapses onto a segment, 
  // because otherwise a strong input would be sampled many times and grow many synapses.
  // That would give such input a stronger connection. 
  // Synapses are supposed to have binary effects (0 or 1) but duplicate synapses give 
  // them (synapses 0/1) varying levels of strength.
  for (const Synapse& synapse : connections.synapsesForSegment(segment)) {
    const CellIdx presynapticCell = connections.dataForSynapse(synapse).presynapticCell;
    const auto already = std::lower_bound(candidates.cbegin(), candidates.cend(), presynapticCell);
    if (already != candidates.cend() && *already == presynapticCell) {
      candidates.erase(already);
    }
  }

  const size_t nActual = std::min(static_cast<size_t>(nDesiredNewSynapses), candidates.size());

  // Check if we're going to surpass the maximum number of synapses.
  Int overrun = static_cast<Int>(connections.numSynapses(segment) + nActual - maxSynapsesPerSegment);
  if (overrun > 0) {
    connections.destroyMinPermanenceSynapses(segment, static_cast<Int>(overrun), prevWinnerCells);
  }

  // Recalculate in case we weren't able to destroy as many synapses as needed.
  const size_t nActualWithMax = std::min(nActual, static_cast<size_t>(maxSynapsesPerSegment) - connections.numSynapses(segment));

  // Pick nActual cells randomly.
  for (const auto syn : rng.sample(candidates, static_cast<UInt>(nActualWithMax))) {
    connections.createSynapse(segment, syn, initialPermanence); //TODO createSynapse consider creating a vector of new synapses at once?
  }
}

static void activatePredictedColumn(
    vector<CellIdx> &activeCells, 
    vector<CellIdx> &winnerCells,
    Connections &connections, 
    Random &rng,
    vector<Segment>::const_iterator columnActiveSegmentsBegin,
    vector<Segment>::const_iterator columnActiveSegmentsEnd,
    const SDR &prevActiveCells,
    const vector<CellIdx> &prevWinnerCells,
    const vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
    const UInt maxNewSynapseCount, 
    const Permanence initialPermanence,
    const Permanence permanenceIncrement, 
    const Permanence permanenceDecrement,
    const SynapseIdx maxSynapsesPerSegment, 
    const bool learn) {
  auto activeSegment = columnActiveSegmentsBegin;
  do {
    const CellIdx cell = connections.cellForSegment(*activeSegment);
    activeCells.push_back(cell);
    winnerCells.push_back(cell);

    // This cell might have multiple active segments.
    do {
      if (learn) { 
        connections.adaptSegment(*activeSegment, prevActiveCells,
                     permanenceIncrement, permanenceDecrement, true);

        const Int32 nGrowDesired =
            static_cast<Int32>(maxNewSynapseCount) -
            numActivePotentialSynapsesForSegment[*activeSegment];
        if (nGrowDesired > 0) {
          growSynapses(connections, rng, *activeSegment, nGrowDesired,
                       prevWinnerCells, initialPermanence,
                       maxSynapsesPerSegment);
        }
      }
    } while (++activeSegment != columnActiveSegmentsEnd &&
             connections.cellForSegment(*activeSegment) == cell);
  } while (activeSegment != columnActiveSegmentsEnd);
}


static void
burstColumn(vector<CellIdx> &activeCells, 
            vector<CellIdx> &winnerCells,
            Connections &connections, 
            Random &rng,
            UInt column,
            vector<Segment>::const_iterator columnMatchingSegmentsBegin,
            vector<Segment>::const_iterator columnMatchingSegmentsEnd,
            const SDR &prevActiveCells,
            const vector<CellIdx> &prevWinnerCells,
            const vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
            CellIdx cellsPerColumn, 
            UInt maxNewSynapseCount,
            const Permanence initialPermanence, 
            const Permanence permanenceIncrement,
            const Permanence permanenceDecrement, 
            const SegmentIdx maxSegmentsPerCell,
            const SynapseIdx maxSynapsesPerSegment, 
            const bool learn) {
  // Calculate the active cells.
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++) {
    activeCells.push_back(cell);
  }

  const auto bestMatchingSegment =
      std::max_element(columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
                       [&](Segment a, Segment b) {
                         return (numActivePotentialSynapsesForSegment[a] <
                                 numActivePotentialSynapsesForSegment[b]);
                       });

  const CellIdx winnerCell =
      (bestMatchingSegment != columnMatchingSegmentsEnd)
          ? connections.cellForSegment(*bestMatchingSegment)
          : getLeastUsedCell(rng, column, connections, cellsPerColumn); //TODO replace (with random?) this is extremely costly, removing makes TM 6x faster!

  winnerCells.push_back(winnerCell);

  // Learn.
  if (learn) {
    if (bestMatchingSegment != columnMatchingSegmentsEnd) {
      // Learn on the best matching segment.
      connections.adaptSegment(*bestMatchingSegment, prevActiveCells,
                   permanenceIncrement, permanenceDecrement, true);

      const Int32 nGrowDesired =
          maxNewSynapseCount -
          numActivePotentialSynapsesForSegment[*bestMatchingSegment];
      if (nGrowDesired > 0) {
        growSynapses(connections, rng, *bestMatchingSegment, nGrowDesired,
                     prevWinnerCells, initialPermanence, maxSynapsesPerSegment);
      }
    } else {
      // No matching segments.
      // Grow a new segment and learn on it.

      // Don't grow a segment that will never match.
      const UInt32 nGrowExact =
          std::min(maxNewSynapseCount, (UInt32)prevWinnerCells.size());
      if (nGrowExact > 0) {
        const Segment segment =
            connections.createSegment(winnerCell, maxSegmentsPerCell);

        growSynapses(connections, rng, segment, nGrowExact, prevWinnerCells,
                     initialPermanence, maxSynapsesPerSegment);
        NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
      }
    }
  }
}

static void punishPredictedColumn(
    Connections &connections,
    vector<Segment>::const_iterator columnMatchingSegmentsBegin,
    vector<Segment>::const_iterator columnMatchingSegmentsEnd,
    const SDR &prevActiveCells,
    Permanence predictedSegmentDecrement) {
  if (predictedSegmentDecrement > 0.0) {
    for (auto matchingSegment = columnMatchingSegmentsBegin;
         matchingSegment != columnMatchingSegmentsEnd; matchingSegment++) {
      connections.adaptSegment(*matchingSegment, prevActiveCells,
                   -predictedSegmentDecrement, 0.0, true);
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
    //   get its matching segs   ( >= mmm
    std::tie(column, 
             activeColumnsBegin, activeColumnsEnd, 
             columnActiveSegmentsBegin, columnActiveSegmentsEnd, 
             columnMatchingSegmentsBegin, columnMatchingSegmentsEnd
	) = columnData;

    const bool isActiveColumn = activeColumnsBegin != activeColumnsEnd;
    if (isActiveColumn) { //current active column...
      if (columnActiveSegmentsBegin != columnActiveSegmentsEnd) {
	//...was also predicted -> learn :o)
        activatePredictedColumn(
            activeCells_, winnerCells_, connections, rng_,
            columnActiveSegmentsBegin, columnActiveSegmentsEnd,
            prevActiveCells, prevWinnerCells,
            numActivePotentialSynapsesForSegment_, maxNewSynapseCount_,
            initialPermanence_, permanenceIncrement_, permanenceDecrement_,
            maxSynapsesPerSegment_, learn);
      } else {
	//...has not been predicted -> 
        burstColumn(activeCells_, winnerCells_, connections, rng_,
                    column,
                    columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
                    prevActiveCells, prevWinnerCells,
                    numActivePotentialSynapsesForSegment_,
                    cellsPerColumn_, maxNewSynapseCount_, initialPermanence_,
                    permanenceIncrement_, permanenceDecrement_,
                    maxSegmentsPerCell_, maxSynapsesPerSegment_, learn);
      }

    } else { // predicted but not active column -> unlearn
      if (learn) {
        punishPredictedColumn(connections, columnMatchingSegmentsBegin,
                              columnMatchingSegmentsEnd, prevActiveCells,
                              predictedSegmentDecrement_);
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

  numActiveConnectedSynapsesForSegment_.assign(length, 0);
  numActivePotentialSynapsesForSegment_.assign(length, 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment_,
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
      connections.dataForSegment(segment).lastUsed = connections.iteration(); //TODO the destroySegments based on LRU is expensive. Better random? or "energy" based on sum permanences?
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

  // Update Anomaly Metric.  The anomaly is the percent of active columns that
  // were not predicted.
  anomaly_ = computeRawAnomalyScore(
                activeColumns,
                cellsToColumns( getPredictiveCells() ));
  // TODO: Update mean & standard deviation of anomaly here.

  activateCells(activeColumns, learn);
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
  anomaly_ = -1.0f;
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


vector<CellIdx> TemporalMemory::cellsForColumn(CellIdx column) { 
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

  auto& predictiveCells = predictive.getSparse();

  for (auto segment = activeSegments_.cbegin(); segment != activeSegments_.cend();
       segment++) {
    const CellIdx cell = connections.cellForSegment(*segment);
    if (segment == activeSegments_.begin() || cell != predictiveCells.back()) {
      predictiveCells.push_back(cell);
    }
  }

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
      anomaly_ != other.anomaly_ ) {
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
