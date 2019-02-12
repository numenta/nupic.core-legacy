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

#include <climits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>


#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/utils/GroupBy.hpp>

using namespace std;
using namespace nupic;
using nupic::algorithms::temporal_memory::TemporalMemory;
using nupic::algorithms::connections::SynapseIdx;
using nupic::algorithms::connections::SynapseData;
using nupic::algorithms::connections::EPSILON;
using nupic::algorithms::connections::SegmentIdx;
using nupic::algorithms::connections::CellIdx;
using nupic::algorithms::connections::Connections;
using nupic::algorithms::connections::Segment;
using nupic::algorithms::connections::Permanence;
using nupic::algorithms::connections::Synapse;


static const UInt TM_VERSION = 2;

template <typename Iterator>
bool isSortedWithoutDuplicates(const Iterator begin, const Iterator end) {
    NTA_ASSERT(begin <= end) << "provide begin, and end";

    Iterator now = begin;
    while (now != end) {
      if (*now >= *(now+1)) {
        return false;
      }
      now++;
    }
  return true;
}

TemporalMemory::TemporalMemory() {}

TemporalMemory::TemporalMemory(
    vector<UInt> columnDimensions, UInt cellsPerColumn,
    UInt activationThreshold, Permanence initialPermanence,
    Permanence connectedPermanence, UInt minThreshold, UInt maxNewSynapseCount,
    Permanence permanenceIncrement, Permanence permanenceDecrement,
    Permanence predictedSegmentDecrement, Int seed, UInt maxSegmentsPerCell,
    UInt maxSynapsesPerSegment, bool checkInputs, UInt extra) {
  initialize(columnDimensions, cellsPerColumn, activationThreshold,
             initialPermanence, connectedPermanence, minThreshold,
             maxNewSynapseCount, permanenceIncrement, permanenceDecrement,
             predictedSegmentDecrement, seed, maxSegmentsPerCell,
             maxSynapsesPerSegment, checkInputs, extra);
}

TemporalMemory::~TemporalMemory() {}

void TemporalMemory::initialize(
    vector<UInt> columnDimensions, UInt cellsPerColumn,
    UInt activationThreshold, Permanence initialPermanence,
    Permanence connectedPermanence, UInt minThreshold, UInt maxNewSynapseCount,
    Permanence permanenceIncrement, Permanence permanenceDecrement,
    Permanence predictedSegmentDecrement, Int seed, UInt maxSegmentsPerCell,
    UInt maxSynapsesPerSegment, bool checkInputs, UInt extra) {
  // Validate all input parameters

  if (columnDimensions.size() <= 0) {
    NTA_THROW << "Number of column dimensions must be greater than 0";
  }

  if (cellsPerColumn <= 0) {
    NTA_THROW << "Number of cells per column must be greater than 0";
  }

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

  cellsPerColumn_ = cellsPerColumn;
  activationThreshold_ = activationThreshold;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  checkInputs_ = checkInputs;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;
  predictedSegmentDecrement_ = predictedSegmentDecrement;
  extra_ = extra;

  // Initialize member variables
  connections = Connections(numberOfColumns() * cellsPerColumn_);
  seed_((UInt64)(seed < 0 ? rand() : seed));

  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;
  iteration_ = 0;

  reset();
}

static CellIdx getLeastUsedCell(Random &rng, UInt column,
                                const Connections &connections,
                                UInt cellsPerColumn) {
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;

  UInt32 minNumSegments = UINT_MAX;
  UInt32 numTiedCells = 0;
  for (CellIdx cell = start; cell < end; cell++) {
    const UInt32 numSegments = connections.numSegments(cell);
    if (numSegments < minNumSegments) {
      minNumSegments = numSegments;
      numTiedCells = 1;
    } else if (numSegments == minNumSegments) {
      numTiedCells++;
    }
  }

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

static void adaptSegment(Connections &connections, Segment segment,
                         const vector<bool> &prevActiveCellsDense,
                         Permanence permanenceIncrement,
                         Permanence permanenceDecrement) {
  const vector<Synapse> &synapses = connections.synapsesForSegment(segment);

  for (SynapseIdx i = 0; i < synapses.size();) {
    const SynapseData &synapseData = connections.dataForSynapse(synapses[i]);

    Permanence permanence = synapseData.permanence;
    if (prevActiveCellsDense[synapseData.presynapticCell]) {
      permanence += permanenceIncrement;
    } else {
      permanence -= permanenceDecrement;
    }

    permanence = min(permanence, (Permanence)1.0);
    permanence = max(permanence, (Permanence)0.0);

    if (permanence < EPSILON) {
      connections.destroySynapse(synapses[i]);
      // Synapses vector is modified in-place, so don't update `i`.
    } else {
      connections.updateSynapsePermanence(synapses[i], permanence);
      i++;
    }
  }

  if (synapses.size() == 0) {
    connections.destroySegment(segment);
  }
}

static void destroyMinPermanenceSynapses(Connections &connections, Random &rng,
                                         Segment segment, Int nDestroy,
                                         const vector<CellIdx> &excludeCells) {
  // Don't destroy any cells that are in excludeCells.
  vector<Synapse> destroyCandidates;
  for (Synapse synapse : connections.synapsesForSegment(segment)) {
    const CellIdx presynapticCell =
        connections.dataForSynapse(synapse).presynapticCell;

    if (!std::binary_search(excludeCells.begin(), excludeCells.end(),
                            presynapticCell)) {
      destroyCandidates.push_back(synapse);
    }
  }

  // Find cells one at a time. This is slow, but this code rarely runs, and it
  // needs to work around floating point differences between environments.
  for (Int32 i = 0; i < nDestroy && !destroyCandidates.empty(); i++) {
    Permanence minPermanence = std::numeric_limits<Permanence>::max();
    vector<Synapse>::iterator minSynapse = destroyCandidates.end();

    for (auto synapse = destroyCandidates.begin();
         synapse != destroyCandidates.end(); synapse++) {
      const Permanence permanence =
          connections.dataForSynapse(*synapse).permanence;

      // Use special EPSILON logic to compensate for floating point
      // differences between C++ and other environments.
      if (permanence < minPermanence - EPSILON) {
        minSynapse = synapse;
        minPermanence = permanence;
      }
    }

    connections.destroySynapse(*minSynapse);
    destroyCandidates.erase(minSynapse);
  }
}

static void growSynapses(Connections &connections, Random &rng, Segment segment,
                         UInt32 nDesiredNewSynapses,
                         const vector<CellIdx> &prevWinnerCells,
                         Permanence initialPermanence,
                         UInt maxSynapsesPerSegment) {
  // It's possible to optimize this, swapping candidates to the end as
  // they're used. But this is awkward to mimic in other
  // implementations, especially because it requires iterating over
  // the existing synapses in a particular order.

  vector<CellIdx> candidates(prevWinnerCells.begin(), prevWinnerCells.end());
  NTA_ASSERT(std::is_sorted(candidates.begin(), candidates.end()));

  // Remove cells that are already synapsed on by this segment
  for (Synapse synapse : connections.synapsesForSegment(segment)) {
    CellIdx presynapticCell =
        connections.dataForSynapse(synapse).presynapticCell;
    auto ineligible =
        std::lower_bound(candidates.begin(), candidates.end(), presynapticCell);
    if (ineligible != candidates.end() && *ineligible == presynapticCell) {
      candidates.erase(ineligible);
    }
  }

  const UInt32 nActual =
      std::min(nDesiredNewSynapses, (UInt32)candidates.size());

  // Check if we're going to surpass the maximum number of synapses.
  const Int32 overrun =
      (connections.numSynapses(segment) + nActual - maxSynapsesPerSegment);
  if (overrun > 0) {
    destroyMinPermanenceSynapses(connections, rng, segment, overrun,
                                 prevWinnerCells);
  }

  // Recalculate in case we weren't able to destroy as many synapses as needed.
  const UInt32 nActualWithMax = std::min(
      nActual, maxSynapsesPerSegment - connections.numSynapses(segment));

  // Pick nActual cells randomly.
  for (UInt32 c = 0; c < nActualWithMax; c++) {
    UInt32 i = rng.getUInt32((UInt32)candidates.size());
    connections.createSynapse(segment, candidates[i], initialPermanence);
    candidates.erase(candidates.begin() + i);
  }
}

static void activatePredictedColumn(
    vector<CellIdx> &activeCells, vector<CellIdx> &winnerCells,
    Connections &connections, Random &rng,
    vector<Segment>::const_iterator columnActiveSegmentsBegin,
    vector<Segment>::const_iterator columnActiveSegmentsEnd,
    const vector<bool> &prevActiveCellsDense,
    const vector<CellIdx> &prevWinnerCells,
    const vector<UInt32> &numActivePotentialSynapsesForSegment,
    UInt maxNewSynapseCount, Permanence initialPermanence,
    Permanence permanenceIncrement, Permanence permanenceDecrement,
    UInt maxSynapsesPerSegment, bool learn) {
  auto activeSegment = columnActiveSegmentsBegin;
  do {
    const CellIdx cell = connections.cellForSegment(*activeSegment);
    activeCells.push_back(cell);
    winnerCells.push_back(cell);

    // This cell might have multiple active segments.
    do {
      if (learn) {
        adaptSegment(connections, *activeSegment, prevActiveCellsDense,
                     permanenceIncrement, permanenceDecrement);

        const Int32 nGrowDesired =
            maxNewSynapseCount -
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

static Segment createSegment(Connections &connections,
                             vector<UInt64> &lastUsedIterationForSegment,
                             CellIdx cell, UInt64 iteration,
                             UInt maxSegmentsPerCell) {
  while (connections.numSegments(cell) >= maxSegmentsPerCell) {
    const vector<Segment> &destroyCandidates =
        connections.segmentsForCell(cell);

    auto leastRecentlyUsedSegment =
        std::min_element(destroyCandidates.begin(), destroyCandidates.end(),
                         [&](Segment a, Segment b) {
                           return (lastUsedIterationForSegment[a] <
                                   lastUsedIterationForSegment[b]);
                         });

    connections.destroySegment(*leastRecentlyUsedSegment);
  }

  const Segment segment = connections.createSegment(cell);
  lastUsedIterationForSegment.resize(connections.segmentFlatListLength());
  lastUsedIterationForSegment[segment] = iteration;

  return segment;
}

static void
burstColumn(vector<CellIdx> &activeCells, vector<CellIdx> &winnerCells,
            Connections &connections, Random &rng,
            vector<UInt64> &lastUsedIterationForSegment, UInt column,
            vector<Segment>::const_iterator columnMatchingSegmentsBegin,
            vector<Segment>::const_iterator columnMatchingSegmentsEnd,
            const vector<bool> &prevActiveCellsDense,
            const vector<CellIdx> &prevWinnerCells,
            const vector<UInt32> &numActivePotentialSynapsesForSegment,
            UInt64 iteration, UInt cellsPerColumn, UInt maxNewSynapseCount,
            Permanence initialPermanence, Permanence permanenceIncrement,
            Permanence permanenceDecrement, UInt maxSegmentsPerCell,
            UInt maxSynapsesPerSegment, bool learn) {
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
          : getLeastUsedCell(rng, column, connections, cellsPerColumn);

  winnerCells.push_back(winnerCell);

  // Learn.
  if (learn) {
    if (bestMatchingSegment != columnMatchingSegmentsEnd) {
      // Learn on the best matching segment.
      adaptSegment(connections, *bestMatchingSegment, prevActiveCellsDense,
                   permanenceIncrement, permanenceDecrement);

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
            createSegment(connections, lastUsedIterationForSegment, winnerCell,
                          iteration, maxSegmentsPerCell);

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
    const vector<bool> &prevActiveCellsDense,
    Permanence predictedSegmentDecrement) {
  if (predictedSegmentDecrement > 0.0) {
    for (auto matchingSegment = columnMatchingSegmentsBegin;
         matchingSegment != columnMatchingSegmentsEnd; matchingSegment++) {
      adaptSegment(connections, *matchingSegment, prevActiveCellsDense,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void TemporalMemory::activateCells(const size_t activeColumnsSize,
                                   const UInt activeColumns[], bool learn) {
  if (checkInputs_ && activeColumnsSize > 0) {
    NTA_CHECK(isSortedWithoutDuplicates(activeColumns,
                                        activeColumns + activeColumnsSize-1))
        << "The activeColumns must be a sorted list of indices without "
           "duplicates.";
  }

  vector<bool> prevActiveCellsDense(numberOfCells() + extra_, false);
  for (CellIdx cell : activeCells_) {
    prevActiveCellsDense[cell] = true;
  }
  activeCells_.clear();

  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  const auto columnForSegment = [&](Segment segment) {
    return connections.cellForSegment(segment) / cellsPerColumn_;
  };

  for (auto &columnData : iterGroupBy( //TODO explain this
           activeColumns, activeColumns + activeColumnsSize, identity<UInt>,
           activeSegments_.begin(), activeSegments_.end(), columnForSegment,
           matchingSegments_.begin(), matchingSegments_.end(),
           columnForSegment)) {
    UInt column;
    const UInt *activeColumnsBegin;
    const UInt *activeColumnsEnd;
    vector<Segment>::const_iterator columnActiveSegmentsBegin,
        columnActiveSegmentsEnd, columnMatchingSegmentsBegin,
        columnMatchingSegmentsEnd;
    tie(column, activeColumnsBegin, activeColumnsEnd, columnActiveSegmentsBegin,
        columnActiveSegmentsEnd, columnMatchingSegmentsBegin,
        columnMatchingSegmentsEnd) = columnData;

    const bool isActiveColumn = activeColumnsBegin != activeColumnsEnd;
    if (isActiveColumn) {
      if (columnActiveSegmentsBegin != columnActiveSegmentsEnd) {
        activatePredictedColumn(
            activeCells_, winnerCells_, connections, rng_,
            columnActiveSegmentsBegin, columnActiveSegmentsEnd,
            prevActiveCellsDense, prevWinnerCells,
            numActivePotentialSynapsesForSegment_, maxNewSynapseCount_,
            initialPermanence_, permanenceIncrement_, permanenceDecrement_,
            maxSynapsesPerSegment_, learn);
      } else {
        burstColumn(activeCells_, winnerCells_, connections, rng_,
                    lastUsedIterationForSegment_, column,
                    columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
                    prevActiveCellsDense, prevWinnerCells,
                    numActivePotentialSynapsesForSegment_, iteration_,
                    cellsPerColumn_, maxNewSynapseCount_, initialPermanence_,
                    permanenceIncrement_, permanenceDecrement_,
                    maxSegmentsPerCell_, maxSynapsesPerSegment_, learn);
      }
    } else {
      if (learn) {
        punishPredictedColumn(connections, columnMatchingSegmentsBegin,
                              columnMatchingSegmentsEnd, prevActiveCellsDense,
                              predictedSegmentDecrement_);
      }
    }
  }
  segmentsValid_ = false;
}

void TemporalMemory::activateDendrites(bool learn,
                                       const vector<UInt> &extraActive,
                                       const vector<UInt> &extraWinners)
{
  if( segmentsValid_ )
    return;

  // Handle external predictive inputs.  extraActive & extraWinners default
  // values are `vector({ SENTINEL })`
  const auto SENTINEL = std::numeric_limits<UInt>::max();
  if( extra_ )
  {
    NTA_CHECK( extraActive.size()  != 1 || extraActive[0]  != SENTINEL )
        << "TM.ActivateDendrites() missing argument extraActive!";
    NTA_CHECK( extraWinners.size() != 1 || extraWinners[0] != SENTINEL )
        << "TM.ActivateDendrites() missing argument extraWinners!";

    for(const auto &active : extraActive) {
      NTA_ASSERT( active < extra_ );
      activeCells_.push_back( active + numberOfCells() );
    }
    for(const auto &winner : extraWinners) {
      NTA_ASSERT( winner < extra_ );
      winnerCells_.push_back( winner + numberOfCells() );
    }
  }
  else {
    NTA_CHECK( extraActive.size()  == 1 && extraActive[0]  == SENTINEL )
        << "External predictive inputs must be declared to TM constructor!";
    NTA_CHECK( extraWinners.size() == 1 && extraWinners[0] == SENTINEL )
        << "External predictive inputs must be declared to TM constructor!";
  }

  const UInt32 length = connections.segmentFlatListLength();

  numActiveConnectedSynapsesForSegment_.assign(length, 0);
  numActivePotentialSynapsesForSegment_.assign(length, 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment_,
                              numActivePotentialSynapsesForSegment_,
                              activeCells_, connectedPermanence_);

  // Active segments, connected synapses.
  activeSegments_.clear();
  for (Segment segment = 0;
       segment < numActiveConnectedSynapsesForSegment_.size(); segment++) {
    if (numActiveConnectedSynapsesForSegment_[segment] >=
        activationThreshold_) {
      activeSegments_.push_back(segment);
    }
  }
  std::sort(
      activeSegments_.begin(), activeSegments_.end(),
      [&](Segment a, Segment b) { return connections.compareSegments(a, b); });
  // Update segment bookkeeping.
  if (learn) {
    for (const auto &segment : activeSegments_) {
      lastUsedIterationForSegment_[segment] = iteration_;
    }
    iteration_++;
  }

  // Matching segments, potential synapses.
  matchingSegments_.clear();
  for (Segment segment = 0;
       segment < numActivePotentialSynapsesForSegment_.size(); segment++) {
    if (numActivePotentialSynapsesForSegment_[segment] >= minThreshold_) {
      matchingSegments_.push_back(segment);
    }
  }
  std::sort(
      matchingSegments_.begin(), matchingSegments_.end(),
      [&](Segment a, Segment b) { return connections.compareSegments(a, b); });

  segmentsValid_ = true;
}

void TemporalMemory::compute(const size_t activeColumnsSize,
                             const UInt activeColumns[], 
			     bool learn,
                             const vector<UInt> &extraActive,
                             const vector<UInt> &extraWinners) {

  activateDendrites(learn, extraActive, extraWinners);
  activateCells(activeColumnsSize, activeColumns, learn);
}

void TemporalMemory::reset(void) {
  activeCells_.clear();
  winnerCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
  segmentsValid_ = false;
}

// ==============================
//  Helper functions
// ==============================

Segment TemporalMemory::createSegment(CellIdx cell) {
  return ::createSegment(connections, lastUsedIterationForSegment_, cell,
                         iteration_, maxSegmentsPerCell_);
}

UInt TemporalMemory::columnForCell(const CellIdx cell) const {

  NTA_ASSERT(cell < numberOfCells());
  return cell / cellsPerColumn_;
}

vector<CellIdx> TemporalMemory::cellsForColumn(Int column) {
  const CellIdx start = cellsPerColumn_ * column;
  const CellIdx end = start + cellsPerColumn_;

  vector<CellIdx> cellsInColumn;
  for (CellIdx i = start; i < end; i++) {
    cellsInColumn.push_back(i);
  }

  return cellsInColumn;
}

UInt TemporalMemory::numberOfCells(void) const { return connections.numCells(); }

vector<CellIdx> TemporalMemory::getActiveCells() const { return activeCells_; }

vector<CellIdx> TemporalMemory::getPredictiveCells() const {

  NTA_CHECK( segmentsValid_ )
    << "Call TM.activateDendrites() before TM.getPredictiveCells()!";

  vector<CellIdx> predictiveCells;

  for (auto segment = activeSegments_.cbegin(); segment != activeSegments_.cend();
       segment++) {
    CellIdx cell = connections.cellForSegment(*segment);
    if (segment == activeSegments_.begin() || cell != predictiveCells.back()) {
      predictiveCells.push_back(cell);
    }
  }

  return predictiveCells;
}

vector<CellIdx> TemporalMemory::getWinnerCells() const { return winnerCells_; }

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

UInt TemporalMemory::numberOfColumns() const { return numColumns_; }

bool TemporalMemory::_validateCell(CellIdx cell) const
{
  if (cell < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell;
  return false;
}

vector<UInt> TemporalMemory::getColumnDimensions() const
{
  return columnDimensions_;
}

UInt TemporalMemory::getCellsPerColumn() const { return cellsPerColumn_; }

UInt TemporalMemory::getActivationThreshold() const {
  return activationThreshold_;
}

void TemporalMemory::setActivationThreshold(UInt activationThreshold) {
  activationThreshold_ = activationThreshold;
}

Permanence TemporalMemory::getInitialPermanence() const {
  return initialPermanence_;
}

void TemporalMemory::setInitialPermanence(Permanence initialPermanence) {
  initialPermanence_ = initialPermanence;
}

Permanence TemporalMemory::getConnectedPermanence() const {
  return connectedPermanence_;
}

void TemporalMemory::setConnectedPermanence(Permanence connectedPermanence) {
  connectedPermanence_ = connectedPermanence;
}

UInt TemporalMemory::getMinThreshold() const { return minThreshold_; }

void TemporalMemory::setMinThreshold(UInt minThreshold) {
  minThreshold_ = minThreshold;
}

UInt TemporalMemory::getMaxNewSynapseCount() const {
  return maxNewSynapseCount_;
}

void TemporalMemory::setMaxNewSynapseCount(UInt maxNewSynapseCount) {
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

UInt TemporalMemory::getMaxSegmentsPerCell() const {
  return maxSegmentsPerCell_;
}

UInt TemporalMemory::getMaxSynapsesPerSegment() const {
  return maxSynapsesPerSegment_;
}

UInt TemporalMemory::version() const { return TM_VERSION; }

/**
 * Create a RNG with given seed
 */
void TemporalMemory::seed_(UInt64 seed) { rng_ = Random(seed); }

size_t TemporalMemory::persistentSize() const {
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}

template <typename FloatType>
static void saveFloat_(ostream &outStream, FloatType v) {
  outStream << std::setprecision(std::numeric_limits<FloatType>::max_digits10)
            << v << " ";
}

void TemporalMemory::save(ostream &outStream) const {
  // Write a starting marker and version.
  outStream << "TemporalMemory" << endl;
  outStream << TM_VERSION << endl;

  outStream << numColumns_ << " " << cellsPerColumn_ << " "
            << activationThreshold_ << " ";

  saveFloat_(outStream, initialPermanence_);
  saveFloat_(outStream, connectedPermanence_);

  outStream << minThreshold_ << " " << maxNewSynapseCount_ << " "
            << checkInputs_ << " ";

  saveFloat_(outStream, permanenceIncrement_);
  saveFloat_(outStream, permanenceDecrement_);
  saveFloat_(outStream, predictedSegmentDecrement_);

  outStream << extra_ << " ";
  outStream << maxSegmentsPerCell_ << " " << maxSynapsesPerSegment_ << " "
            << iteration_ << " ";

  outStream << endl;

  connections.save(outStream);
  outStream << endl;

  outStream << rng_ << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto &elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << activeCells_.size() << " ";
  for (CellIdx cell : activeCells_) {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << winnerCells_.size() << " ";
  for (CellIdx cell : winnerCells_) {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << segmentsValid_ << " ";
  outStream << activeSegments_.size() << " ";
  for (Segment segment : activeSegments_) {
    const CellIdx cell = connections.cellForSegment(segment);
    const vector<Segment> &segments = connections.segmentsForCell(cell);

    SegmentIdx idx = (SegmentIdx)std::distance(
        segments.begin(), std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActiveConnectedSynapsesForSegment_[segment] << " ";
  }
  outStream << endl;

  outStream << matchingSegments_.size() << " ";
  for (Segment segment : matchingSegments_) {
    const CellIdx cell = connections.cellForSegment(segment);
    const vector<Segment> &segments = connections.segmentsForCell(cell);

    SegmentIdx idx = (SegmentIdx)std::distance(
        segments.begin(), std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActivePotentialSynapsesForSegment_[segment] << " ";
  }
  outStream << endl;

  outStream << "~TemporalMemory" << endl;
}



void TemporalMemory::load(istream &inStream) {
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "TemporalMemory");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= TM_VERSION);

  // Retrieve simple variables
  inStream >> numColumns_ >> cellsPerColumn_ >> activationThreshold_ >>
      initialPermanence_ >> connectedPermanence_ >> minThreshold_ >>
      maxNewSynapseCount_ >> checkInputs_ >> permanenceIncrement_ >>
      permanenceDecrement_ >> predictedSegmentDecrement_ >> extra_ >>
      maxSegmentsPerCell_ >> maxSynapsesPerSegment_ >> iteration_;

  connections.load(inStream);

  numActiveConnectedSynapsesForSegment_.assign(
      connections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForSegment_.assign(
      connections.segmentFlatListLength(), 0);

  inStream >> rng_;

  // Retrieve vectors.
  UInt numColumnDimensions;
  inStream >> numColumnDimensions;
  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++) {
    inStream >> columnDimensions_[i];
  }

  UInt numActiveCells;
  inStream >> numActiveCells;
  for (UInt i = 0; i < numActiveCells; i++) {
    CellIdx cell;
    inStream >> cell;
    activeCells_.push_back(cell);
  }

  if (version < 2) {
    UInt numPredictiveCells;
    inStream >> numPredictiveCells;
    for (UInt i = 0; i < numPredictiveCells; i++) {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++) {
    CellIdx cell;
    inStream >> cell;
    winnerCells_.push_back(cell);
  }

  inStream >> segmentsValid_;
  UInt numActiveSegments;
  inStream >> numActiveSegments;
  activeSegments_.resize(numActiveSegments);
  for (UInt i = 0; i < numActiveSegments; i++) {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = connections.getSegment(cellIdx, idx);
    activeSegments_[i] = segment;

    if (version < 2) {
      numActiveConnectedSynapsesForSegment_[segment] = 0; // Unknown
    } else {
      inStream >> numActiveConnectedSynapsesForSegment_[segment];
    }
  }

  UInt numMatchingSegments;
  inStream >> numMatchingSegments;
  matchingSegments_.resize(numMatchingSegments);
  for (UInt i = 0; i < numMatchingSegments; i++) {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = connections.getSegment(cellIdx, idx);
    matchingSegments_[i] = segment;

    if (version < 2) {
      numActivePotentialSynapsesForSegment_[segment] = 0; // Unknown
    } else {
      inStream >> numActivePotentialSynapsesForSegment_[segment];
    }
  }

  if (version < 2) {
    UInt numMatchingCells;
    inStream >> numMatchingCells;
    for (UInt i = 0; i < numMatchingCells; i++) {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  lastUsedIterationForSegment_.resize(connections.segmentFlatListLength());

  inStream >> marker;
  NTA_CHECK(marker == "~TemporalMemory");
}

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

bool TemporalMemory::operator==(const TemporalMemory &other) {
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
      iteration_ != other.iteration_) {
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

bool TemporalMemory::operator!=(const TemporalMemory &other) {
  return !(*this == other);
}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main TM creation parameters
void TemporalMemory::printParameters() {
  std::cout << "------------CPP TemporalMemory Parameters ------------------\n";
  std::cout
      << "version                   = " << TM_VERSION << std::endl
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

void TemporalMemory::printState(vector<UInt> &state) {
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void TemporalMemory::printState(vector<Real> &state) {
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::printf("%6.3f ", state[i]);
  }
  std::cout << "]\n";
}
