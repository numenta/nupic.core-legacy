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

#include <cstring>
#include <climits>
#include <iomanip>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/utils/GroupBy.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

static const Permanence EPSILON = 0.000001;
static const UInt TM_VERSION = 2;



template<typename Iterator>
bool isSortedWithoutDuplicates(Iterator begin, Iterator end)
{
  if (std::distance(begin, end) >= 2)
  {
    Iterator now = begin;
    Iterator next = begin + 1;
    while (next != end)
    {
      if (*now >= *next)
      {
        return false;
      }

      now = next++;
    }
  }

  return true;
}


TemporalMemory::TemporalMemory()
{
}

TemporalMemory::TemporalMemory(
  vector<UInt> columnDimensions,
  UInt cellsPerColumn,
  UInt activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  UInt minThreshold,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Permanence predictedSegmentDecrement,
  Int seed,
  UInt maxSegmentsPerCell,
  UInt maxSynapsesPerSegment)
{
  initialize(
    columnDimensions,
    cellsPerColumn,
    activationThreshold,
    initialPermanence,
    connectedPermanence,
    minThreshold,
    maxNewSynapseCount,
    permanenceIncrement,
    permanenceDecrement,
    predictedSegmentDecrement,
    seed,
    maxSegmentsPerCell,
    maxSynapsesPerSegment);
}

TemporalMemory::~TemporalMemory()
{
}

void TemporalMemory::initialize(
  vector<UInt> columnDimensions,
  UInt cellsPerColumn,
  UInt activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  UInt minThreshold,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Permanence predictedSegmentDecrement,
  Int seed,
  UInt maxSegmentsPerCell,
  UInt maxSynapsesPerSegment)
{
  // Validate all input parameters

  if (columnDimensions.size() <= 0)
  {
    NTA_THROW << "Number of column dimensions must be greater than 0";
  }

  if (cellsPerColumn <= 0)
  {
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
  for (auto & columnDimension : columnDimensions)
  {
    numColumns_ *= columnDimension;
    columnDimensions_.push_back(columnDimension);
  }

  cellsPerColumn_ = cellsPerColumn;
  activationThreshold_ = activationThreshold;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;
  predictedSegmentDecrement_ = predictedSegmentDecrement;

  // Initialize member variables
  connections = Connections(numberOfColumns() * cellsPerColumn_);
  seed_((UInt64)(seed < 0 ? rand() : seed));

  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;
  iteration_ = 0;

  activeCells_.clear();
  winnerCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
}

static CellIdx getLeastUsedCell(
  Random& rng,
  UInt column,
  const Connections& connections,
  UInt cellsPerColumn)
{
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;

  UInt32 minNumSegments = UINT_MAX;
  UInt32 numTiedCells = 0;
  for (CellIdx cell = start; cell < end; cell++)
  {
    const UInt32 numSegments = connections.numSegments(cell);
    if (numSegments < minNumSegments)
    {
      minNumSegments = numSegments;
      numTiedCells = 1;
    }
    else if (numSegments == minNumSegments)
    {
      numTiedCells++;
    }
  }

  const UInt32 tieWinnerIndex = rng.getUInt32(numTiedCells);

  UInt32 tieIndex = 0;
  for (CellIdx cell = start; cell < end; cell++)
  {
    if (connections.numSegments(cell) == minNumSegments)
    {
      if (tieIndex == tieWinnerIndex)
      {
        return cell;
      }
      else
      {
        tieIndex++;
      }
    }
  }

  NTA_THROW << "getLeastUsedCell failed to find a cell";
}

static void adaptSegment(
  Connections& connections,
  Segment segment,
  const vector<bool>& prevActiveCellsDense,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  const vector<Synapse>& synapses = connections.synapsesForSegment(segment);

  for (SynapseIdx i = 0; i < synapses.size();)
  {
    const SynapseData& synapseData = connections.dataForSynapse(synapses[i]);

    NTA_ASSERT(synapseData.presynapticCell < connections.numCells());

    Permanence permanence = synapseData.permanence;
    if (prevActiveCellsDense[synapseData.presynapticCell])
    {
      permanence += permanenceIncrement;
    }
    else
    {
      permanence -= permanenceDecrement;
    }

    permanence = min(permanence, (Permanence)1.0);
    permanence = max(permanence, (Permanence)0.0);

    if (permanence < EPSILON)
    {
      connections.destroySynapse(synapses[i]);
      // Synapses vector is modified in-place, so don't update `i`.
    }
    else
    {
      connections.updateSynapsePermanence(synapses[i], permanence);
      i++;
    }
  }

  if (synapses.size() == 0)
  {
    connections.destroySegment(segment);
  }
}

static void destroyMinPermanenceSynapses(
  Connections& connections,
  Random& rng,
  Segment segment,
  Int nDestroy,
  const vector<CellIdx>& excludeCells)
{
  // Don't destroy any cells that are in excludeCells.
  vector<Synapse> destroyCandidates;
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    const CellIdx presynapticCell =
      connections.dataForSynapse(synapse).presynapticCell;

    if (!std::binary_search(excludeCells.begin(), excludeCells.end(),
                            presynapticCell))
    {
      destroyCandidates.push_back(synapse);
    }
  }

  // Find cells one at a time. This is slow, but this code rarely runs, and it
  // needs to work around floating point differences between environments.
  for (Int32 i = 0; i < nDestroy && !destroyCandidates.empty(); i++)
  {
    Permanence minPermanence = std::numeric_limits<Permanence>::max();
    vector<Synapse>::iterator minSynapse = destroyCandidates.end();

    for (auto synapse = destroyCandidates.begin();
         synapse != destroyCandidates.end();
         synapse++)
    {
      const Permanence permanence =
        connections.dataForSynapse(*synapse).permanence;

      // Use special EPSILON logic to compensate for floating point
      // differences between C++ and other environments.
      if (permanence < minPermanence - EPSILON)
      {
        minSynapse = synapse;
        minPermanence = permanence;
      }
    }

    connections.destroySynapse(*minSynapse);
    destroyCandidates.erase(minSynapse);
  }
}

static void growSynapses(
  Connections& connections,
  Random& rng,
  Segment segment,
  UInt32 nDesiredNewSynapses,
  const vector<CellIdx>& prevWinnerCells,
  Permanence initialPermanence,
  UInt maxSynapsesPerSegment)
{
  // It's possible to optimize this, swapping candidates to the end as
  // they're used. But this is awkward to mimic in other
  // implementations, especially because it requires iterating over
  // the existing synapses in a particular order.

  vector<CellIdx> candidates(prevWinnerCells.begin(), prevWinnerCells.end());
  NTA_ASSERT(std::is_sorted(candidates.begin(), candidates.end()));

  // Remove cells that are already synapsed on by this segment
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    CellIdx presynapticCell =
      connections.dataForSynapse(synapse).presynapticCell;
    auto ineligible = std::lower_bound(candidates.begin(), candidates.end(),
                                       presynapticCell);
    if (ineligible != candidates.end() && *ineligible == presynapticCell)
    {
      candidates.erase(ineligible);
    }
  }

  const UInt32 nActual = std::min(nDesiredNewSynapses,
                                  (UInt32)candidates.size());

  // Check if we're going to surpass the maximum number of synapses.
  const Int32 overrun = (connections.numSynapses(segment) +
                         nActual - maxSynapsesPerSegment);
  if (overrun > 0)
  {
    destroyMinPermanenceSynapses(connections, rng, segment, overrun,
                                 prevWinnerCells);
  }

  // Recalculate in case we weren't able to destroy as many synapses as needed.
  const UInt32 nActualWithMax = std::min(nActual,
                                         maxSynapsesPerSegment -
                                         connections.numSynapses(segment));

  // Pick nActual cells randomly.
  for (UInt32 c = 0; c < nActualWithMax; c++)
  {
    size_t i = rng.getUInt32(candidates.size());
    connections.createSynapse(segment, candidates[i], initialPermanence);
    candidates.erase(candidates.begin() + i);
  }
}

static void activatePredictedColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  Random& rng,
  vector<Segment>::const_iterator columnActiveSegmentsBegin,
  vector<Segment>::const_iterator columnActiveSegmentsEnd,
  const vector<bool>& prevActiveCellsDense,
  const vector<CellIdx>& prevWinnerCells,
  const vector<UInt32>& numActivePotentialSynapsesForSegment,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  UInt maxSynapsesPerSegment,
  bool learn)
{
  auto activeSegment = columnActiveSegmentsBegin;
  do
  {
    const CellIdx cell = connections.cellForSegment(*activeSegment);
    activeCells.push_back(cell);
    winnerCells.push_back(cell);

    // This cell might have multiple active segments.
    do
    {
      if (learn)
      {
        adaptSegment(connections,
                     *activeSegment,
                     prevActiveCellsDense,
                     permanenceIncrement, permanenceDecrement);

        const Int32 nGrowDesired = maxNewSynapseCount -
          numActivePotentialSynapsesForSegment[*activeSegment];
        if (nGrowDesired > 0)
        {
          growSynapses(connections, rng,
                       *activeSegment, nGrowDesired,
                       prevWinnerCells,
                       initialPermanence, maxSynapsesPerSegment);
        }
      }
    } while (++activeSegment != columnActiveSegmentsEnd &&
             connections.cellForSegment(*activeSegment) == cell);
  } while (activeSegment != columnActiveSegmentsEnd);
}

static Segment createSegment(
  Connections& connections,
  vector<UInt64>& lastUsedIterationForSegment,
  CellIdx cell,
  UInt64 iteration,
  UInt maxSegmentsPerCell)
{
  while (connections.numSegments(cell) >= maxSegmentsPerCell)
  {
    const vector<Segment>& destroyCandidates =
      connections.segmentsForCell(cell);

    auto leastRecentlyUsedSegment = std::min_element(
      destroyCandidates.begin(), destroyCandidates.end(),
      [&](Segment a, Segment b)
      {
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

static void burstColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  Random& rng,
  vector<UInt64>& lastUsedIterationForSegment,
  UInt column,
  vector<Segment>::const_iterator columnMatchingSegmentsBegin,
  vector<Segment>::const_iterator columnMatchingSegmentsEnd,
  const vector<bool>& prevActiveCellsDense,
  const vector<CellIdx>& prevWinnerCells,
  const vector<UInt32>& numActivePotentialSynapsesForSegment,
  UInt64 iteration,
  UInt cellsPerColumn,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  UInt maxSegmentsPerCell,
  UInt maxSynapsesPerSegment,
  bool learn)
{
  // Calculate the active cells.
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++)
  {
    activeCells.push_back(cell);
  }

  const auto bestMatchingSegment = std::max_element(
    columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
    [&](Segment a, Segment b)
    {
      return (numActivePotentialSynapsesForSegment[a] <
              numActivePotentialSynapsesForSegment[b]);
    });

  const CellIdx winnerCell = (bestMatchingSegment != columnMatchingSegmentsEnd)
    ? connections.cellForSegment(*bestMatchingSegment)
    : getLeastUsedCell(rng, column, connections, cellsPerColumn);

  winnerCells.push_back(winnerCell);

  // Learn.
  if (learn)
  {
    if (bestMatchingSegment != columnMatchingSegmentsEnd)
    {
      // Learn on the best matching segment.
      adaptSegment(connections,
                   *bestMatchingSegment,
                   prevActiveCellsDense,
                   permanenceIncrement, permanenceDecrement);

      const Int32 nGrowDesired = maxNewSynapseCount -
        numActivePotentialSynapsesForSegment[*bestMatchingSegment];
      if (nGrowDesired > 0)
      {
        growSynapses(connections, rng,
                     *bestMatchingSegment, nGrowDesired,
                     prevWinnerCells,
                     initialPermanence, maxSynapsesPerSegment);
      }
    }
    else
    {
      // No matching segments.
      // Grow a new segment and learn on it.

      // Don't grow a segment that will never match.
      const UInt32 nGrowExact = std::min(maxNewSynapseCount,
                                         (UInt32)prevWinnerCells.size());
      if (nGrowExact > 0)
      {
        const Segment segment =
          createSegment(connections, lastUsedIterationForSegment,
                        winnerCell, iteration, maxSegmentsPerCell);

        growSynapses(connections, rng,
                     segment, nGrowExact,
                     prevWinnerCells,
                     initialPermanence, maxSynapsesPerSegment);
        NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
      }
    }
  }
}

static void punishPredictedColumn(
  Connections& connections,
  vector<Segment>::const_iterator columnMatchingSegmentsBegin,
  vector<Segment>::const_iterator columnMatchingSegmentsEnd,
  const vector<bool>& prevActiveCellsDense,
  Permanence predictedSegmentDecrement)
{
  if (predictedSegmentDecrement > 0.0)
  {
    for (auto matchingSegment = columnMatchingSegmentsBegin;
         matchingSegment != columnMatchingSegmentsEnd; matchingSegment++)
    {
      adaptSegment(connections, *matchingSegment, prevActiveCellsDense,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void TemporalMemory::activateCells(
  size_t activeColumnsSize,
  const UInt activeColumns[],
  bool learn)
{
  NTA_CHECK(isSortedWithoutDuplicates(activeColumns,
                                      activeColumns + activeColumnsSize))
    << "The activeColumns must be a sorted list of indices without duplicates.";

  vector<bool> prevActiveCellsDense(numberOfCells(), false);
  for (CellIdx cell : activeCells_)
  {
    prevActiveCellsDense[cell] = true;
  }
  activeCells_.clear();

  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  const auto columnForSegment = [&](Segment segment)
    { return connections.cellForSegment(segment) / cellsPerColumn_; };

  for (auto& columnData : iterGroupBy(
         activeColumns, activeColumns + activeColumnsSize, identity<UInt>,
         activeSegments_.begin(), activeSegments_.end(), columnForSegment,
         matchingSegments_.begin(), matchingSegments_.end(), columnForSegment))
  {
    UInt column;
    const UInt* activeColumnsBegin;
    const UInt* activeColumnsEnd;
    vector<Segment>::const_iterator
      columnActiveSegmentsBegin, columnActiveSegmentsEnd,
      columnMatchingSegmentsBegin, columnMatchingSegmentsEnd;
    tie(column,
        activeColumnsBegin, activeColumnsEnd,
        columnActiveSegmentsBegin, columnActiveSegmentsEnd,
        columnMatchingSegmentsBegin, columnMatchingSegmentsEnd) = columnData;

    const bool isActiveColumn = activeColumnsBegin != activeColumnsEnd;
    if (isActiveColumn)
    {
      if (columnActiveSegmentsBegin != columnActiveSegmentsEnd)
      {
        activatePredictedColumn(
          activeCells_, winnerCells_, connections, rng_,
          columnActiveSegmentsBegin, columnActiveSegmentsEnd,
          prevActiveCellsDense, prevWinnerCells,
          numActivePotentialSynapsesForSegment_,
          maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
          maxSynapsesPerSegment_, learn);
      }
      else
      {
        burstColumn(
          activeCells_, winnerCells_, connections, rng_,
          lastUsedIterationForSegment_,
          column, columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
          prevActiveCellsDense, prevWinnerCells,
          numActivePotentialSynapsesForSegment_, iteration_,
          cellsPerColumn_, maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
          maxSegmentsPerCell_, maxSynapsesPerSegment_, learn);
      }
    }
    else
    {
      if (learn)
      {
        punishPredictedColumn(
          connections,
          columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
          prevActiveCellsDense,
          predictedSegmentDecrement_);
      }
    }
  }
}

void TemporalMemory::activateDendrites(bool learn)
{
  const UInt32 length = connections.segmentFlatListLength();

  numActiveConnectedSynapsesForSegment_.assign(length, 0);
  numActivePotentialSynapsesForSegment_.assign(length, 0);
  connections.computeActivity(numActiveConnectedSynapsesForSegment_,
                              numActivePotentialSynapsesForSegment_,
                              activeCells_,
                              connectedPermanence_);

  // Active segments, connected synapses.
  activeSegments_.clear();
  for (Segment segment = 0;
       segment < numActiveConnectedSynapsesForSegment_.size();
       segment++)
  {
    if (numActiveConnectedSynapsesForSegment_[segment] >= activationThreshold_)
    {
      activeSegments_.push_back(segment);
    }
  }
  std::sort(activeSegments_.begin(), activeSegments_.end(),
            [&](Segment a, Segment b)
            {
              return connections.compareSegments(a, b);
            });

  // Matching segments, potential synapses.
  matchingSegments_.clear();
  for (Segment segment = 0;
       segment < numActivePotentialSynapsesForSegment_.size();
       segment++)
  {
    if (numActivePotentialSynapsesForSegment_[segment] >= minThreshold_)
    {
      matchingSegments_.push_back(segment);
    }
  }
  std::sort(matchingSegments_.begin(), matchingSegments_.end(),
            [&](Segment a, Segment b)
            {
              return connections.compareSegments(a, b);
            });

  if (learn)
  {
    for (Segment segment : activeSegments_)
    {
      lastUsedIterationForSegment_[segment] = iteration_;
    }

    iteration_++;
  }
}

void TemporalMemory::compute(
  size_t activeColumnsSize,
  const UInt activeColumns[],
  bool learn)
{
  activateCells(activeColumnsSize, activeColumns, learn);
  activateDendrites(learn);
}

void TemporalMemory::reset(void)
{
  activeCells_.clear();
  winnerCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
}

// ==============================
//  Helper functions
// ==============================

Segment TemporalMemory::createSegment(CellIdx cell)
{
  return ::createSegment(connections, lastUsedIterationForSegment_,
                         cell, iteration_, maxSegmentsPerCell_);
}

Int TemporalMemory::columnForCell(CellIdx cell)
{
  _validateCell(cell);

  return cell / cellsPerColumn_;
}

vector<CellIdx> TemporalMemory::cellsForColumn(Int column)
{
  const CellIdx start = cellsPerColumn_ * column;
  const CellIdx end = start + cellsPerColumn_;

  vector<CellIdx> cellsInColumn;
  for (CellIdx i = start; i < end; i++)
  {
    cellsInColumn.push_back(i);
  }

  return cellsInColumn;
}

UInt TemporalMemory::numberOfCells(void)
{
  return connections.numCells();
}

vector<CellIdx> TemporalMemory::getActiveCells() const
{
  return activeCells_;
}

vector<CellIdx> TemporalMemory::getPredictiveCells() const
{
  vector<CellIdx> predictiveCells;

  for (auto segment = activeSegments_.begin();
       segment != activeSegments_.end(); segment++)
  {
    CellIdx cell = connections.cellForSegment(*segment);
    if (segment == activeSegments_.begin() ||
        cell != predictiveCells.back())
    {
      predictiveCells.push_back(cell);
    }
  }

  return predictiveCells;
}

vector<CellIdx> TemporalMemory::getWinnerCells() const
{
  return winnerCells_;
}

vector<Segment> TemporalMemory::getActiveSegments() const
{
  return activeSegments_;
}

vector<Segment> TemporalMemory::getMatchingSegments() const
{
  return matchingSegments_;
}

UInt TemporalMemory::numberOfColumns() const
{
  return numColumns_;
}

bool TemporalMemory::_validateCell(CellIdx cell)
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

UInt TemporalMemory::getCellsPerColumn() const
{
  return cellsPerColumn_;
}

UInt TemporalMemory::getActivationThreshold() const
{
  return activationThreshold_;
}

void TemporalMemory::setActivationThreshold(UInt activationThreshold)
{
  activationThreshold_ = activationThreshold;
}

Permanence TemporalMemory::getInitialPermanence() const
{
  return initialPermanence_;
}

void TemporalMemory::setInitialPermanence(Permanence initialPermanence)
{
  initialPermanence_ = initialPermanence;
}

Permanence TemporalMemory::getConnectedPermanence() const
{
  return connectedPermanence_;
}

void TemporalMemory::setConnectedPermanence(Permanence connectedPermanence)
{
  connectedPermanence_ = connectedPermanence;
}

UInt TemporalMemory::getMinThreshold() const
{
  return minThreshold_;
}

void TemporalMemory::setMinThreshold(UInt minThreshold)
{
  minThreshold_ = minThreshold;
}

UInt TemporalMemory::getMaxNewSynapseCount() const
{
  return maxNewSynapseCount_;
}

void TemporalMemory::setMaxNewSynapseCount(UInt maxNewSynapseCount)
{
  maxNewSynapseCount_ = maxNewSynapseCount;
}

Permanence TemporalMemory::getPermanenceIncrement() const
{
  return permanenceIncrement_;
}

void TemporalMemory::setPermanenceIncrement(Permanence permanenceIncrement)
{
  permanenceIncrement_ = permanenceIncrement;
}

Permanence TemporalMemory::getPermanenceDecrement() const
{
  return permanenceDecrement_;
}

void TemporalMemory::setPermanenceDecrement(Permanence permanenceDecrement)
{
  permanenceDecrement_ = permanenceDecrement;
}

Permanence TemporalMemory::getPredictedSegmentDecrement() const
{
  return predictedSegmentDecrement_;
}

void TemporalMemory::setPredictedSegmentDecrement(Permanence predictedSegmentDecrement)
{
  predictedSegmentDecrement_ = predictedSegmentDecrement;
}

UInt TemporalMemory::getMaxSegmentsPerCell() const
{
  return maxSegmentsPerCell_;
}

UInt TemporalMemory::getMaxSynapsesPerSegment() const
{
  return maxSynapsesPerSegment_;
}

UInt TemporalMemory::version() const
{
  return TM_VERSION;
}

/**
* Create a RNG with given seed
*/
void TemporalMemory::seed_(UInt64 seed)
{
  rng_ = Random(seed);
}

UInt TemporalMemory::persistentSize() const
{
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}

template<typename FloatType>
static void saveFloat_(ostream& outStream, FloatType v)
{
  outStream << std::setprecision(std::numeric_limits<FloatType>::max_digits10)
            << v
            << " ";
}

void TemporalMemory::save(ostream& outStream) const
{
  // Write a starting marker and version.
  outStream << "TemporalMemory" << endl;
  outStream << TM_VERSION << endl;

  outStream << numColumns_ << " "
            << cellsPerColumn_ << " "
            << activationThreshold_ << " ";

  saveFloat_(outStream, initialPermanence_);
  saveFloat_(outStream, connectedPermanence_);

  outStream << minThreshold_ << " "
            << maxNewSynapseCount_ << " ";

  saveFloat_(outStream, permanenceIncrement_);
  saveFloat_(outStream, permanenceDecrement_);
  saveFloat_(outStream, predictedSegmentDecrement_);

  outStream << maxSegmentsPerCell_ << " "
            << maxSynapsesPerSegment_ << " "
            << iteration_ << " ";

  outStream << endl;

  connections.save(outStream);
  outStream << endl;

  outStream << rng_ << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto & elem : columnDimensions_)
  {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << activeCells_.size() << " ";
  for (CellIdx cell : activeCells_)
  {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << winnerCells_.size() << " ";
  for (CellIdx cell : winnerCells_)
  {
    outStream << cell << " ";
  }
  outStream << endl;

  outStream << activeSegments_.size() << " ";
  for (Segment segment : activeSegments_)
  {
    const CellIdx cell = connections.cellForSegment(segment);
    const vector<Segment>& segments = connections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActiveConnectedSynapsesForSegment_[segment] << " ";
  }
  outStream << endl;

  outStream << matchingSegments_.size() << " ";
  for (Segment segment : matchingSegments_)
  {
    const CellIdx cell = connections.cellForSegment(segment);
    const vector<Segment>& segments = connections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActivePotentialSynapsesForSegment_[segment] << " ";
  }
  outStream << endl;

  outStream << "~TemporalMemory" << endl;
}

void TemporalMemory::write(TemporalMemoryProto::Builder& proto) const
{
  auto columnDims = proto.initColumnDimensions(columnDimensions_.size());
  for (UInt i = 0; i < columnDimensions_.size(); i++)
  {
    columnDims.set(i, columnDimensions_[i]);
  }

  proto.setCellsPerColumn(cellsPerColumn_);
  proto.setActivationThreshold(activationThreshold_);
  proto.setInitialPermanence(initialPermanence_);
  proto.setConnectedPermanence(connectedPermanence_);
  proto.setMinThreshold(minThreshold_);
  proto.setMaxNewSynapseCount(maxNewSynapseCount_);
  proto.setPermanenceIncrement(permanenceIncrement_);
  proto.setPermanenceDecrement(permanenceDecrement_);
  proto.setPredictedSegmentDecrement(predictedSegmentDecrement_);

  proto.setMaxSegmentsPerCell(maxSegmentsPerCell_);
  proto.setMaxSynapsesPerSegment(maxSynapsesPerSegment_);

  auto _connections = proto.initConnections();
  connections.write(_connections);

  auto random = proto.initRandom();
  rng_.write(random);

  auto activeCells = proto.initActiveCells(activeCells_.size());
  UInt i = 0;
  for (CellIdx cell : activeCells_)
  {
    activeCells.set(i++, cell);
  }

  auto winnerCells = proto.initWinnerCells(winnerCells_.size());
  i = 0;
  for (CellIdx cell : winnerCells_)
  {
    winnerCells.set(i++, cell);
  }

  auto activeSegments = proto.initActiveSegments(activeSegments_.size());
  for (UInt i = 0; i < activeSegments_.size(); ++i)
  {
    activeSegments[i].setCell(
      connections.cellForSegment(activeSegments_[i]));
    activeSegments[i].setIdxOnCell(
      connections.idxOnCellForSegment(activeSegments_[i]));
  }

  auto matchingSegments = proto.initMatchingSegments(matchingSegments_.size());
  for (UInt i = 0; i < matchingSegments_.size(); ++i)
  {
    matchingSegments[i].setCell(
      connections.cellForSegment(matchingSegments_[i]));
    matchingSegments[i].setIdxOnCell(
      connections.idxOnCellForSegment(matchingSegments_[i]));
  }

  auto numActivePotentialSynapsesForSegment =
    proto.initNumActivePotentialSynapsesForSegment(
      numActivePotentialSynapsesForSegment_.size());
  for (Segment segment = 0;
       segment < numActivePotentialSynapsesForSegment_.size();
       segment++)
  {
    numActivePotentialSynapsesForSegment[segment].setCell(
      connections.cellForSegment(segment));
    numActivePotentialSynapsesForSegment[segment].setIdxOnCell(
      connections.idxOnCellForSegment(segment));
    numActivePotentialSynapsesForSegment[segment].setNumber(
      numActivePotentialSynapsesForSegment_[segment]);
  }

  proto.setIteration(iteration_);

  auto lastUsedIterationForSegment =
    proto.initLastUsedIterationForSegment(lastUsedIterationForSegment_.size());
  for (Segment segment = 0;
       segment < lastUsedIterationForSegment_.size();
       ++segment)
  {
    lastUsedIterationForSegment[segment].setCell(
      connections.cellForSegment(segment));
    lastUsedIterationForSegment[segment].setIdxOnCell(
      connections.idxOnCellForSegment(segment));
    lastUsedIterationForSegment[segment].setNumber(
      lastUsedIterationForSegment_[segment]);
  }
}

// Implementation note: this method sets up the instance using data from
// proto. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void TemporalMemory::read(TemporalMemoryProto::Reader& proto)
{
  numColumns_ = 1;
  columnDimensions_.clear();
  for (UInt dimension : proto.getColumnDimensions())
  {
    numColumns_ *= dimension;
    columnDimensions_.push_back(dimension);
  }

  cellsPerColumn_ = proto.getCellsPerColumn();
  activationThreshold_ = proto.getActivationThreshold();
  initialPermanence_ = proto.getInitialPermanence();
  connectedPermanence_ = proto.getConnectedPermanence();
  minThreshold_ = proto.getMinThreshold();
  maxNewSynapseCount_ = proto.getMaxNewSynapseCount();
  permanenceIncrement_ = proto.getPermanenceIncrement();
  permanenceDecrement_ = proto.getPermanenceDecrement();
  predictedSegmentDecrement_ = proto.getPredictedSegmentDecrement();

  maxSegmentsPerCell_ = proto.getMaxSegmentsPerCell();
  maxSynapsesPerSegment_ = proto.getMaxSynapsesPerSegment();

  auto _connections = proto.getConnections();
  connections.read(_connections);

  numActiveConnectedSynapsesForSegment_.assign(
    connections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForSegment_.assign(
    connections.segmentFlatListLength(), 0);

  auto random = proto.getRandom();
  rng_.read(random);

  activeCells_.clear();
  for (auto cell : proto.getActiveCells())
  {
    activeCells_.push_back(cell);
  }

  winnerCells_.clear();
  for (auto cell : proto.getWinnerCells())
  {
    winnerCells_.push_back(cell);
  }

  activeSegments_.clear();
  for (auto value : proto.getActiveSegments())
  {
    const Segment segment = connections.getSegment(value.getCell(),
                                                   value.getIdxOnCell());
    activeSegments_.push_back(segment);
  }

  matchingSegments_.clear();
  for (auto value : proto.getMatchingSegments())
  {
    const Segment segment = connections.getSegment(value.getCell(),
                                                   value.getIdxOnCell());
    matchingSegments_.push_back(segment);
  }

  numActivePotentialSynapsesForSegment_.clear();
  numActivePotentialSynapsesForSegment_.resize(
    connections.segmentFlatListLength());
  for (auto segmentNumPair : proto.getNumActivePotentialSynapsesForSegment())
  {
    const Segment segment = connections.getSegment(
      segmentNumPair.getCell(), segmentNumPair.getIdxOnCell());
    numActivePotentialSynapsesForSegment_[segment] = segmentNumPair.getNumber();
  }

  iteration_ = proto.getIteration();

  lastUsedIterationForSegment_.clear();
  lastUsedIterationForSegment_.resize(connections.segmentFlatListLength());
  for (auto segmentIterationPair : proto.getLastUsedIterationForSegment())
  {
    const Segment segment = connections.getSegment(
      segmentIterationPair.getCell(), segmentIterationPair.getIdxOnCell());
    lastUsedIterationForSegment_[segment] = segmentIterationPair.getNumber();
  }
}

void TemporalMemory::load(istream& inStream)
{
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "TemporalMemory");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= TM_VERSION);

  // Retrieve simple variables
  inStream >> numColumns_
    >> cellsPerColumn_
    >> activationThreshold_
    >> initialPermanence_
    >> connectedPermanence_
    >> minThreshold_
    >> maxNewSynapseCount_
    >> permanenceIncrement_
    >> permanenceDecrement_
    >> predictedSegmentDecrement_
    >> maxSegmentsPerCell_
    >> maxSynapsesPerSegment_
    >> iteration_;

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
  for (UInt i = 0; i < numColumnDimensions; i++)
  {
    inStream >> columnDimensions_[i];
  }

  UInt numActiveCells;
  inStream >> numActiveCells;
  for (UInt i = 0; i < numActiveCells; i++)
  {
    CellIdx cell;
    inStream >> cell;
    activeCells_.push_back(cell);
  }

  if (version < 2)
  {
    UInt numPredictiveCells;
    inStream >> numPredictiveCells;
    for (UInt i = 0; i < numPredictiveCells; i++)
    {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++)
  {
    CellIdx cell;
    inStream >> cell;
    winnerCells_.push_back(cell);
  }

  UInt numActiveSegments;
  inStream >> numActiveSegments;
  activeSegments_.resize(numActiveSegments);
  for (UInt i = 0; i < numActiveSegments; i++)
  {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = connections.getSegment(cellIdx, idx);
    activeSegments_[i] = segment;

    if (version < 2)
    {
      numActiveConnectedSynapsesForSegment_[segment] = 0; // Unknown
    }
    else
    {
      inStream >> numActiveConnectedSynapsesForSegment_[segment];
    }
  }

  UInt numMatchingSegments;
  inStream >> numMatchingSegments;
  matchingSegments_.resize(numMatchingSegments);
  for (UInt i = 0; i < numMatchingSegments; i++)
  {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = connections.getSegment(cellIdx, idx);
    matchingSegments_[i] = segment;

    if (version < 2)
    {
      numActivePotentialSynapsesForSegment_[segment] = 0; // Unknown
    }
    else
    {
      inStream >> numActivePotentialSynapsesForSegment_[segment];
    }
  }

  if (version < 2)
  {
    UInt numMatchingCells;
    inStream >> numMatchingCells;
    for (UInt i = 0; i < numMatchingCells; i++)
    {
      CellIdx cell;
      inStream >> cell; // Ignore
    }
  }

  lastUsedIterationForSegment_.resize(connections.segmentFlatListLength());

  inStream >> marker;
  NTA_CHECK(marker == "~TemporalMemory");

}

static set< pair<CellIdx,SynapseIdx> >
getComparableSegmentSet(const Connections& connections,
                        const vector<Segment>& segments)
{
  set< pair<CellIdx,SynapseIdx> > segmentSet;
  for (Segment segment : segments)
  {
    segmentSet.emplace(connections.cellForSegment(segment),
                       connections.idxOnCellForSegment(segment));
  }
  return segmentSet;
}

bool TemporalMemory::operator==(const TemporalMemory& other)
{
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
      iteration_ != other.iteration_)
  {
    return false;
  }

  if (connections != other.connections)
  {
    return false;
  }

  if (getComparableSegmentSet(connections, activeSegments_) !=
      getComparableSegmentSet(other.connections, other.activeSegments_) ||
      getComparableSegmentSet(connections, matchingSegments_) !=
      getComparableSegmentSet(other.connections, other.matchingSegments_))
  {
    return false;
  }

  return true;
}

bool TemporalMemory::operator!=(const TemporalMemory& other)
{
  return !(*this == other);
}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main TM creation parameters
void TemporalMemory::printParameters()
{
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
    << "predictedSegmentDecrement = " << getPredictedSegmentDecrement() << std::endl
    << "maxSegmentsPerCell        = " << getMaxSegmentsPerCell() << std::endl
    << "maxSynapsesPerSegment     = " << getMaxSynapsesPerSegment() << std::endl;
}

void TemporalMemory::printState(vector<UInt> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i)
  {
    if (i > 0 && i % 10 == 0)
    {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void TemporalMemory::printState(vector<Real> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i)
  {
    if (i > 0 && i % 10 == 0)
    {
      std::cout << "\n   ";
    }
    std::printf("%6.3f ", state[i]);
  }
  std::cout << "]\n";
}
