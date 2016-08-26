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
 * Implementation of ExtendedTemporalMemory
 */

#include <cstring>
#include <climits>
#include <iostream>
#include <string>
#include <iterator>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>
#include <nupic/experimental/ExtendedTemporalMemory.hpp>
#include <nupic/utils/GroupBy.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::experimental::extended_temporal_memory;

static const Permanence EPSILON = 0.000001;
static const UInt EXTENDED_TM_VERSION = 1;
static const UInt32 MIN_PREDICTIVE_THRESHOLD = 2;
static const vector<CellIdx> CELLS_NONE = {};



ExtendedTemporalMemory::ExtendedTemporalMemory()
{
}

ExtendedTemporalMemory::ExtendedTemporalMemory(
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
  bool formInternalBasalConnections,
  bool learnOnOneCell,
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
    formInternalBasalConnections,
    learnOnOneCell,
    seed,
    maxSegmentsPerCell,
    maxSynapsesPerSegment);
}

ExtendedTemporalMemory::~ExtendedTemporalMemory()
{
}

void ExtendedTemporalMemory::initialize(
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
  bool formInternalBasalConnections,
  bool learnOnOneCell,
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
  formInternalBasalConnections_ = formInternalBasalConnections;
  learnOnOneCell_ = learnOnOneCell;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;
  predictedSegmentDecrement_ = predictedSegmentDecrement;

  // Initialize member variables
  basalConnections = Connections(numberOfCells(),
                                 maxSegmentsPerCell,
                                 maxSynapsesPerSegment);
  apicalConnections = Connections(numberOfCells(),
                                  maxSegmentsPerCell,
                                  maxSynapsesPerSegment);

  seed_((UInt64)(seed < 0 ? rand() : seed));

  activeCells_.clear();
  winnerCells_.clear();
  activeBasalSegments_.clear();
  matchingBasalSegments_.clear();
  activeApicalSegments_.clear();
  matchingApicalSegments_.clear();
  chosenCellForColumn_.clear();
}

static UInt32 predictiveScore(
  vector<SegmentOverlap>::const_iterator cellActiveBasalBegin,
  vector<SegmentOverlap>::const_iterator cellActiveBasalEnd,
  vector<SegmentOverlap>::const_iterator cellActiveApicalBegin,
  vector<SegmentOverlap>::const_iterator cellActiveApicalEnd)
{
  UInt32 score = 0;

  if (cellActiveBasalBegin != cellActiveBasalEnd)
  {
    score += 2;
  }

  if (cellActiveApicalBegin != cellActiveApicalEnd)
  {
    score += 1;
  }

  return score;
}

static CellIdx cellForSegment(const SegmentOverlap& s)
{
  return s.segment.cell;
}

static tuple<vector<SegmentOverlap>::const_iterator,
             vector<SegmentOverlap>::const_iterator>
segmentsForCell(vector<SegmentOverlap>::const_iterator segmentsStart,
                vector<SegmentOverlap>::const_iterator segmentsEnd,
                CellIdx cell)
{
  const auto cellBegin = std::find_if(segmentsStart, segmentsEnd,
                                      [&](const SegmentOverlap& x)
                                      {
                                        return x.segment.cell == cell;
                                      });
  const auto cellEnd = std::find_if(cellBegin, segmentsEnd,
                                    [&](const SegmentOverlap& x)
                                    {
                                      return x.segment.cell != cell;
                                    });
  return tuple<vector<SegmentOverlap>::const_iterator,
               vector<SegmentOverlap>::const_iterator>(cellBegin, cellEnd);
}

static CellIdx getLeastUsedCell(
  Connections& connections,
  Random& rng,
  UInt column,
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
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevActiveExternalCells,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  vector<Synapse> synapses = connections.synapsesForSegment(segment);

  for (Synapse synapse : synapses)
  {
    const SynapseData synapseData = connections.dataForSynapse(synapse);

    bool isActive;
    if (synapseData.presynapticCell < connections.numCells())
    {
      isActive = std::binary_search(
        prevActiveCells.begin(),
        prevActiveCells.end(),
        synapseData.presynapticCell);
    }
    else
    {
      isActive = std::binary_search(
        prevActiveExternalCells.begin(),
        prevActiveExternalCells.end(),
        synapseData.presynapticCell - connections.numCells());
    }

    Permanence permanence = synapseData.permanence;

    if (isActive)
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
      connections.destroySynapse(synapse);
    }
    else
    {
      connections.updateSynapsePermanence(synapse, permanence);
    }
  }

  if (connections.numSynapses(segment) == 0)
  {
    connections.destroySegment(segment);
  }
}

static void growSynapses(
  Connections& connections,
  Random& rng,
  Segment segment,
  UInt32 nDesiredNewSynapses,
  const vector<CellIdx>& internalCandidates,
  const vector<CellIdx>& externalCandidates,
  Permanence initialPermanence)
{
  vector<CellIdx> candidates;
  candidates.reserve(internalCandidates.size() + externalCandidates.size());
  candidates.insert(candidates.begin(), internalCandidates.begin(),
                    internalCandidates.end());
  for (CellIdx cell : externalCandidates)
  {
    candidates.push_back(cell + connections.numCells());
  }

  // Instead of erasing candidates, swap them to the end, and remember where the
  // "eligible" candidates end.
  auto eligibleEnd = candidates.end();

  // Remove cells that are already synapsed on by this segment
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    CellIdx presynapticCell =
      connections.dataForSynapse(synapse).presynapticCell;
    auto ineligible = find(candidates.begin(), eligibleEnd, presynapticCell);
    if (ineligible != eligibleEnd)
    {
      eligibleEnd--;
      std::iter_swap(ineligible, eligibleEnd);
    }
  }

  const UInt32 nActual =
    std::min(nDesiredNewSynapses,
             (UInt32)std::distance(candidates.begin(), eligibleEnd));

  // Pick nActual cells randomly.
  for (UInt32 c = 0; c < nActual; c++)
  {
    size_t i = rng.getUInt32(std::distance(candidates.begin(), eligibleEnd));
    connections.createSynapse(segment, candidates[i], initialPermanence);
    eligibleEnd--;
    std::swap(candidates[i], *eligibleEnd);
  }
}

static void learnOnCell(
  Connections& connections,
  Random& rng,
  CellIdx cell,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& internalCandidates,
  const vector<CellIdx>& externalCandidates,
  vector<SegmentOverlap>::const_iterator cellActiveSegmentsBegin,
  vector<SegmentOverlap>::const_iterator cellActiveSegmentsEnd,
  vector<SegmentOverlap>::const_iterator cellMatchingSegmentsBegin,
  vector<SegmentOverlap>::const_iterator cellMatchingSegmentsEnd,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  if (cellActiveSegmentsBegin != cellActiveSegmentsEnd)
  {
    // Learn on every active segment.

    auto bySegment = [](const SegmentOverlap& x) { return x.segment; };
    for (auto segmentData : iterGroupBy(
           cellActiveSegmentsBegin, cellActiveSegmentsEnd, bySegment,
           cellMatchingSegmentsBegin, cellMatchingSegmentsEnd, bySegment))
    {
      // Find the active segment's corresponding "matching" overlap.
      Segment segment;
      vector<SegmentOverlap>::const_iterator
        activeOverlapsBegin, activeOverlapsEnd,
        matchingOverlapsBegin, matchingOverlapsEnd;
      tie(segment,
          activeOverlapsBegin, activeOverlapsEnd,
          matchingOverlapsBegin, matchingOverlapsEnd) = segmentData;
      if (activeOverlapsBegin != activeOverlapsEnd)
      {
        // Active segments are a superset of matching segments.
        NTA_ASSERT(std::distance(activeOverlapsBegin, activeOverlapsEnd) == 1);
        NTA_ASSERT(std::distance(matchingOverlapsBegin, matchingOverlapsEnd) == 1);

        adaptSegment(connections,
                     segment,
                     prevActiveCells, externalCandidates,
                     permanenceIncrement, permanenceDecrement);

        const Int32 nActivePotentialSynapses = matchingOverlapsBegin->overlap;
        const Int32 nGrowDesired = (maxNewSynapseCount -
                                    nActivePotentialSynapses);
        if (nGrowDesired > 0)
        {
          growSynapses(connections, rng,
                       segment, nGrowDesired,
                       internalCandidates, externalCandidates,
                       initialPermanence);
        }
      }
    }
  }
  else if (cellMatchingSegmentsBegin != cellMatchingSegmentsEnd)
  {
    // No active segments.
    // Learn on the best matching segment.

    const SegmentOverlap& bestMatching = *std::max_element(
      cellMatchingSegmentsBegin, cellMatchingSegmentsEnd,
      [](const SegmentOverlap& a, const SegmentOverlap& b)
      {
        return a.overlap < b.overlap;
      });

    adaptSegment(connections,
                 bestMatching.segment,
                 prevActiveCells, externalCandidates,
                 permanenceIncrement, permanenceDecrement);

    const Int32 nGrowDesired = maxNewSynapseCount - bestMatching.overlap;
    if (nGrowDesired > 0)
    {
      growSynapses(connections, rng,
                   bestMatching.segment, nGrowDesired,
                   internalCandidates, externalCandidates,
                   initialPermanence);
    }
  }
  else
  {
    // No matching segments.
    // Grow a new segment and learn on it.

    // Don't grow a segment that will never match.
    const UInt32 nGrowExact = std::min(maxNewSynapseCount,
                                       (UInt)(internalCandidates.size() +
                                              externalCandidates.size()));
    if (nGrowExact > 0)
    {
      const Segment segment = connections.createSegment(cell);
      growSynapses(connections, rng,
                   segment, nGrowExact,
                   internalCandidates, externalCandidates,
                   initialPermanence);
      NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
    }
  }
}

static void activatePredictedColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& basalConnections,
  Connections& apicalConnections,
  Random& rng,
  vector<SegmentOverlap>::const_iterator columnActiveBasalBegin,
  vector<SegmentOverlap>::const_iterator columnActiveBasalEnd,
  vector<SegmentOverlap>::const_iterator columnMatchingBasalBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingBasalEnd,
  vector<SegmentOverlap>::const_iterator columnActiveApicalBegin,
  vector<SegmentOverlap>::const_iterator columnActiveApicalEnd,
  vector<SegmentOverlap>::const_iterator columnMatchingApicalBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingApicalEnd,
  UInt32 predictiveThreshold,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevWinnerCells,
  const vector<CellIdx>& prevActiveExternalCellsBasal,
  const vector<CellIdx>& prevActiveExternalCellsApical,
  Permanence initialPermanence,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  bool formInternalBasalConnections,
  bool learn)
{
  for (auto& cellData : iterGroupBy(
         columnActiveBasalBegin, columnActiveBasalEnd, cellForSegment,
         columnMatchingBasalBegin, columnMatchingBasalEnd, cellForSegment,
         columnActiveApicalBegin, columnActiveApicalEnd, cellForSegment,
         columnMatchingApicalBegin, columnMatchingApicalEnd, cellForSegment))
  {
    CellIdx cell;
    vector<SegmentOverlap>::const_iterator
      cellActiveBasalBegin, cellActiveBasalEnd,
      cellMatchingBasalBegin, cellMatchingBasalEnd,
      cellActiveApicalBegin, cellActiveApicalEnd,
      cellMatchingApicalBegin, cellMatchingApicalEnd;
    tie(cell,
        cellActiveBasalBegin, cellActiveBasalEnd,
        cellMatchingBasalBegin, cellMatchingBasalEnd,
        cellActiveApicalBegin, cellActiveApicalEnd,
        cellMatchingApicalBegin, cellMatchingApicalEnd) = cellData;

    if (predictiveScore(cellActiveBasalBegin, cellActiveBasalEnd,
                        cellActiveApicalBegin, cellActiveApicalEnd)
        >= predictiveThreshold)
    {
      activeCells.push_back(cell);
      winnerCells.push_back(cell);

      if (learn)
      {
        learnOnCell(basalConnections, rng,
                    cell, prevActiveCells,
                    (formInternalBasalConnections
                     ? prevWinnerCells : CELLS_NONE),
                    prevActiveExternalCellsBasal,
                    cellActiveBasalBegin, cellActiveBasalEnd,
                    cellMatchingBasalBegin, cellMatchingBasalEnd,
                    maxNewSynapseCount, initialPermanence,
                    permanenceIncrement, permanenceDecrement);

        learnOnCell(apicalConnections, rng,
                    cell, prevActiveCells,
                    CELLS_NONE, prevActiveExternalCellsApical,
                    cellActiveApicalBegin, cellActiveApicalEnd,
                    cellMatchingApicalBegin, cellMatchingApicalEnd,
                    maxNewSynapseCount, initialPermanence,
                    permanenceIncrement, permanenceDecrement);
      }
    }
  }
}

static void burstColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& basalConnections,
  Connections& apicalConnections,
  Random& rng,
  map<UInt, CellIdx>& chosenCellForColumn,
  UInt column,
  vector<SegmentOverlap>::const_iterator columnActiveBasalBegin,
  vector<SegmentOverlap>::const_iterator columnActiveBasalEnd,
  vector<SegmentOverlap>::const_iterator columnMatchingBasalBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingBasalEnd,
  vector<SegmentOverlap>::const_iterator columnActiveApicalBegin,
  vector<SegmentOverlap>::const_iterator columnActiveApicalEnd,
  vector<SegmentOverlap>::const_iterator columnMatchingApicalBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingApicalEnd,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevWinnerCells,
  const vector<CellIdx>& prevActiveExternalCellsBasal,
  const vector<CellIdx>& prevActiveExternalCellsApical,
  UInt cellsPerColumn,
  Permanence initialPermanence,
  UInt maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  bool formInternalBasalConnections,
  bool learnOnOneCell,
  bool learn)
{
  // Calculate the active cells.
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++)
  {
    activeCells.push_back(cell);
  }

  // Mini optimization: don't search for the best basal segment twice.
  auto basalCandidatesBegin = columnMatchingBasalBegin;
  auto basalCandidatesEnd = columnMatchingBasalEnd;

  // Calculate the winner cell.
  CellIdx winnerCell;
  if (learnOnOneCell && chosenCellForColumn.count(column))
  {
    winnerCell = chosenCellForColumn.at(column);
  }
  else
  {
    if (columnMatchingBasalBegin != columnMatchingBasalEnd)
    {
      auto bestBasal = std::max_element(
        columnMatchingBasalBegin, columnMatchingBasalEnd,
        [](const SegmentOverlap& a, const SegmentOverlap& b)
        {
          return a.overlap < b.overlap;
        });

      basalCandidatesBegin = bestBasal;
      basalCandidatesEnd = bestBasal + 1;

      winnerCell = bestBasal->segment.cell;
    }
    else
    {
      winnerCell = getLeastUsedCell(basalConnections, rng, column,
                                    cellsPerColumn);
    }

    if (learnOnOneCell)
    {
      chosenCellForColumn[column] = winnerCell;
    }
  }
  winnerCells.push_back(winnerCell);

  // Learn.
  if (learn)
  {
    vector<SegmentOverlap>::const_iterator
      cellActiveApicalBegin, cellActiveApicalEnd,
      cellMatchingApicalBegin, cellMatchingApicalEnd,
      cellActiveBasalBegin, cellActiveBasalEnd;
    tie(cellActiveApicalBegin,
        cellActiveApicalEnd) = segmentsForCell(columnActiveApicalBegin,
                                               columnActiveApicalEnd,
                                               winnerCell);
    tie(cellMatchingApicalBegin,
        cellMatchingApicalEnd) = segmentsForCell(columnMatchingApicalBegin,
                                                 columnMatchingApicalEnd,
                                                 winnerCell);
    tie(cellActiveBasalBegin,
        cellActiveBasalEnd) = segmentsForCell(columnActiveBasalBegin,
                                              columnActiveBasalEnd,
                                              winnerCell);

    learnOnCell(basalConnections, rng,
                winnerCell, prevActiveCells,
                (formInternalBasalConnections
                 ? prevWinnerCells : CELLS_NONE),
                prevActiveExternalCellsBasal,
                cellActiveBasalBegin, cellActiveBasalEnd,
                basalCandidatesBegin, basalCandidatesEnd,
                maxNewSynapseCount, initialPermanence,
                permanenceIncrement, permanenceDecrement);

    learnOnCell(apicalConnections, rng,
                winnerCell, prevActiveCells,
                CELLS_NONE, prevActiveExternalCellsApical,
                cellActiveApicalBegin, cellActiveApicalEnd,
                cellMatchingApicalBegin, cellMatchingApicalEnd,
                maxNewSynapseCount, initialPermanence,
                permanenceIncrement, permanenceDecrement);
  }
}

static void punishPredictedColumn(
  Connections& connections,
  vector<SegmentOverlap>::const_iterator matchingSegmentsBegin,
  vector<SegmentOverlap>::const_iterator matchingSegmentsEnd,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevActiveExternalCells,
  Permanence predictedSegmentDecrement)
{
  if (predictedSegmentDecrement > 0.0)
  {
    for (auto matching = matchingSegmentsBegin;
         matching != matchingSegmentsEnd; matching++)
    {
      adaptSegment(connections, matching->segment,
                   prevActiveCells, prevActiveExternalCells,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void ExtendedTemporalMemory::activateCells(
  const vector<UInt>& activeColumns,
  const vector<CellIdx>& prevActiveExternalCellsBasal,
  const vector<CellIdx>& prevActiveExternalCellsApical,
  bool learn)
{
  NTA_ASSERT(std::is_sorted(activeColumns.begin(), activeColumns.end()));
  NTA_ASSERT(std::is_sorted(prevActiveExternalCellsBasal.begin(),
                            prevActiveExternalCellsBasal.end()));
  NTA_ASSERT(std::is_sorted(prevActiveExternalCellsApical.begin(),
                            prevActiveExternalCellsApical.end()));

  const vector<CellIdx> prevActiveCells = std::move(activeCells_);
  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  const auto columnForSegment =
    [&](const SegmentOverlap& s) { return s.segment.cell / cellsPerColumn_; };

  for (auto& columnData : groupBy(activeColumns, identity<UInt>,
                                  activeBasalSegments_, columnForSegment,
                                  matchingBasalSegments_, columnForSegment,
                                  activeApicalSegments_, columnForSegment,
                                  matchingApicalSegments_, columnForSegment))
  {
    UInt column;
    vector<UInt>::const_iterator
      activeColumnsBegin, activeColumnsEnd;
    vector<SegmentOverlap>::const_iterator
      columnActiveBasalBegin, columnActiveBasalEnd,
      columnMatchingBasalBegin, columnMatchingBasalEnd,
      columnActiveApicalBegin, columnActiveApicalEnd,
      columnMatchingApicalBegin, columnMatchingApicalEnd;
    tie(column,
        activeColumnsBegin, activeColumnsEnd,
        columnActiveBasalBegin, columnActiveBasalEnd,
        columnMatchingBasalBegin, columnMatchingBasalEnd,
        columnActiveApicalBegin, columnActiveApicalEnd,
        columnMatchingApicalBegin, columnMatchingApicalEnd) = columnData;

    const bool isActiveColumn = activeColumnsBegin != activeColumnsEnd;
    if (isActiveColumn)
    {
      UInt32 maxPredictiveScore = 0;

      for (auto& cellData : iterGroupBy(
             columnActiveBasalBegin, columnActiveBasalEnd, cellForSegment,
             columnActiveApicalBegin, columnActiveApicalEnd, cellForSegment))
      {
        CellIdx cell;
        vector<SegmentOverlap>::const_iterator
          cellActiveBasalBegin, cellActiveBasalEnd,
          cellActiveApicalBegin, cellActiveApicalEnd;
        tie(cell,
            cellActiveBasalBegin, cellActiveBasalEnd,
            cellActiveApicalBegin, cellActiveApicalEnd) = cellData;
        maxPredictiveScore = std::max(maxPredictiveScore,
                                      predictiveScore(cellActiveBasalBegin,
                                                      cellActiveBasalEnd,
                                                      cellActiveApicalBegin,
                                                      cellActiveApicalEnd));
      }

      if (maxPredictiveScore >= MIN_PREDICTIVE_THRESHOLD)
      {
        activatePredictedColumn(
          activeCells_, winnerCells_,
          basalConnections, apicalConnections, rng_,
          columnActiveBasalBegin, columnActiveBasalEnd,
          columnMatchingBasalBegin, columnMatchingBasalEnd,
          columnActiveApicalBegin, columnActiveApicalEnd,
          columnMatchingApicalBegin, columnMatchingApicalEnd,
          maxPredictiveScore,
          prevActiveCells, prevWinnerCells,
          prevActiveExternalCellsBasal, prevActiveExternalCellsApical,
          initialPermanence_, maxNewSynapseCount_,
          permanenceIncrement_, permanenceDecrement_,
          formInternalBasalConnections_, learn);
      }
      else
      {
        burstColumn(
          activeCells_, winnerCells_,
          basalConnections, apicalConnections, rng_,
          chosenCellForColumn_,
          column,
          columnActiveBasalBegin, columnActiveBasalEnd,
          columnMatchingBasalBegin, columnMatchingBasalEnd,
          columnActiveApicalBegin, columnActiveApicalEnd,
          columnMatchingApicalBegin, columnMatchingApicalEnd,
          prevActiveCells, prevWinnerCells,
          prevActiveExternalCellsBasal, prevActiveExternalCellsApical,
          cellsPerColumn_, initialPermanence_, maxNewSynapseCount_,
          permanenceIncrement_, permanenceDecrement_,
          formInternalBasalConnections_, learnOnOneCell_, learn);
      }
    }
    else
    {
      if (learn)
      {
        punishPredictedColumn(
          basalConnections,
          columnMatchingBasalBegin, columnMatchingBasalEnd,
          prevActiveCells, prevActiveExternalCellsBasal,
          predictedSegmentDecrement_);

        // Don't punish apical segments.
      }
    }
  }
}

static void activateDendrites(
  vector<SegmentOverlap>& activeSegments,
  vector<SegmentOverlap>& matchingSegments,
  Connections& connections,
  const vector<CellIdx>& activeCells,
  const vector<CellIdx>& activeExternalCells,
  Permanence connectedPermanence,
  UInt activationThreshold,
  UInt minThreshold,
  bool learn)
{
  SegmentExcitationTally excitations(connections, connectedPermanence, 0.0);
  for (CellIdx cell : activeCells)
  {
    excitations.addActivePresynapticCell(cell);
  }
  for (CellIdx cell : activeExternalCells)
  {
    excitations.addActivePresynapticCell(cell + connections.numCells());
  }

  excitations.getResults(activationThreshold, minThreshold,
                         activeSegments, matchingSegments);

  if (learn)
  {
    for (const SegmentOverlap& segmentOverlap : activeSegments)
    {
      connections.recordSegmentActivity(segmentOverlap.segment);
    }

    connections.startNewIteration();
  }
}

void ExtendedTemporalMemory::activateBasalDendrites(
  const vector<CellIdx>& activeExternalCells,
  bool learn)
{
  activeBasalSegments_.clear();
  matchingBasalSegments_.clear();
  activateDendrites(activeBasalSegments_, matchingBasalSegments_,
                    basalConnections,
                    activeCells_, activeExternalCells,
                    connectedPermanence_, activationThreshold_, minThreshold_,
                    learn);
}

void ExtendedTemporalMemory::activateApicalDendrites(
  const vector<CellIdx>& activeExternalCells,
  bool learn)
{
  activeApicalSegments_.clear();
  matchingApicalSegments_.clear();
  activateDendrites(activeApicalSegments_, matchingApicalSegments_,
                    apicalConnections,
                    activeCells_, activeExternalCells,
                    connectedPermanence_, activationThreshold_, minThreshold_,
                    learn);
}

void ExtendedTemporalMemory::compute(
  UInt activeColumnsSize, const UInt activeColumnsUnsorted[], bool learn)
{
  const vector<UInt> activeColumns(activeColumnsUnsorted,
                                   activeColumnsUnsorted + activeColumnsSize);
  compute(activeColumns, {}, {}, {}, {}, learn);
}

void ExtendedTemporalMemory::compute(
  const vector<UInt>& activeColumnsUnsorted,
  const vector<CellIdx>& prevActiveExternalCellsBasal,
  const vector<CellIdx>& activeExternalCellsBasal,
  const vector<CellIdx>& prevActiveExternalCellsApical,
  const vector<CellIdx>& activeExternalCellsApical,
  bool learn)
{
  vector<UInt> activeColumns(activeColumnsUnsorted.begin(),
                             activeColumnsUnsorted.end());
  std::sort(activeColumns.begin(), activeColumns.end());

  activateCells(activeColumns,
                prevActiveExternalCellsBasal,
                prevActiveExternalCellsApical,
                learn);

  activateBasalDendrites(activeExternalCellsBasal, learn);
  activateApicalDendrites(activeExternalCellsApical, learn);
}

void ExtendedTemporalMemory::reset(void)
{
  activeCells_.clear();
  winnerCells_.clear();
  activeBasalSegments_.clear();
  matchingBasalSegments_.clear();
  activeApicalSegments_.clear();
  matchingApicalSegments_.clear();
  chosenCellForColumn_.clear();
}

// ==============================
//  Helper functions
// ==============================

Int ExtendedTemporalMemory::columnForCell(CellIdx cell)
{
  _validateCell(cell);

  return cell / cellsPerColumn_;
}

vector<CellIdx> ExtendedTemporalMemory::cellsForColumn(Int column)
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

UInt ExtendedTemporalMemory::numberOfCells(void)
{
  return numberOfColumns() * cellsPerColumn_;
}

vector<CellIdx> ExtendedTemporalMemory::getActiveCells() const
{
  return activeCells_;
}

vector<CellIdx> ExtendedTemporalMemory::getPredictiveCells() const
{
  vector<CellIdx> predictiveCells;

  const auto columnForSegment =
    [&](const SegmentOverlap& s) { return s.segment.cell / cellsPerColumn_; };

  for (auto& columnData : groupBy(activeBasalSegments_, columnForSegment,
                                  activeApicalSegments_, columnForSegment))
  {
    UInt column;
    vector<SegmentOverlap>::const_iterator
      columnActiveBasalBegin, columnActiveBasalEnd,
      columnActiveApicalBegin, columnActiveApicalEnd;
    tie(column,
        columnActiveBasalBegin, columnActiveBasalEnd,
        columnActiveApicalBegin, columnActiveApicalEnd) = columnData;

    const auto groupedByCell = iterGroupBy(
       columnActiveBasalBegin, columnActiveBasalEnd, cellForSegment,
       columnActiveApicalBegin, columnActiveApicalEnd, cellForSegment);

    UInt32 maxDepolarization = 0;
    for (auto& cellData : groupedByCell)
    {
      vector<SegmentOverlap>::const_iterator
        cellActiveBasalBegin, cellActiveBasalEnd,
        cellActiveApicalBegin, cellActiveApicalEnd;
      tie(std::ignore,
          cellActiveBasalBegin, cellActiveBasalEnd,
          cellActiveApicalBegin, cellActiveApicalEnd) = cellData;

      maxDepolarization = std::max(maxDepolarization,
                                   predictiveScore(cellActiveBasalBegin,
                                                   cellActiveBasalEnd,
                                                   cellActiveApicalBegin,
                                                   cellActiveApicalEnd));
    }

    if (maxDepolarization >= MIN_PREDICTIVE_THRESHOLD)
    {
      for (auto& cellData : groupedByCell)
      {
        CellIdx cell;
        vector<SegmentOverlap>::const_iterator
          cellActiveBasalBegin, cellActiveBasalEnd,
          cellActiveApicalBegin, cellActiveApicalEnd;
        tie(cell,
            cellActiveBasalBegin, cellActiveBasalEnd,
            cellActiveApicalBegin, cellActiveApicalEnd) = cellData;

        if (predictiveScore(cellActiveBasalBegin, cellActiveBasalEnd,
                            cellActiveApicalBegin, cellActiveApicalEnd)
            >= maxDepolarization)
        {
          predictiveCells.push_back(cell);
        }
      }
    }
  }

  return predictiveCells;
}

vector<CellIdx> ExtendedTemporalMemory::getWinnerCells() const
{
  return winnerCells_;
}

vector<Segment> ExtendedTemporalMemory::getActiveBasalSegments() const
{
  vector<Segment> ret;
  ret.reserve(activeBasalSegments_.size());
  for (const SegmentOverlap& segmentOverlap : activeBasalSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

vector<Segment> ExtendedTemporalMemory::getMatchingBasalSegments() const
{
  vector<Segment> ret;
  ret.reserve(matchingBasalSegments_.size());
  for (const SegmentOverlap& segmentOverlap : matchingBasalSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

vector<Segment> ExtendedTemporalMemory::getActiveApicalSegments() const
{
  vector<Segment> ret;
  ret.reserve(activeApicalSegments_.size());
  for (const SegmentOverlap& segmentOverlap : activeApicalSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

vector<Segment> ExtendedTemporalMemory::getMatchingApicalSegments() const
{
  vector<Segment> ret;
  ret.reserve(matchingApicalSegments_.size());
  for (const SegmentOverlap& segmentOverlap : matchingApicalSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

UInt ExtendedTemporalMemory::numberOfColumns() const
{
  return numColumns_;
}

bool ExtendedTemporalMemory::_validateCell(CellIdx cell)
{
  if (cell < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell;
  return false;
}

vector<UInt> ExtendedTemporalMemory::getColumnDimensions() const
{
  return columnDimensions_;
}

UInt ExtendedTemporalMemory::getCellsPerColumn() const
{
  return cellsPerColumn_;
}

UInt ExtendedTemporalMemory::getActivationThreshold() const
{
  return activationThreshold_;
}

void ExtendedTemporalMemory::setActivationThreshold(UInt activationThreshold)
{
  activationThreshold_ = activationThreshold;
}

Permanence ExtendedTemporalMemory::getInitialPermanence() const
{
  return initialPermanence_;
}

void ExtendedTemporalMemory::setInitialPermanence(Permanence initialPermanence)
{
  initialPermanence_ = initialPermanence;
}

Permanence ExtendedTemporalMemory::getConnectedPermanence() const
{
  return connectedPermanence_;
}

void ExtendedTemporalMemory::setConnectedPermanence(
  Permanence connectedPermanence)
{
  connectedPermanence_ = connectedPermanence;
}

UInt ExtendedTemporalMemory::getMinThreshold() const
{
  return minThreshold_;
}

void ExtendedTemporalMemory::setMinThreshold(UInt minThreshold)
{
  minThreshold_ = minThreshold;
}

UInt ExtendedTemporalMemory::getMaxNewSynapseCount() const
{
  return maxNewSynapseCount_;
}

void ExtendedTemporalMemory::setMaxNewSynapseCount(UInt maxNewSynapseCount)
{
  maxNewSynapseCount_ = maxNewSynapseCount;
}

bool ExtendedTemporalMemory::getFormInternalBasalConnections() const
{
  return formInternalBasalConnections_;
}

void ExtendedTemporalMemory::setFormInternalBasalConnections(
  bool formInternalBasalConnections)
{
  formInternalBasalConnections_ = formInternalBasalConnections;
}

bool ExtendedTemporalMemory::getLearnOnOneCell() const
{
  return learnOnOneCell_;
}

void ExtendedTemporalMemory::setLearnOnOneCell(bool learnOnOneCell)
{
  learnOnOneCell_ = learnOnOneCell;
}

Permanence ExtendedTemporalMemory::getPermanenceIncrement() const
{
  return permanenceIncrement_;
}

void ExtendedTemporalMemory::setPermanenceIncrement(
  Permanence permanenceIncrement)
{
  permanenceIncrement_ = permanenceIncrement;
}

Permanence ExtendedTemporalMemory::getPermanenceDecrement() const
{
  return permanenceDecrement_;
}

void ExtendedTemporalMemory::setPermanenceDecrement(
  Permanence permanenceDecrement)
{
  permanenceDecrement_ = permanenceDecrement;
}

Permanence ExtendedTemporalMemory::getPredictedSegmentDecrement() const
{
  return predictedSegmentDecrement_;
}

void ExtendedTemporalMemory::setPredictedSegmentDecrement(
  Permanence predictedSegmentDecrement)
{
  predictedSegmentDecrement_ = predictedSegmentDecrement;
}

UInt ExtendedTemporalMemory::version() const
{
  return EXTENDED_TM_VERSION;
}

/**
* Create a RNG with given seed
*/
void ExtendedTemporalMemory::seed_(UInt64 seed)
{
  rng_ = Random(seed);
}

UInt ExtendedTemporalMemory::persistentSize() const
{
  // TODO: this won't scale!
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}

void ExtendedTemporalMemory::save(ostream& outStream) const
{
  // Write a starting marker and version.
  outStream << "ExtendedTemporalMemory" << endl;
  outStream << EXTENDED_TM_VERSION << endl;

  outStream << numColumns_ << " "
    << cellsPerColumn_ << " "
    << activationThreshold_ << " "
    << initialPermanence_ << " "
    << connectedPermanence_ << " "
    << minThreshold_ << " "
    << maxNewSynapseCount_ << " "
    << permanenceIncrement_ << " "
    << permanenceDecrement_ << " "
    << predictedSegmentDecrement_ << " "
    << formInternalBasalConnections_ << " "
    << endl;

  basalConnections.save(outStream);
  outStream << endl;

  apicalConnections.save(outStream);
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

  outStream << activeBasalSegments_.size() << " ";
  for (SegmentOverlap elem : activeBasalSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << matchingBasalSegments_.size() << " ";
  for (SegmentOverlap elem : matchingBasalSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << activeApicalSegments_.size() << " ";
  for (SegmentOverlap elem : activeApicalSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << matchingApicalSegments_.size() << " ";
  for (SegmentOverlap elem : matchingApicalSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  NTA_CHECK(learnOnOneCell_ == false) <<
    "Serialization is not supported for learnOnOneCell";
  NTA_CHECK(chosenCellForColumn_.empty()) <<
    "Serialization is not supported for learnOnOneCell";

  outStream << "~ExtendedTemporalMemory" << endl;
}

void ExtendedTemporalMemory::write(ExtendedTemporalMemoryProto::Builder& proto) const
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
  proto.setFormInternalBasalConnections(formInternalBasalConnections_);

  auto _basalConnections = proto.initBasalConnections();
  basalConnections.write(_basalConnections);

  auto _apicalConnections = proto.initApicalConnections();
  apicalConnections.write(_apicalConnections);

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

  auto activeBasalSegmentOverlaps =
    proto.initActiveBasalSegmentOverlaps(activeBasalSegments_.size());
  for (UInt i = 0; i < activeBasalSegments_.size(); ++i)
  {
    Segment segment = activeBasalSegments_[i].segment;
    activeBasalSegmentOverlaps[i].setCell(segment.cell);
    activeBasalSegmentOverlaps[i].setSegment(segment.idx);
    activeBasalSegmentOverlaps[i].setOverlap(
      activeBasalSegments_[i].overlap);
  }

  auto matchingBasalSegmentOverlaps =
    proto.initMatchingBasalSegmentOverlaps(matchingBasalSegments_.size());
  for (UInt i = 0; i < matchingBasalSegments_.size(); ++i)
  {
    Segment segment = matchingBasalSegments_[i].segment;
    matchingBasalSegmentOverlaps[i].setCell(segment.cell);
    matchingBasalSegmentOverlaps[i].setSegment(segment.idx);
    matchingBasalSegmentOverlaps[i].setOverlap(
      matchingBasalSegments_[i].overlap);
  }

  auto activeApicalSegmentOverlaps =
    proto.initActiveApicalSegmentOverlaps(activeApicalSegments_.size());
  for (UInt i = 0; i < activeApicalSegments_.size(); ++i)
  {
    Segment segment = activeApicalSegments_[i].segment;
    activeApicalSegmentOverlaps[i].setCell(segment.cell);
    activeApicalSegmentOverlaps[i].setSegment(segment.idx);
    activeApicalSegmentOverlaps[i].setOverlap(
      activeApicalSegments_[i].overlap);
  }

  auto matchingApicalSegmentOverlaps =
    proto.initMatchingApicalSegmentOverlaps(matchingApicalSegments_.size());
  for (UInt i = 0; i < matchingApicalSegments_.size(); ++i)
  {
    Segment segment = matchingApicalSegments_[i].segment;
    matchingApicalSegmentOverlaps[i].setCell(segment.cell);
    matchingApicalSegmentOverlaps[i].setSegment(segment.idx);
    matchingApicalSegmentOverlaps[i].setOverlap(
      matchingApicalSegments_[i].overlap);
  }

  NTA_CHECK(learnOnOneCell_ == false) <<
    "Serialization is not supported for learnOnOneCell";
  NTA_CHECK(chosenCellForColumn_.empty()) <<
    "Serialization is not supported for learnOnOneCell";
}

// Implementation note: this method sets up the instance using data from
// proto. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void ExtendedTemporalMemory::read(ExtendedTemporalMemoryProto::Reader& proto)
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
  formInternalBasalConnections_ = proto.getFormInternalBasalConnections();

  auto _basalConnections = proto.getBasalConnections();
  basalConnections.read(_basalConnections);

  auto _apicalConnections = proto.getApicalConnections();
  apicalConnections.read(_apicalConnections);

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

  activeBasalSegments_.clear();
  for (auto value : proto.getActiveBasalSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    activeBasalSegments_.push_back({segment, value.getOverlap()});
  }

  matchingBasalSegments_.clear();
  for (auto value : proto.getMatchingBasalSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    matchingBasalSegments_.push_back({segment, value.getOverlap()});
  }

  activeApicalSegments_.clear();
  for (auto value : proto.getActiveApicalSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    activeApicalSegments_.push_back({segment, value.getOverlap()});
  }

  matchingApicalSegments_.clear();
  for (auto value : proto.getMatchingApicalSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    matchingApicalSegments_.push_back({segment, value.getOverlap()});
  }

  learnOnOneCell_ = false;
  chosenCellForColumn_.clear();
}

void ExtendedTemporalMemory::load(istream& inStream)
{
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "ExtendedTemporalMemory");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= EXTENDED_TM_VERSION);

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
    >> formInternalBasalConnections_;

  basalConnections.load(inStream);
  apicalConnections.load(inStream);

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

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++)
  {
    CellIdx cell;
    inStream >> cell;
    winnerCells_.push_back(cell);
  }

  UInt numActiveBasalSegments;
  inStream >> numActiveBasalSegments;
  activeBasalSegments_.resize(numActiveBasalSegments);
  for (UInt i = 0; i < numActiveBasalSegments; i++)
  {
    inStream >> activeBasalSegments_[i].segment.idx;
    inStream >> activeBasalSegments_[i].segment.cell;
    inStream >> activeBasalSegments_[i].overlap;
  }

  UInt numMatchingBasalSegments;
  inStream >> numMatchingBasalSegments;
  matchingBasalSegments_.resize(numMatchingBasalSegments);
  for (UInt i = 0; i < numMatchingBasalSegments; i++)
  {
    inStream >> matchingBasalSegments_[i].segment.idx;
    inStream >> matchingBasalSegments_[i].segment.cell;
    inStream >> matchingBasalSegments_[i].overlap;
  }

  UInt numActiveApicalSegments;
  inStream >> numActiveApicalSegments;
  activeApicalSegments_.resize(numActiveApicalSegments);
  for (UInt i = 0; i < numActiveApicalSegments; i++)
  {
    inStream >> activeApicalSegments_[i].segment.idx;
    inStream >> activeApicalSegments_[i].segment.cell;
    inStream >> activeApicalSegments_[i].overlap;
  }

  UInt numMatchingApicalSegments;
  inStream >> numMatchingApicalSegments;
  matchingApicalSegments_.resize(numMatchingApicalSegments);
  for (UInt i = 0; i < numMatchingApicalSegments; i++)
  {
    inStream >> matchingApicalSegments_[i].segment.idx;
    inStream >> matchingApicalSegments_[i].segment.cell;
    inStream >> matchingApicalSegments_[i].overlap;
  }

  learnOnOneCell_ = false;
  chosenCellForColumn_.clear();

  inStream >> marker;
  NTA_CHECK(marker == "~ExtendedTemporalMemory");
}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main TM creation parameters
void ExtendedTemporalMemory::printParameters()
{
  std::cout << "------------CPP ExtendedTemporalMemory Parameters ------------------\n";
  std::cout
    << "version                   = " << EXTENDED_TM_VERSION << std::endl
    << "numColumns                = " << numberOfColumns() << std::endl
    << "cellsPerColumn            = " << getCellsPerColumn() << std::endl
    << "activationThreshold       = " << getActivationThreshold() << std::endl
    << "initialPermanence         = " << getInitialPermanence() << std::endl
    << "connectedPermanence       = " << getConnectedPermanence() << std::endl
    << "minThreshold              = " << getMinThreshold() << std::endl
    << "maxNewSynapseCount        = " << getMaxNewSynapseCount() << std::endl
    << "formInternalBasalConnections = " << getFormInternalBasalConnections() << std::endl
    << "learnOnOneCell            = " << getLearnOnOneCell() << std::endl
    << "permanenceIncrement       = " << getPermanenceIncrement() << std::endl
    << "permanenceDecrement       = " << getPermanenceDecrement() << std::endl
    << "predictedSegmentDecrement = " << getPredictedSegmentDecrement() << std::endl;
}

void ExtendedTemporalMemory::printState(vector<UInt> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void ExtendedTemporalMemory::printState(vector<Real> &state)
{
  std::cout << "[  ";
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::printf("%6.3f ", state[i]);
  }
  std::cout << "]\n";
}
