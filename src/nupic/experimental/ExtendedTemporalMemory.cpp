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
 *
 * The functions in this file use the following parameter ordering
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


ExtendedTemporalMemory::ExtendedTemporalMemory()
{
}

ExtendedTemporalMemory::ExtendedTemporalMemory(
  vector<UInt> columnDimensions,
  vector<UInt> basalInputDimensions,
  vector<UInt> apicalInputDimensions,
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
  UInt maxSynapsesPerSegment,
  bool checkInputs)
{
  initialize(
    columnDimensions,
    basalInputDimensions,
    apicalInputDimensions,
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
    maxSynapsesPerSegment,
    checkInputs);
}

ExtendedTemporalMemory::~ExtendedTemporalMemory()
{
}

static UInt countPoints(const vector<UInt> dimensions)
{
  if (dimensions.size() == 0)
  {
    return 0;
  }
  else
  {
    UInt numPoints = 1;
    for (UInt dimension : dimensions)
    {
      numPoints *= dimension;
    }
    return numPoints;
  }
}

void ExtendedTemporalMemory::initialize(
  vector<UInt> columnDimensions,
  vector<UInt> basalInputDimensions,
  vector<UInt> apicalInputDimensions,
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
  UInt maxSynapsesPerSegment,
  bool checkInputs)
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

  columnDimensions_ = columnDimensions;
  basalInputDimensions_ = basalInputDimensions;
  apicalInputDimensions_ = apicalInputDimensions;

  numColumns_ = countPoints(columnDimensions);
  numBasalInputs_ = countPoints(basalInputDimensions);
  numApicalInputs_ = countPoints(apicalInputDimensions);

  cellsPerColumn_ = cellsPerColumn;
  activationThreshold_ = activationThreshold;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  formInternalBasalConnections_ = formInternalBasalConnections;
  learnOnOneCell_ = learnOnOneCell;
  checkInputs_ = checkInputs;
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
  vector<Segment>::const_iterator cellActiveBasalBegin,
  vector<Segment>::const_iterator cellActiveBasalEnd,
  vector<Segment>::const_iterator cellActiveApicalBegin,
  vector<Segment>::const_iterator cellActiveApicalEnd)
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

static tuple<vector<Segment>::const_iterator,
             vector<Segment>::const_iterator>
segmentsForCell(vector<Segment>::const_iterator segmentsStart,
                vector<Segment>::const_iterator segmentsEnd,
                CellIdx cell,
                const Connections& connections)
{
  const auto cellBegin = std::find_if(
    segmentsStart, segmentsEnd,
    [&](Segment segment)
    {
      return connections.cellForSegment(segment) == cell;
    });
  const auto cellEnd = std::find_if(
    cellBegin, segmentsEnd,
    [&](Segment segment)
    {
      return connections.cellForSegment(segment) != cell;
    });
  return std::make_tuple(cellBegin, cellEnd);
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
  const vector<CellIdx>& reinforceCandidatesInternal,
  size_t reinforceCandidatesExternalSize,
  const CellIdx reinforceCandidatesExternal[],
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
        reinforceCandidatesInternal.begin(),
        reinforceCandidatesInternal.end(),
        synapseData.presynapticCell);
    }
    else
    {
      isActive = std::binary_search(
        reinforceCandidatesExternal,
        reinforceCandidatesExternal + reinforceCandidatesExternalSize,
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
  const vector<CellIdx>& growthCandidatesInternal,
  size_t growthCandidatesExternalSize,
  const CellIdx growthCandidatesExternal[],
  Permanence initialPermanence)
{
  // It's possible to optimize this, swapping candidates to the end as
  // they're used. But this is awkward to mimic in other
  // implementations, especially because it requires iterating over
  // the existing synapses in a particular order.

  vector<CellIdx> candidates;
  candidates.reserve(growthCandidatesInternal.size() + growthCandidatesExternalSize);
  candidates.insert(candidates.begin(), growthCandidatesInternal.begin(),
                    growthCandidatesInternal.end());
  for (size_t i = 0; i < growthCandidatesExternalSize; i++)
  {
    const CellIdx cell = growthCandidatesExternal[i];
    candidates.push_back(cell + connections.numCells());
  }
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

  // Pick nActual cells randomly.
  for (UInt32 c = 0; c < nActual; c++)
  {
    size_t i = rng.getUInt32(candidates.size());
    connections.createSynapse(segment, candidates[i], initialPermanence);
    candidates.erase(candidates.begin() + i);
  }
}

static void learnOnCell(
  Connections& connections,
  Random& rng,
  CellIdx cell,
  vector<Segment>::const_iterator cellActiveSegmentsBegin,
  vector<Segment>::const_iterator cellActiveSegmentsEnd,
  vector<Segment>::const_iterator cellMatchingSegmentsBegin,
  vector<Segment>::const_iterator cellMatchingSegmentsEnd,
  const vector<CellIdx>& reinforceCandidatesInternal,
  size_t reinforceCandidatesExternalSize,
  const CellIdx reinforceCandidatesExternal[],
  const vector<CellIdx>& growthCandidatesInternal,
  size_t growthCandidatesExternalSize,
  const CellIdx growthCandidatesExternal[],
  const vector<UInt32>& numActivePotentialSynapsesForSegment,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  if (cellActiveSegmentsBegin != cellActiveSegmentsEnd)
  {
    // Learn on every active segment.

    auto activeSegment = cellActiveSegmentsBegin;
    do
    {
      adaptSegment(connections,
                   *activeSegment,
                   reinforceCandidatesInternal,
                   reinforceCandidatesExternalSize, reinforceCandidatesExternal,
                   permanenceIncrement, permanenceDecrement);

      const Int32 nGrowDesired = maxNewSynapseCount -
        numActivePotentialSynapsesForSegment[activeSegment->flatIdx];
        if (nGrowDesired > 0)
        {
          growSynapses(connections, rng,
                       *activeSegment, nGrowDesired,
                       growthCandidatesInternal,
                       growthCandidatesExternalSize, growthCandidatesExternal,
                       initialPermanence);
        }
    } while (++activeSegment != cellActiveSegmentsEnd);
  }
  else if (cellMatchingSegmentsBegin != cellMatchingSegmentsEnd)
  {
    // No active segments.
    // Learn on the best matching segment.

    const Segment bestMatchingSegment = *std::max_element(
      cellMatchingSegmentsBegin, cellMatchingSegmentsEnd,
      [&](Segment a, Segment b)
      {
        return (numActivePotentialSynapsesForSegment[a.flatIdx] <
                numActivePotentialSynapsesForSegment[b.flatIdx]);
      });

    adaptSegment(connections,
                 bestMatchingSegment,
                 reinforceCandidatesInternal,
                 reinforceCandidatesExternalSize, reinforceCandidatesExternal,
                 permanenceIncrement, permanenceDecrement);

    const Int32 nGrowDesired = maxNewSynapseCount -
      numActivePotentialSynapsesForSegment[bestMatchingSegment.flatIdx];
    if (nGrowDesired > 0)
    {
      growSynapses(connections, rng,
                   bestMatchingSegment, nGrowDesired,
                   growthCandidatesInternal,
                   growthCandidatesExternalSize, growthCandidatesExternal,
                   initialPermanence);
    }
  }
  else
  {
    // No matching segments.
    // Grow a new segment and learn on it.

    // Don't grow a segment that will never match.
    const UInt32 nGrowExact = std::min(maxNewSynapseCount,
                                       (UInt)(growthCandidatesInternal.size() +
                                              growthCandidatesExternalSize));
    if (nGrowExact > 0)
    {
      const Segment segment = connections.createSegment(cell);
      growSynapses(connections, rng,
                   segment, nGrowExact,
                   growthCandidatesInternal,
                   growthCandidatesExternalSize, growthCandidatesExternal,
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
  vector<Segment>::const_iterator columnActiveBasalBegin,
  vector<Segment>::const_iterator columnActiveBasalEnd,
  vector<Segment>::const_iterator columnMatchingBasalBegin,
  vector<Segment>::const_iterator columnMatchingBasalEnd,
  vector<Segment>::const_iterator columnActiveApicalBegin,
  vector<Segment>::const_iterator columnActiveApicalEnd,
  vector<Segment>::const_iterator columnMatchingApicalBegin,
  vector<Segment>::const_iterator columnMatchingApicalEnd,
  UInt32 predictiveThreshold,
  const vector<CellIdx>& reinforceCandidatesInternal,
  size_t reinforceCandidatesExternalBasalSize,
  const CellIdx reinforceCandidatesExternalBasal[],
  size_t reinforceCandidatesExternalApicalSize,
  const CellIdx reinforceCandidatesExternalApical[],
  const vector<CellIdx>& growthCandidatesInternal,
  size_t growthCandidatesExternalBasalSize,
  const CellIdx growthCandidatesExternalBasal[],
  size_t growthCandidatesExternalApicalSize,
  const CellIdx growthCandidatesExternalApical[],
  const vector<UInt32>& numActivePotentialSynapsesForBasalSegment,
  const vector<UInt32>& numActivePotentialSynapsesForApicalSegment,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  bool formInternalBasalConnections,
  bool learn)
{
  const auto cellForBasalSegment = [&](Segment segment)
    { return basalConnections.cellForSegment(segment); };
  const auto cellForApicalSegment = [&](Segment segment)
    { return apicalConnections.cellForSegment(segment); };

  for (auto& cellData : iterGroupBy(
         columnActiveBasalBegin, columnActiveBasalEnd, cellForBasalSegment,
         columnMatchingBasalBegin, columnMatchingBasalEnd, cellForBasalSegment,
         columnActiveApicalBegin, columnActiveApicalEnd, cellForApicalSegment,
         columnMatchingApicalBegin, columnMatchingApicalEnd, cellForApicalSegment))
  {
    CellIdx cell;
    vector<Segment>::const_iterator
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
                    cell,
                    cellActiveBasalBegin, cellActiveBasalEnd,
                    cellMatchingBasalBegin, cellMatchingBasalEnd,
                    reinforceCandidatesInternal,
                    reinforceCandidatesExternalBasalSize,
                    reinforceCandidatesExternalBasal,
                    (formInternalBasalConnections
                     ? growthCandidatesInternal : CELLS_NONE),
                    growthCandidatesExternalBasalSize,
                    growthCandidatesExternalBasal,
                    numActivePotentialSynapsesForBasalSegment,
                    maxNewSynapseCount, initialPermanence,
                    permanenceIncrement, permanenceDecrement);

        learnOnCell(apicalConnections, rng,
                    cell,
                    cellActiveApicalBegin, cellActiveApicalEnd,
                    cellMatchingApicalBegin, cellMatchingApicalEnd,
                    reinforceCandidatesInternal,
                    reinforceCandidatesExternalApicalSize,
                    reinforceCandidatesExternalApical,
                    CELLS_NONE,
                    growthCandidatesExternalApicalSize,
                    growthCandidatesExternalApical,
                    numActivePotentialSynapsesForApicalSegment,
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
  vector<Segment>::const_iterator columnActiveBasalBegin,
  vector<Segment>::const_iterator columnActiveBasalEnd,
  vector<Segment>::const_iterator columnMatchingBasalBegin,
  vector<Segment>::const_iterator columnMatchingBasalEnd,
  vector<Segment>::const_iterator columnActiveApicalBegin,
  vector<Segment>::const_iterator columnActiveApicalEnd,
  vector<Segment>::const_iterator columnMatchingApicalBegin,
  vector<Segment>::const_iterator columnMatchingApicalEnd,
  const vector<CellIdx>& reinforceCandidatesInternal,
  size_t reinforceCandidatesExternalBasalSize,
  const CellIdx reinforceCandidatesExternalBasal[],
  size_t reinforceCandidatesExternalApicalSize,
  const CellIdx reinforceCandidatesExternalApical[],
  const vector<CellIdx>& growthCandidatesInternal,
  size_t growthCandidatesExternalBasalSize,
  const CellIdx growthCandidatesExternalBasal[],
  size_t growthCandidatesExternalApicalSize,
  const CellIdx growthCandidatesExternalApical[],
  const vector<UInt32>& numActivePotentialSynapsesForBasalSegment,
  const vector<UInt32>& numActivePotentialSynapsesForApicalSegment,
  UInt cellsPerColumn,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
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
      auto bestBasalSegment = std::max_element(
        columnMatchingBasalBegin, columnMatchingBasalEnd,
        [&](Segment a, Segment b)
        {
          return (numActivePotentialSynapsesForBasalSegment[a.flatIdx] <
                  numActivePotentialSynapsesForBasalSegment[b.flatIdx]);
        });

      basalCandidatesBegin = bestBasalSegment;
      basalCandidatesEnd = bestBasalSegment + 1;

      winnerCell = basalConnections.cellForSegment(*bestBasalSegment);
    }
    else
    {
      winnerCell = getLeastUsedCell(rng, column, basalConnections,
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
    vector<Segment>::const_iterator
      cellActiveBasalBegin, cellActiveBasalEnd,
      cellMatchingBasalBegin, cellMatchingBasalEnd,
      cellActiveApicalBegin, cellActiveApicalEnd,
      cellMatchingApicalBegin, cellMatchingApicalEnd;
    tie(cellActiveBasalBegin,
        cellActiveBasalEnd) = segmentsForCell(columnActiveBasalBegin,
                                              columnActiveBasalEnd,
                                              winnerCell,
                                              basalConnections);
    tie(cellMatchingBasalBegin,
        cellMatchingBasalEnd) = segmentsForCell(basalCandidatesBegin,
                                                basalCandidatesEnd,
                                                winnerCell,
                                                basalConnections);
    tie(cellActiveApicalBegin,
        cellActiveApicalEnd) = segmentsForCell(columnActiveApicalBegin,
                                               columnActiveApicalEnd,
                                               winnerCell,
                                               apicalConnections);
    tie(cellMatchingApicalBegin,
        cellMatchingApicalEnd) = segmentsForCell(columnMatchingApicalBegin,
                                                 columnMatchingApicalEnd,
                                                 winnerCell,
                                                 apicalConnections);

    learnOnCell(basalConnections, rng,
                winnerCell,
                cellActiveBasalBegin, cellActiveBasalEnd,
                cellMatchingBasalBegin, cellMatchingBasalEnd,
                reinforceCandidatesInternal,
                reinforceCandidatesExternalBasalSize,
                reinforceCandidatesExternalBasal,
                (formInternalBasalConnections
                 ? growthCandidatesInternal : CELLS_NONE),
                growthCandidatesExternalBasalSize,
                growthCandidatesExternalBasal,
                numActivePotentialSynapsesForBasalSegment,
                maxNewSynapseCount, initialPermanence,
                permanenceIncrement, permanenceDecrement);

    learnOnCell(apicalConnections, rng,
                winnerCell,
                cellActiveApicalBegin, cellActiveApicalEnd,
                cellMatchingApicalBegin, cellMatchingApicalEnd,
                reinforceCandidatesInternal,
                reinforceCandidatesExternalApicalSize,
                reinforceCandidatesExternalApical,
                CELLS_NONE,
                growthCandidatesExternalApicalSize,
                growthCandidatesExternalApical,
                numActivePotentialSynapsesForApicalSegment,
                maxNewSynapseCount, initialPermanence,
                permanenceIncrement, permanenceDecrement);
  }
}

static void punishPredictedColumn(
  Connections& connections,
  vector<Segment>::const_iterator matchingSegmentsBegin,
  vector<Segment>::const_iterator matchingSegmentsEnd,
  const vector<CellIdx>& reinforceCandidatesInternal,
  size_t reinforceCandidatesExternalSize,
  const CellIdx reinforceCandidatesExternal[],
  Permanence predictedSegmentDecrement)
{
  if (predictedSegmentDecrement > 0.0)
  {
    for (auto matchingSegment = matchingSegmentsBegin;
         matchingSegment != matchingSegmentsEnd; matchingSegment++)
    {
      adaptSegment(connections, *matchingSegment,
                   reinforceCandidatesInternal,
                   reinforceCandidatesExternalSize, reinforceCandidatesExternal,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void ExtendedTemporalMemory::activateCells(
  size_t activeColumnsSize,
  const UInt activeColumns[],

  size_t reinforceCandidatesExternalBasalSize,
  const CellIdx reinforceCandidatesExternalBasal[],

  size_t reinforceCandidatesExternalApicalSize,
  const CellIdx reinforceCandidatesExternalApical[],

  size_t growthCandidatesExternalBasalSize,
  const CellIdx growthCandidatesExternalBasal[],

  size_t growthCandidatesExternalApicalSize,
  const CellIdx growthCandidatesExternalApical[],

  bool learn)
{
  if (checkInputs_)
  {
    NTA_CHECK(isSortedWithoutDuplicates(activeColumns,
                                        activeColumns + activeColumnsSize))
      << "activeColumns must be sorted without duplicates.";
    NTA_CHECK(isSortedWithoutDuplicates(reinforceCandidatesExternalBasal,
                                        reinforceCandidatesExternalBasal +
                                        reinforceCandidatesExternalBasalSize))
      << "reinforceCandidatesExternalBasal must be sorted without duplicates.";
    NTA_CHECK(isSortedWithoutDuplicates(reinforceCandidatesExternalApical,
                                        reinforceCandidatesExternalApical +
                                        reinforceCandidatesExternalApicalSize))
      << "reinforceCandidatesExternalApical must be sorted without duplicates.";
    NTA_CHECK(isSortedWithoutDuplicates(growthCandidatesExternalBasal,
                                        growthCandidatesExternalBasal +
                                        growthCandidatesExternalBasalSize))
      << "growthCandidatesExternalBasal must be sorted without duplicates.";
    NTA_CHECK(isSortedWithoutDuplicates(growthCandidatesExternalApical,
                                        growthCandidatesExternalApical +
                                        growthCandidatesExternalApicalSize))
      << "growthCandidatesExternalApical must be sorted without duplicates.";

    NTA_CHECK(std::all_of(activeColumns,
                          activeColumns + activeColumnsSize,
                          [&](UInt c) { return c < numColumns_; }))
      << "Values in activeColumns must be within the range "
      << "[0," << numColumns_ << ").";
    NTA_CHECK(std::all_of(
                reinforceCandidatesExternalBasal,
                reinforceCandidatesExternalBasal +
                reinforceCandidatesExternalBasalSize,
                [&](UInt c) { return c < numBasalInputs_; }))
      << "Values in reinforceCandidatesExternalBasal must be within the range "
      << "[0," << numBasalInputs_ << ").";
    NTA_CHECK(std::all_of(
                reinforceCandidatesExternalApical,
                reinforceCandidatesExternalApical +
                reinforceCandidatesExternalApicalSize,
                [&](UInt c) { return c < numApicalInputs_; }))
      << "Values in reinforceCandidatesExternalApical must be within the range "
      << "[0," << numApicalInputs_ << ").";
    NTA_CHECK(std::all_of(
                growthCandidatesExternalBasal,
                growthCandidatesExternalBasal +
                growthCandidatesExternalBasalSize,
                [&](UInt c) { return c < numBasalInputs_; }))
      << "Values in growthCandidatesExternalBasal must be within the range " <<
      "[0," << numBasalInputs_ << ").";
    NTA_CHECK(std::all_of(
                growthCandidatesExternalApical,
                growthCandidatesExternalApical +
                growthCandidatesExternalApicalSize,
                [&](UInt c) { return c < numApicalInputs_; }))
      << "Values in growthCandidatesExternalApical must be within the range "
      << "[0," << numApicalInputs_ << ").";
  }

  const vector<CellIdx> reinforceCandidatesInternal = std::move(activeCells_);
  const vector<CellIdx> growthCandidatesInternal = std::move(winnerCells_);

  const auto columnForBasalSegment = [&](Segment segment)
    { return basalConnections.cellForSegment(segment) / cellsPerColumn_; };
  const auto columnForApicalSegment = [&](Segment segment)
    { return apicalConnections.cellForSegment(segment) / cellsPerColumn_; };

  for (auto& columnData : iterGroupBy(
         activeColumns,
         activeColumns + activeColumnsSize, identity<UInt>,
         activeBasalSegments_.begin(),
         activeBasalSegments_.end(), columnForBasalSegment,
         matchingBasalSegments_.begin(),
         matchingBasalSegments_.end(), columnForBasalSegment,
         activeApicalSegments_.begin(),
         activeApicalSegments_.end(), columnForApicalSegment,
         matchingApicalSegments_.begin(),
         matchingApicalSegments_.end(), columnForApicalSegment))
  {
    UInt column;
    const UInt* activeColumnsBegin;
    const UInt* activeColumnsEnd;
    vector<Segment>::const_iterator
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

      const auto cellForBasalSegment = [&](Segment segment)
        { return basalConnections.cellForSegment(segment); };
      const auto cellForApicalSegment = [&](Segment segment)
        { return apicalConnections.cellForSegment(segment); };

      for (auto& cellData : iterGroupBy(
             columnActiveBasalBegin, columnActiveBasalEnd, cellForBasalSegment,
             columnActiveApicalBegin, columnActiveApicalEnd, cellForApicalSegment))
      {
        CellIdx cell;
        vector<Segment>::const_iterator
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
          reinforceCandidatesInternal,
          reinforceCandidatesExternalBasalSize,
          reinforceCandidatesExternalBasal,
          reinforceCandidatesExternalApicalSize,
          reinforceCandidatesExternalApical,
          growthCandidatesInternal,
          growthCandidatesExternalBasalSize,
          growthCandidatesExternalBasal,
          growthCandidatesExternalApicalSize,
          growthCandidatesExternalApical,
          numActivePotentialSynapsesForBasalSegment_,
          numActivePotentialSynapsesForApicalSegment_,
          maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
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
          reinforceCandidatesInternal,
          reinforceCandidatesExternalBasalSize,
          reinforceCandidatesExternalBasal,
          reinforceCandidatesExternalApicalSize,
          reinforceCandidatesExternalApical,
          growthCandidatesInternal,
          growthCandidatesExternalBasalSize,
          growthCandidatesExternalBasal,
          growthCandidatesExternalApicalSize,
          growthCandidatesExternalApical,
          numActivePotentialSynapsesForBasalSegment_,
          numActivePotentialSynapsesForApicalSegment_,
          cellsPerColumn_, maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
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
          reinforceCandidatesInternal,
          reinforceCandidatesExternalBasalSize,
          reinforceCandidatesExternalBasal,
          predictedSegmentDecrement_);

        // Don't punish apical segments.
      }
    }
  }
}

static void calculateExcitation(
  vector<UInt32>& numActiveConnectedSynapsesForSegment,
  vector<Segment>& activeSegments,
  vector<UInt32>& numActivePotentialSynapsesForSegment,
  vector<Segment>& matchingSegments,
  Connections& connections,
  const vector<CellIdx>& activeCells,
  size_t externalActiveCellsSize,
  const CellIdx externalActiveCells[],
  Permanence connectedPermanence,
  UInt activationThreshold,
  UInt minThreshold,
  bool learn)
{
  const UInt32 length = connections.segmentFlatListLength();
  numActiveConnectedSynapsesForSegment.assign(length, 0);
  numActivePotentialSynapsesForSegment.assign(length, 0);

  connections.computeActivity(numActiveConnectedSynapsesForSegment,
                              numActivePotentialSynapsesForSegment,
                              activeCells,
                              connectedPermanence);

  for (size_t i = 0; i < externalActiveCellsSize; i++)
  {
    const CellIdx cell = externalActiveCells[i];
    connections.computeActivity(numActiveConnectedSynapsesForSegment,
                                numActivePotentialSynapsesForSegment,
                                cell + connections.numCells(),
                                connectedPermanence);
  }

  // Active segments, connected synapses.
  activeSegments.clear();
  for (size_t i = 0; i < numActiveConnectedSynapsesForSegment.size(); i++)
  {
    if (numActiveConnectedSynapsesForSegment[i] >= activationThreshold)
    {
      activeSegments.push_back(connections.segmentForFlatIdx(i));
    }
  }
  std::sort(activeSegments.begin(), activeSegments.end(),
            [&](Segment a, Segment b)
            {
              return connections.compareSegments(a, b);
            });

  // Matching segments, potential synapses.
  matchingSegments.clear();
  for (size_t i = 0; i < numActivePotentialSynapsesForSegment.size(); i++)
  {
    if (numActivePotentialSynapsesForSegment[i] >= minThreshold)
    {
      matchingSegments.push_back(connections.segmentForFlatIdx(i));
    }
  }
  std::sort(matchingSegments.begin(), matchingSegments.end(),
            [&](Segment a, Segment b)
            {
              return connections.compareSegments(a, b);
            });

  if (learn)
  {
    for (Segment segment : activeSegments)
    {
      connections.recordSegmentActivity(segment);
    }

    connections.startNewIteration();
  }
}

void ExtendedTemporalMemory::depolarizeCells(
  size_t activeCellsExternalBasalSize,
  const CellIdx activeCellsExternalBasal[],

  size_t activeCellsExternalApicalSize,
  const CellIdx activeCellsExternalApical[],

  bool learn)
{
  if (checkInputs_)
  {
    NTA_CHECK(std::all_of(
                activeCellsExternalBasal,
                activeCellsExternalBasal +
                activeCellsExternalBasalSize,
                [&](UInt c) { return c < numBasalInputs_; }))
      << "Values in activeCellsExternalBasal must be within the range [0,"
      << numBasalInputs_ << ").";
    NTA_CHECK(std::all_of(
                activeCellsExternalApical,
                activeCellsExternalApical +
                activeCellsExternalApicalSize,
                [&](UInt c) { return c < numApicalInputs_; }))
      << "Values in activeCellsExternalApical must be within the range [0,"
      << numApicalInputs_ << ").";
  }

  calculateExcitation(
    numActiveConnectedSynapsesForBasalSegment_, activeBasalSegments_,
    numActivePotentialSynapsesForBasalSegment_, matchingBasalSegments_,
    basalConnections,
    activeCells_, activeCellsExternalBasalSize, activeCellsExternalBasal,
    connectedPermanence_, activationThreshold_, minThreshold_,
    learn);

  calculateExcitation(
    numActiveConnectedSynapsesForApicalSegment_, activeApicalSegments_,
    numActivePotentialSynapsesForApicalSegment_, matchingApicalSegments_,
    apicalConnections,
    activeCells_, activeCellsExternalApicalSize, activeCellsExternalApical,
    connectedPermanence_, activationThreshold_, minThreshold_,
    learn);
}

void ExtendedTemporalMemory::compute(
  size_t activeColumnsSize,
  const UInt activeColumns[],

  size_t activeCellsExternalBasalSize,
  const CellIdx activeCellsExternalBasal[],

  size_t activeCellsExternalApicalSize,
  const CellIdx activeCellsExternalApical[],

  size_t reinforceCandidatesExternalBasalSize,
  const CellIdx reinforceCandidatesExternalBasal[],

  size_t reinforceCandidatesExternalApicalSize,
  const CellIdx reinforceCandidatesExternalApical[],

  size_t growthCandidatesExternalBasalSize,
  const CellIdx growthCandidatesExternalBasal[],

  size_t growthCandidatesExternalApicalSize,
  const CellIdx growthCandidatesExternalApical[],

  bool learn)
{
  activateCells(activeColumnsSize,
                activeColumns,
                reinforceCandidatesExternalBasalSize,
                reinforceCandidatesExternalBasal,
                reinforceCandidatesExternalApicalSize,
                reinforceCandidatesExternalApical,
                growthCandidatesExternalBasalSize,
                growthCandidatesExternalBasal,
                growthCandidatesExternalApicalSize,
                growthCandidatesExternalApical,
                learn);

  depolarizeCells(activeCellsExternalBasalSize,
                  activeCellsExternalBasal,
                  activeCellsExternalApicalSize,
                  activeCellsExternalApical,
                  learn);
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
//  Helper methods
// ==============================

Int ExtendedTemporalMemory::columnForCell(CellIdx cell)
{
  _validateCell(cell);

  return cell / cellsPerColumn_;
}

vector<CellIdx> ExtendedTemporalMemory::cellsForColumn(UInt column)
{
  NTA_CHECK(column < numberOfColumns()) << "Invalid column " << column;

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

  const auto columnForBasalSegment = [&](Segment segment)
    { return basalConnections.cellForSegment(segment) / cellsPerColumn_; };
  const auto columnForApicalSegment = [&](Segment segment)
    { return apicalConnections.cellForSegment(segment) / cellsPerColumn_; };

  for (auto& columnData : groupBy(activeBasalSegments_, columnForBasalSegment,
                                  activeApicalSegments_, columnForApicalSegment))
  {
    UInt column;
    vector<Segment>::const_iterator
      columnActiveBasalBegin, columnActiveBasalEnd,
      columnActiveApicalBegin, columnActiveApicalEnd;
    tie(column,
        columnActiveBasalBegin, columnActiveBasalEnd,
        columnActiveApicalBegin, columnActiveApicalEnd) = columnData;

    const auto cellForBasalSegment = [&](Segment segment)
      { return basalConnections.cellForSegment(segment); };
    const auto cellForApicalSegment = [&](Segment segment)
      { return apicalConnections.cellForSegment(segment); };

    const auto groupedByCell = iterGroupBy(
      columnActiveBasalBegin, columnActiveBasalEnd, cellForBasalSegment,
      columnActiveApicalBegin, columnActiveApicalEnd, cellForApicalSegment);

    UInt32 maxDepolarization = 0;
    for (auto& cellData : groupedByCell)
    {
      vector<Segment>::const_iterator
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
        vector<Segment>::const_iterator
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
  return activeBasalSegments_;
}

vector<Segment> ExtendedTemporalMemory::getMatchingBasalSegments() const
{
  return matchingBasalSegments_;
}

vector<Segment> ExtendedTemporalMemory::getActiveApicalSegments() const
{
  return activeApicalSegments_;
}

vector<Segment> ExtendedTemporalMemory::getMatchingApicalSegments() const
{
  return matchingApicalSegments_;
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

vector<UInt> ExtendedTemporalMemory::getBasalInputDimensions() const
{
  return basalInputDimensions_;
}

vector<UInt> ExtendedTemporalMemory::getApicalInputDimensions() const
{
  return apicalInputDimensions_;
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

bool ExtendedTemporalMemory::getCheckInputs() const
{
  return checkInputs_;
}

void ExtendedTemporalMemory::setCheckInputs(bool checkInputs)
{
  checkInputs_ = checkInputs;
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
  for (Segment segment : activeBasalSegments_)
  {
    const CellIdx cell = basalConnections.cellForSegment(segment);
    const vector<Segment>& segments = basalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActiveConnectedSynapsesForBasalSegment_[segment.flatIdx]
              << " ";
  }
  outStream << endl;

  outStream << matchingBasalSegments_.size() << " ";
  for (Segment segment : matchingBasalSegments_)
  {
    const CellIdx cell = basalConnections.cellForSegment(segment);
    const vector<Segment>& segments = basalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActivePotentialSynapsesForBasalSegment_[segment.flatIdx]
              << " ";
  }
  outStream << endl;

  outStream << activeApicalSegments_.size() << " ";
  for (Segment segment : activeApicalSegments_)
  {
    const CellIdx cell = apicalConnections.cellForSegment(segment);
    const vector<Segment>& segments = apicalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActiveConnectedSynapsesForApicalSegment_[segment.flatIdx]
              << " ";
  }
  outStream << endl;

  outStream << matchingApicalSegments_.size() << " ";
  for (Segment segment : matchingApicalSegments_)
  {
    const CellIdx cell = apicalConnections.cellForSegment(segment);
    const vector<Segment>& segments = apicalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    outStream << idx << " ";
    outStream << cell << " ";
    outStream << numActivePotentialSynapsesForApicalSegment_[segment.flatIdx]
              << " ";
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
    const Segment segment = activeBasalSegments_[i];
    const CellIdx cell = basalConnections.cellForSegment(segment);
    const vector<Segment>& segments = basalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    activeBasalSegmentOverlaps[i].setCell(cell);
    activeBasalSegmentOverlaps[i].setSegment(idx);
    activeBasalSegmentOverlaps[i].setOverlap(
      numActiveConnectedSynapsesForBasalSegment_[segment.flatIdx]);
  }

  auto matchingBasalSegmentOverlaps =
    proto.initMatchingBasalSegmentOverlaps(matchingBasalSegments_.size());
  for (UInt i = 0; i < matchingBasalSegments_.size(); ++i)
  {
    const Segment segment = matchingBasalSegments_[i];
    const CellIdx cell = basalConnections.cellForSegment(segment);
    const vector<Segment>& segments = basalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    matchingBasalSegmentOverlaps[i].setCell(cell);
    matchingBasalSegmentOverlaps[i].setSegment(idx);
    matchingBasalSegmentOverlaps[i].setOverlap(
      numActivePotentialSynapsesForBasalSegment_[segment.flatIdx]);
  }

  auto activeApicalSegmentOverlaps =
    proto.initActiveApicalSegmentOverlaps(activeApicalSegments_.size());
  for (UInt i = 0; i < activeApicalSegments_.size(); ++i)
  {
    Segment segment = activeApicalSegments_[i];
    const CellIdx cell = apicalConnections.cellForSegment(segment);
    const vector<Segment>& segments = apicalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    activeApicalSegmentOverlaps[i].setCell(cell);
    activeApicalSegmentOverlaps[i].setSegment(idx);
    activeApicalSegmentOverlaps[i].setOverlap(
      numActiveConnectedSynapsesForApicalSegment_[segment.flatIdx]);
  }

  auto matchingApicalSegmentOverlaps =
    proto.initMatchingApicalSegmentOverlaps(matchingApicalSegments_.size());
  for (UInt i = 0; i < matchingApicalSegments_.size(); ++i)
  {
    Segment segment = matchingApicalSegments_[i];
    const CellIdx cell = apicalConnections.cellForSegment(segment);
    const vector<Segment>& segments = apicalConnections.segmentsForCell(cell);

    SegmentIdx idx = std::distance(
      segments.begin(),
      std::find(segments.begin(), segments.end(), segment));

    matchingApicalSegmentOverlaps[i].setCell(cell);
    matchingApicalSegmentOverlaps[i].setSegment(idx);
    matchingApicalSegmentOverlaps[i].setOverlap(
      numActivePotentialSynapsesForApicalSegment_[segment.flatIdx]);
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

  numActiveConnectedSynapsesForBasalSegment_.assign(
    basalConnections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForBasalSegment_.assign(
    basalConnections.segmentFlatListLength(), 0);

  numActiveConnectedSynapsesForApicalSegment_.assign(
    apicalConnections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForApicalSegment_.assign(
    apicalConnections.segmentFlatListLength(), 0);

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
    Segment segment = basalConnections.getSegment(value.getCell(),
                                                  value.getSegment());

    activeBasalSegments_.push_back(segment);
    numActiveConnectedSynapsesForBasalSegment_[segment.flatIdx] =
      value.getOverlap();
  }

  matchingBasalSegments_.clear();
  for (auto value : proto.getMatchingBasalSegmentOverlaps())
  {
    Segment segment = basalConnections.getSegment(value.getCell(),
                                                  value.getSegment());

    matchingBasalSegments_.push_back(segment);
    numActivePotentialSynapsesForBasalSegment_[segment.flatIdx] =
      value.getOverlap();
  }

  activeApicalSegments_.clear();
  for (auto value : proto.getActiveApicalSegmentOverlaps())
  {
    Segment segment = apicalConnections.getSegment(value.getCell(),
                                                   value.getSegment());

    activeApicalSegments_.push_back(segment);
    numActiveConnectedSynapsesForApicalSegment_[segment.flatIdx] =
      value.getOverlap();
  }

  matchingApicalSegments_.clear();
  for (auto value : proto.getMatchingApicalSegmentOverlaps())
  {
    Segment segment = apicalConnections.getSegment(value.getCell(),
                                                   value.getSegment());

    matchingApicalSegments_.push_back(segment);
    numActivePotentialSynapsesForApicalSegment_[segment.flatIdx] =
      value.getOverlap();
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

  numActiveConnectedSynapsesForBasalSegment_.assign(
    basalConnections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForBasalSegment_.assign(
    basalConnections.segmentFlatListLength(), 0);

  numActiveConnectedSynapsesForApicalSegment_.assign(
    apicalConnections.segmentFlatListLength(), 0);
  numActivePotentialSynapsesForApicalSegment_.assign(
    apicalConnections.segmentFlatListLength(), 0);

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
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = basalConnections.getSegment(cellIdx, idx);
    activeBasalSegments_[i] = segment;

    inStream >> numActiveConnectedSynapsesForBasalSegment_[segment.flatIdx];
  }

  UInt numMatchingBasalSegments;
  inStream >> numMatchingBasalSegments;
  matchingBasalSegments_.resize(numMatchingBasalSegments);
  for (UInt i = 0; i < numMatchingBasalSegments; i++)
  {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = basalConnections.getSegment(cellIdx, idx);
    matchingBasalSegments_[i] = segment;

    inStream >> numActivePotentialSynapsesForBasalSegment_[segment.flatIdx];
  }

  UInt numActiveApicalSegments;
  inStream >> numActiveApicalSegments;
  activeApicalSegments_.resize(numActiveApicalSegments);
  for (UInt i = 0; i < numActiveApicalSegments; i++)
  {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = apicalConnections.getSegment(cellIdx, idx);
    activeApicalSegments_[i] = segment;

    inStream >> numActiveConnectedSynapsesForApicalSegment_[segment.flatIdx];
  }

  UInt numMatchingApicalSegments;
  inStream >> numMatchingApicalSegments;
  matchingApicalSegments_.resize(numMatchingApicalSegments);
  for (UInt i = 0; i < numMatchingApicalSegments; i++)
  {
    SegmentIdx idx;
    inStream >> idx;

    CellIdx cellIdx;
    inStream >> cellIdx;

    Segment segment = apicalConnections.getSegment(cellIdx, idx);
    matchingApicalSegments_[i] = segment;

    inStream >> numActivePotentialSynapsesForApicalSegment_[segment.flatIdx];
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
