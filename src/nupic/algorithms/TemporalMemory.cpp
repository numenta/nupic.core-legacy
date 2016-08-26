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
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/utils/GroupBy.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

static const Permanence EPSILON = 0.000001;
static const UInt TM_VERSION = 2;



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
  connections = Connections(
    numberOfCells(),
    maxSegmentsPerCell,
    maxSynapsesPerSegment);
  seed_((UInt64)(seed < 0 ? rand() : seed));

  activeCells_.clear();
  winnerCells_.clear();
  activeSegments_.clear();
  matchingSegments_.clear();
}

static CellIdx cellForSegment(const SegmentOverlap& s)
{
  return s.segment.cell;
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
  Permanence permanenceIncrement,
  Permanence permanenceDecrement)
{
  vector<Synapse> synapses = connections.synapsesForSegment(segment);

  for (Synapse synapse : synapses)
  {
    const SynapseData synapseData = connections.dataForSynapse(synapse);
    const bool isActive =
      std::binary_search(prevActiveCells.begin(), prevActiveCells.end(),
                         synapseData.presynapticCell);
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
  const vector<CellIdx>& prevWinnerCells,
  Permanence initialPermanence)
{
  vector<CellIdx> candidates(prevWinnerCells.begin(), prevWinnerCells.end());

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

static void activatePredictedColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  Random& rng,
  vector<SegmentOverlap>::const_iterator columnActiveSegmentsBegin,
  vector<SegmentOverlap>::const_iterator columnActiveSegmentsEnd,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsEnd,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevWinnerCells,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  bool learn)
{
  for (auto& cellData : iterGroupBy(
         columnActiveSegmentsBegin, columnActiveSegmentsEnd, cellForSegment,
         columnMatchingSegmentsBegin, columnMatchingSegmentsEnd, cellForSegment))
  {
    CellIdx cell;
    vector<SegmentOverlap>::const_iterator
      cellActiveSegmentsBegin, cellActiveSegmentsEnd,
      cellMatchingSegmentsBegin, cellMatchingSegmentsEnd;
    tie(cell,
        cellActiveSegmentsBegin, cellActiveSegmentsEnd,
        cellMatchingSegmentsBegin, cellMatchingSegmentsEnd) = cellData;

    if (cellActiveSegmentsBegin != cellActiveSegmentsEnd)
    {
      activeCells.push_back(cell);
      winnerCells.push_back(cell);

      if (learn)
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
            NTA_ASSERT(std::distance(activeOverlapsBegin,
                                     activeOverlapsEnd) == 1);
            NTA_ASSERT(std::distance(matchingOverlapsBegin,
                                     matchingOverlapsEnd) == 1);

            adaptSegment(connections,
                         segment,
                         prevActiveCells,
                         permanenceIncrement, permanenceDecrement);

            const Int32 nActivePotentialSynapses =
              matchingOverlapsBegin->overlap;
            const Int32 nGrowDesired =
              maxNewSynapseCount - nActivePotentialSynapses;
            if (nGrowDesired > 0)
            {
              growSynapses(connections, rng,
                           segment, nGrowDesired,
                           prevWinnerCells,
                           initialPermanence);
            }
          }
        }
      }
    }
  }
}

static void burstColumn(
  vector<CellIdx>& activeCells,
  vector<CellIdx>& winnerCells,
  Connections& connections,
  Random& rng,
  UInt column,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsEnd,
  const vector<CellIdx>& prevActiveCells,
  const vector<CellIdx>& prevWinnerCells,
  UInt cellsPerColumn,
  UInt maxNewSynapseCount,
  Permanence initialPermanence,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  bool learn)
{
  // Calculate the active cells.
  const CellIdx start = column * cellsPerColumn;
  const CellIdx end = start + cellsPerColumn;
  for (CellIdx cell = start; cell < end; cell++)
  {
    activeCells.push_back(cell);
  }

  const auto bestMatching = std::max_element(
    columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
    [](const SegmentOverlap& a, const SegmentOverlap& b)
    {
      return a.overlap < b.overlap;
    });

  const CellIdx winnerCell = (bestMatching != columnMatchingSegmentsEnd)
    ? bestMatching->segment.cell
    : getLeastUsedCell(connections, rng, column, cellsPerColumn);

  winnerCells.push_back(winnerCell);

  // Learn.
  if (learn)
  {
    if (bestMatching != columnMatchingSegmentsEnd)
    {
      // Learn on the best matching segment.
      adaptSegment(connections,
                   bestMatching->segment,
                   prevActiveCells,
                   permanenceIncrement, permanenceDecrement);

      const Int32 nGrowDesired = maxNewSynapseCount - bestMatching->overlap;
      if (nGrowDesired > 0)
      {
        growSynapses(connections, rng,
                     bestMatching->segment, nGrowDesired,
                     prevWinnerCells,
                     initialPermanence);
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
        const Segment segment = connections.createSegment(winnerCell);
        growSynapses(connections, rng,
                     segment, nGrowExact,
                     prevWinnerCells,
                     initialPermanence);
        NTA_ASSERT(connections.numSynapses(segment) == nGrowExact);
      }
    }
  }
}

static void punishPredictedColumn(
  Connections& connections,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsBegin,
  vector<SegmentOverlap>::const_iterator columnMatchingSegmentsEnd,
  const vector<CellIdx>& prevActiveCells,
  Permanence predictedSegmentDecrement)
{
  if (predictedSegmentDecrement > 0.0)
  {
    for (auto matching = columnMatchingSegmentsBegin;
         matching != columnMatchingSegmentsEnd; matching++)
    {
      adaptSegment(connections, matching->segment, prevActiveCells,
                   -predictedSegmentDecrement, 0.0);
    }
  }
}

void TemporalMemory::activateCells(
  const vector<UInt>& activeColumns,
  bool learn)
{
  NTA_ASSERT(std::is_sorted(activeColumns.begin(), activeColumns.end()));

  const vector<CellIdx> prevActiveCells = std::move(activeCells_);
  const vector<CellIdx> prevWinnerCells = std::move(winnerCells_);

  const auto columnForSegment =
    [&](const SegmentOverlap& s) { return s.segment.cell / cellsPerColumn_; };

  for (auto& columnData : groupBy(activeColumns, identity<UInt>,
                                  activeSegments_, columnForSegment,
                                  matchingSegments_, columnForSegment))
  {
    UInt column;
    vector<UInt>::const_iterator
      activeColumnsBegin, activeColumnsEnd;
    vector<SegmentOverlap>::const_iterator
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
          columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
          prevActiveCells, prevWinnerCells,
          maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
          learn);
      }
      else
      {
        burstColumn(
          activeCells_, winnerCells_, connections, rng_,
          column, columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
          prevActiveCells, prevWinnerCells,
          cellsPerColumn_, maxNewSynapseCount_,
          initialPermanence_, permanenceIncrement_, permanenceDecrement_,
          learn);
      }
    }
    else
    {
      if (learn)
      {
        punishPredictedColumn(
          connections,
          columnMatchingSegmentsBegin, columnMatchingSegmentsEnd,
          prevActiveCells,
          predictedSegmentDecrement_);
      }
    }
  }
}

void TemporalMemory::activateDendrites(bool learn)
{
  activeSegments_.clear();
  matchingSegments_.clear();
  connections.computeActivity(activeCells_,
                              connectedPermanence_, activationThreshold_,
                              0.0, minThreshold_,
                              activeSegments_, matchingSegments_);

  if (learn)
  {
    for (const SegmentOverlap& segmentOverlap : activeSegments_)
    {
      connections.recordSegmentActivity(segmentOverlap.segment);
    }

    connections.startNewIteration();
  }
}

void TemporalMemory::compute(
  UInt activeColumnsSize,
  const UInt activeColumnsUnsorted[],
  bool learn)
{
  vector<UInt> activeColumns(activeColumnsUnsorted,
                             activeColumnsUnsorted + activeColumnsSize);
  std::sort(activeColumns.begin(), activeColumns.end());

  activateCells(activeColumns, learn);

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
  return numberOfColumns() * cellsPerColumn_;
}

vector<CellIdx> TemporalMemory::getActiveCells() const
{
  return activeCells_;
}

vector<CellIdx> TemporalMemory::getPredictiveCells() const
{
  vector<CellIdx> predictiveCells;

  for (auto segOverlap = activeSegments_.begin();
       segOverlap != activeSegments_.end();
       segOverlap++)
  {
    if (segOverlap == activeSegments_.begin() ||
        segOverlap->segment.cell != predictiveCells.back())
    {
      predictiveCells.push_back(segOverlap->segment.cell);
    }
  }

  return predictiveCells;
}

vector<CellIdx> TemporalMemory::getWinnerCells() const
{
  return winnerCells_;
}

vector<CellIdx> TemporalMemory::getMatchingCells() const
{
  vector<CellIdx> matchingCells;

  for (auto segOverlap = matchingSegments_.begin();
       segOverlap != matchingSegments_.end();
       segOverlap++)
  {
    if (segOverlap == matchingSegments_.begin() ||
        segOverlap->segment.cell != matchingCells.back())
    {
      matchingCells.push_back(segOverlap->segment.cell);
    }
  }

  return matchingCells;
}

vector<Segment> TemporalMemory::getActiveSegments() const
{
  vector<Segment> ret;
  ret.reserve(activeSegments_.size());
  for (const SegmentOverlap& segmentOverlap : activeSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
}

vector<Segment> TemporalMemory::getMatchingSegments() const
{
  vector<Segment> ret;
  ret.reserve(matchingSegments_.size());
  for (const SegmentOverlap& segmentOverlap : matchingSegments_)
  {
    ret.push_back(segmentOverlap.segment);
  }
  return ret;
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
  // TODO: this won't scale!
  stringstream s;
  s.flags(ios::scientific);
  s.precision(numeric_limits<double>::digits10 + 1);
  this->save(s);
  return s.str().size();
}

void TemporalMemory::save(ostream& outStream) const
{
  // Write a starting marker and version.
  outStream << "TemporalMemory" << endl;
  outStream << TM_VERSION << endl;

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
    << endl;

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
  for (SegmentOverlap elem : activeSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << matchingSegments_.size() << " ";
  for (SegmentOverlap elem : matchingSegments_)
  {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell << " ";
    outStream << elem.overlap << " ";
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

  auto activeSegmentOverlaps =
    proto.initActiveSegmentOverlaps(activeSegments_.size());
  for (UInt i = 0; i < activeSegments_.size(); ++i)
  {
    Segment segment = activeSegments_[i].segment;
    activeSegmentOverlaps[i].setCell(segment.cell);
    activeSegmentOverlaps[i].setSegment(segment.idx);
    activeSegmentOverlaps[i].setOverlap(activeSegments_[i].overlap);
  }

  auto winnerCells = proto.initWinnerCells(winnerCells_.size());
  i = 0;
  for (CellIdx cell : winnerCells_)
  {
    winnerCells.set(i++, cell);
  }

  auto matchingSegmentOverlaps =
    proto.initMatchingSegmentOverlaps(matchingSegments_.size());
  for (UInt i = 0; i < matchingSegments_.size(); ++i)
  {
    Segment segment = matchingSegments_[i].segment;
    matchingSegmentOverlaps[i].setCell(segment.cell);
    matchingSegmentOverlaps[i].setSegment(segment.idx);
    matchingSegmentOverlaps[i].setOverlap(matchingSegments_[i].overlap);
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

  auto _connections = proto.getConnections();
  connections.read(_connections);

  auto random = proto.getRandom();
  rng_.read(random);

  activeCells_.clear();
  for (auto cell : proto.getActiveCells())
  {
    activeCells_.push_back(cell);
  }

  if (proto.getActiveSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "TemporalMemory::read :: Obsolete field 'activeSegments' isn't usable. "
             << "TemporalMemory results will be goofy for one timestep.";
  }

  activeSegments_.clear();
  for (auto value : proto.getActiveSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    activeSegments_.push_back({segment, value.getOverlap()});
  }

  winnerCells_.clear();
  for (auto cell : proto.getWinnerCells())
  {
    winnerCells_.push_back(cell);
  }

  if (proto.getMatchingSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "TemporalMemory::read :: Obsolete field 'matchingSegments' isn't usable. "
             << "TemporalMemory results will be goofy for one timestep.";
  }

  matchingSegments_.clear();
  for (auto value : proto.getMatchingSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(), value.getCell()};
    matchingSegments_.push_back({segment, value.getOverlap()});
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
    >> predictedSegmentDecrement_;

  connections.load(inStream);

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

  if (version < 2)
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments_.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments_[i].segment.idx;
      inStream >> activeSegments_[i].segment.cell;
      activeSegments_[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments_.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments_[i].segment.idx;
      inStream >> activeSegments_[i].segment.cell;
      inStream >> activeSegments_[i].overlap;
    }
  }

  if (version < 2)
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments_.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments_[i].segment.idx;
      inStream >> matchingSegments_[i].segment.cell;
      matchingSegments_[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments_.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments_[i].segment.idx;
      inStream >> matchingSegments_[i].segment.cell;
      inStream >> matchingSegments_[i].overlap;
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

  inStream >> marker;
  NTA_CHECK(marker == "~TemporalMemory");

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
    << "predictedSegmentDecrement = " << getPredictedSegmentDecrement() << std::endl;
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
