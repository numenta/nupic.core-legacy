/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

#define EPSILON 0.000001

TemporalMemory::TemporalMemory()
{
  version_ = 2;
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
    NTA_THROW << "Number of column dimensions must be greater than 0";

  if (cellsPerColumn <= 0)
    NTA_THROW << "Number of cells per column must be greater than 0";

  NTA_CHECK(initialPermanence >= 0.0 && initialPermanence <= 1.0);
  NTA_CHECK(connectedPermanence >= 0.0 && connectedPermanence <= 1.0);
  NTA_CHECK(permanenceIncrement >= 0.0 && permanenceIncrement <= 1.0);
  NTA_CHECK(permanenceDecrement >= 0.0 && permanenceDecrement <= 1.0);

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

  activeCells.clear();
  activeSegments.clear();
  winnerCells.clear();
  matchingSegments.clear();
}

/**
 * Walk the lists of active columns, active segments, and matching segments,
 * finding each case of "predicted column active", "unpredicted column active",
 * and "predicted column inactive". Each list is traversed exactly once.
 *
 * @param activeColumns
 * A sorted list of active columns.
 *
 * @param cellsPerColumn
 * The number of cells per column.
 *
 * @param activeSegments
 * A sorted list of segment-overlap pairs. Sorted by segment.
 *
 * @param matchingSegments
 * A sorted list of segment-overlap pairs. Sorted by segment.
 *
 * @param onPredictedColumnActive
 * A callback that receives 5 parameters:
 *
 * - the column
 * - begin and end iterators for active segments in this column
 * - begin and end iterators for matching segments in this column
 *
 * It gets called for every active column that has at least one active segment.
 *
 * @param onUnpredictedColumnActive
 * A callback that receives 3 parameters:
 *
 * - the column
 * - begin and end iterators for active segments in this column
 * - begin and end iterators for matching segments in this column
 *
 * It gets called for every active column that has zero active segments.
 *
 * @param onPredictedColumnsInactive
 * A callback that receives 4 parameters:
 *
 * - begin and end iterators for active segments in this column
 * - begin and end iterators for matching segments in this column
 *
 * It gets called for every range of active / matching segments that aren't
 * within active columns. The ranges may contain multiple columns. The ranges
 * may be empty.
 */
template <typename FuncA, typename FuncB, typename FuncC>
static void columnSegmentWalk(
  const vector<UInt>& activeColumns,
  UInt cellsPerColumn,
  const vector<SegmentOverlap>& activeSegments,
  const vector<SegmentOverlap>& matchingSegments,
  FuncA onPredictedColumnActive,
  FuncB onUnpredictedColumnActive,
  FuncC onPredictedColumnsInactive)
{
  auto active = activeSegments.begin();
  auto matching = matchingSegments.begin();

  for (UInt activeColumn : activeColumns)
  {
    const Cell firstInColumn = Cell(activeColumn*cellsPerColumn);
    const Cell lastInColumn = Cell((activeColumn+1)*cellsPerColumn - 1);

    // Report segments before this active column.
    const auto wrongActiveBegin = active;
    const auto wrongMatchingBegin = matching;

    while (active != activeSegments.end() &&
           active->segment.cell < firstInColumn)
    {
      active++;
    }

    while (matching != matchingSegments.end() &&
           matching->segment.cell < firstInColumn)
    {
      matching++;
    }

    onPredictedColumnsInactive(wrongActiveBegin, active,
                               wrongMatchingBegin, matching);

    if (active != activeSegments.end() &&
        active->segment.cell <= lastInColumn)
    {
      // Column has one or more predicted cell.
      const auto correctActiveBegin = active;
      const auto correctMatchingBegin = matching;

      while (active != activeSegments.end() &&
             active->segment.cell <= lastInColumn)
      {
        active++;
      }

      while (matching != matchingSegments.end() &&
             matching->segment.cell <= lastInColumn)
      {
        matching++;
      }

      onPredictedColumnActive(activeColumn,
                              correctActiveBegin, active,
                              correctMatchingBegin, matching);
    }
    else
    {
      // Column has no predicted cells.
      const auto correctMatchingBegin = matching;

      while (matching != matchingSegments.end() &&
             matching->segment.cell <= lastInColumn)
      {
        matching++;
      }

      onUnpredictedColumnActive(activeColumn, correctMatchingBegin, matching);
    }
  }

  // Report remaining segments after the last active column.
  auto wrongActiveBegin = active;
  auto wrongMatchingBegin = matching;

  while (active != activeSegments.end())
  {
    active++;
  }

  while (matching != matchingSegments.end())
  {
    matching++;
  }

  onPredictedColumnsInactive(wrongActiveBegin, active,
                             wrongMatchingBegin, matching);
}

void TemporalMemory::compute(UInt activeColumnsSize,
                             const UInt activeColumnsUnsorted[],
                             bool learn)
{
  const vector<Cell> prevActiveCells = activeCells;
  const vector<Cell> prevWinnerCells = winnerCells;

  vector<UInt> activeColumns(activeColumnsUnsorted,
                             activeColumnsUnsorted + activeColumnsSize);
  std::sort(activeColumns.begin(), activeColumns.end());

  activeCells.clear();
  winnerCells.clear();

  columnSegmentWalk(
    activeColumns, cellsPerColumn_,
    activeSegments, matchingSegments,

    /*onPredictedColumnActive*/
    [&](UInt column,
        vector<SegmentOverlap>::const_iterator activeSegmentsBegin,
        vector<SegmentOverlap>::const_iterator activeSegmentsEnd,
        vector<SegmentOverlap>::const_iterator matchingSegmentsBegin,
        vector<SegmentOverlap>::const_iterator matchingSegmentsEnd)
    {
      auto active = activeSegmentsBegin;
      while (active != activeSegmentsEnd)
      {
        Cell cell(active->segment.cell);
        activeCells.push_back(cell);
        winnerCells.push_back(cell);

        // This cell might have multiple active segments.
        do
        {
          if (learn)
          {
            adaptSegment_(active->segment, prevActiveCells,
                          permanenceIncrement_, permanenceDecrement_);
          }
          active++;
        } while (active != activeSegmentsEnd && active->segment.cell == cell);
      }
    },

    /*onUnpredictedColumnActive*/
    [&](UInt column,
        vector<SegmentOverlap>::const_iterator matchingSegmentsBegin,
        vector<SegmentOverlap>::const_iterator matchingSegmentsEnd)
    {
      const CellIdx start = column * cellsPerColumn_;
      const CellIdx end = start + cellsPerColumn_;
      for (CellIdx i = start; i < end; i++)
      {
        activeCells.push_back(Cell(i));
      }

      if (matchingSegmentsBegin != matchingSegmentsEnd)
      {
        auto bestMatch = std::max_element(
          matchingSegmentsBegin, matchingSegmentsEnd,
          [](const SegmentOverlap& a, const SegmentOverlap& b)
          {
            return a.overlap < b.overlap;
          });

        winnerCells.push_back(bestMatch->segment.cell);

        if (learn)
        {
          adaptSegment_(bestMatch->segment, prevActiveCells,
                        permanenceIncrement_, permanenceDecrement_);

          const UInt32 nGrow = maxNewSynapseCount_ - bestMatch->overlap;
          if (nGrow > 0)
          {
            growSynapses_(bestMatch->segment, prevWinnerCells, nGrow);
          }
        }
      }
      else
      {
        const Cell winnerCell = getLeastUsedCell_(column);
        winnerCells.push_back(winnerCell);

        if (learn)
        {
          const UInt32 nGrow = min(maxNewSynapseCount_,
                                   (UInt32)prevWinnerCells.size());
          if (nGrow > 0)
          {
            const Segment segment = connections.createSegment(winnerCell);
            growSynapses_(segment, prevWinnerCells, nGrow);
          }
        }
      }
    },

    /*onPredictedColumnsInactive*/
    [&](vector<SegmentOverlap>::const_iterator activeSegmentsBegin,
        vector<SegmentOverlap>::const_iterator activeSegmentsEnd,
        vector<SegmentOverlap>::const_iterator matchingSegmentsBegin,
        vector<SegmentOverlap>::const_iterator matchingSegmentsEnd)
    {
      if (learn && predictedSegmentDecrement_ > 0.0)
      {
        for (auto matching = matchingSegmentsBegin;
             matching != matchingSegmentsEnd;
             matching++)
        {
          adaptSegment_(matching->segment, prevActiveCells,
                        -predictedSegmentDecrement_, 0.0);
        }
      }
    });

  activeSegments.clear();
  matchingSegments.clear();
  connections.computeActivity(activeCells,
                              connectedPermanence_, activationThreshold_,
                              0.0, minThreshold_,
                              activeSegments, matchingSegments);
}

void TemporalMemory::reset(void)
{
  activeCells.clear();
  activeSegments.clear();
  matchingSegments.clear();
  winnerCells.clear();
}

// ==============================
//  Helper functions
// ==============================

Cell TemporalMemory::getLeastUsedCell_(UInt column)
{
  vector<Cell> leastUsedCells;
  UInt32 minNumSegments = UINT_MAX;
  const CellIdx start = column * cellsPerColumn_;
  const CellIdx end = start + cellsPerColumn_;
  for (CellIdx i = start; i < end; i++)
  {
    Cell cell(i);
    UInt32 numSegments = connections.segmentsForCell(cell).size();

    if (numSegments < minNumSegments)
    {
      minNumSegments = numSegments;
      leastUsedCells.clear();
    }

    if (numSegments == minNumSegments)
    {
      leastUsedCells.push_back(cell);
    }
  }

  return leastUsedCells[_rng.getUInt32(leastUsedCells.size())];
}

void TemporalMemory::adaptSegment_(
  Segment segment,
  const vector<Cell>& prevActiveCells,
  Permanence _permanenceIncrement,
  Permanence _permanenceDecrement)
{
  vector<Synapse> synapses =
    connections.synapsesForSegment(segment);

  for (Synapse synapse : synapses)
  {
    const SynapseData synapseData = connections.dataForSynapse(synapse);
    const bool isActive =
      std::binary_search(prevActiveCells.begin(), prevActiveCells.end(),
                         synapseData.presynapticCell);
    Permanence permanence = synapseData.permanence;

    if (isActive)
    {
      permanence += _permanenceIncrement;
    }
    else
    {
      permanence -= _permanenceDecrement;
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
}

void TemporalMemory::growSynapses_(
  Segment segment,
  const vector<Cell>& prevWinnerCells,
  UInt32 n)
{
  vector<Cell> candidates(prevWinnerCells.begin(), prevWinnerCells.end());

  // Instead of erasing candidates, swap them to the end, and remember where the
  // "eligible" candidates end.
  auto eligibleEnd = candidates.end();

  // Remove cells that are already synapsed on by this segment
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    Cell presynapticCell = connections.dataForSynapse(synapse).presynapticCell;
    auto ineligible = find(candidates.begin(), eligibleEnd, presynapticCell);
    if (ineligible != eligibleEnd)
    {
      eligibleEnd--;
      std::iter_swap(ineligible, eligibleEnd);
    }
  }

  // Pick n cells randomly.
  for (UInt32 c = 0; c < n && eligibleEnd != candidates.begin(); c++)
  {
    size_t i = _rng.getUInt32(std::distance(candidates.begin(), eligibleEnd));;
    connections.createSynapse(segment, candidates[i], initialPermanence_);
    eligibleEnd--;
    std::swap(candidates[i], *eligibleEnd);
  }
}

Int TemporalMemory::columnForCell(Cell& cell)
{
  _validateCell(cell);

  return cell.idx / cellsPerColumn_;
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
  return _cellsToIndices(activeCells);
}

vector<CellIdx> TemporalMemory::getPredictiveCells() const
{
  vector<CellIdx> predictiveCells;

  for (auto segOverlap = activeSegments.begin();
       segOverlap != activeSegments.end();
       segOverlap++)
  {
    if (segOverlap == activeSegments.begin() ||
        segOverlap->segment.cell.idx != predictiveCells.back())
    {
      predictiveCells.push_back(segOverlap->segment.cell.idx);
    }
  }

  return predictiveCells;
}

vector<CellIdx> TemporalMemory::getWinnerCells() const
{
  return _cellsToIndices(winnerCells);
}

vector<CellIdx> TemporalMemory::getMatchingCells() const
{
  vector<CellIdx> matchingCells;

  for (auto segOverlap = matchingSegments.begin();
       segOverlap != matchingSegments.end();
       segOverlap++)
  {
    if (segOverlap == matchingSegments.begin() ||
        segOverlap->segment.cell.idx != matchingCells.back())
    {
      matchingCells.push_back(segOverlap->segment.cell.idx);
    }
  }

  return matchingCells;
}

UInt TemporalMemory::numberOfColumns() const
{
  return numColumns_;
}

template <typename Iterable>
vector<CellIdx> TemporalMemory::_cellsToIndices(const Iterable &cellSet) const
{
  vector<CellIdx> idxVector;
  idxVector.reserve(cellSet.size());
  for (Cell cell : cellSet)
  {
    idxVector.push_back(cell.idx);
  }
  return idxVector;
}

bool TemporalMemory::_validateCell(Cell& cell)
{
  if (cell.idx < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell.idx;
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

/**
* Create a RNG with given seed
*/
void TemporalMemory::seed_(UInt64 seed)
{
  _rng = Random(seed);
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
  outStream << version_ << endl;

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

  outStream << _rng << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto & elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << activeCells.size() << " ";
  for (Cell elem : activeCells) {
    outStream << elem.idx << " ";
  }
  outStream << endl;

  outStream << winnerCells.size() << " ";
  for (Cell elem : winnerCells) {
    outStream << elem.idx << " ";
  }
  outStream << endl;

  outStream << activeSegments.size() << " ";
  for (SegmentOverlap elem : activeSegments) {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell.idx << " ";
    outStream << elem.overlap << " ";
  }
  outStream << endl;

  outStream << matchingSegments.size() << " ";
  for (SegmentOverlap elem : matchingSegments) {
    outStream << elem.segment.idx << " ";
    outStream << elem.segment.cell.idx << " ";
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
  _rng.write(random);

  auto _activeCells = proto.initActiveCells(activeCells.size());
  UInt i = 0;
  for (Cell c : activeCells)
  {
    _activeCells.set(i++, c.idx);
  }

  auto _activeSegmentOverlaps =
    proto.initActiveSegmentOverlaps(activeSegments.size());
  for (UInt i = 0; i < activeSegments.size(); ++i)
  {
    Segment segment = activeSegments[i].segment;
    _activeSegmentOverlaps[i].setCell(segment.cell.idx);
    _activeSegmentOverlaps[i].setSegment(segment.idx);
    _activeSegmentOverlaps[i].setOverlap(activeSegments[i].overlap);
  }

  auto _winnerCells = proto.initWinnerCells(winnerCells.size());
  i = 0;
  for (Cell c : winnerCells)
  {
    _winnerCells.set(i++, c.idx);
  }

  auto _matchingSegmentOverlaps =
    proto.initMatchingSegmentOverlaps(matchingSegments.size());
  for (UInt i = 0; i < matchingSegments.size(); ++i)
  {
    Segment segment = matchingSegments[i].segment;
    _matchingSegmentOverlaps[i].setCell(segment.cell.idx);
    _matchingSegmentOverlaps[i].setSegment(segment.idx);
    _matchingSegmentOverlaps[i].setOverlap(matchingSegments[i].overlap);
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
  _rng.read(random);

  activeCells.clear();
  for (auto value : proto.getActiveCells())
  {
    activeCells.push_back(Cell(value));
  }

  if (proto.getActiveSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "TemporalMemory::read :: Obsolete field 'activeSegments' isn't usable. "
             << "TemporalMemory results will be goofy for one timestep.";
  }

  activeSegments.clear();
  for (auto value : proto.getActiveSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(),
                       {(CellIdx)value.getCell()}};
    activeSegments.push_back({segment, value.getOverlap()});
  }

  winnerCells.clear();
  for (auto value : proto.getWinnerCells())
  {
    winnerCells.push_back(Cell(value));
  }

  if (proto.getMatchingSegments().size())
  {
    // There's no way to convert a UInt32 to a segment. It never worked.
    NTA_WARN << "TemporalMemory::read :: Obsolete field 'matchingSegments' isn't usable. "
             << "TemporalMemory results will be goofy for one timestep.";
  }

  matchingSegments.clear();
  for (auto value : proto.getMatchingSegmentOverlaps())
  {
    Segment segment = {(SegmentIdx)value.getSegment(),
                       {(CellIdx)value.getCell()}};
    matchingSegments.push_back({segment, value.getOverlap()});
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
  NTA_CHECK(version <= version_);

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

  inStream >> _rng;

  // Retrieve vectors.
  UInt numColumnDimensions;
  inStream >> numColumnDimensions;
  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++)
  {
    inStream >> columnDimensions_[i];
  }

  CellIdx cellIndex;

  UInt numActiveCells;
  inStream >> numActiveCells;
  for (UInt i = 0; i < numActiveCells; i++)
  {
    inStream >> cellIndex;
    activeCells.push_back(Cell(cellIndex));
  }

  if (version < 2)
  {
    UInt numPredictiveCells;
    inStream >> numPredictiveCells;
    for (UInt i = 0; i < numPredictiveCells; i++)
    {
      inStream >> cellIndex; // Ignore
    }
  }

  if (version < 2)
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments[i].segment.idx;
      inStream >> activeSegments[i].segment.cell.idx;
      activeSegments[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numActiveSegments;
    inStream >> numActiveSegments;
    activeSegments.resize(numActiveSegments);
    for (UInt i = 0; i < numActiveSegments; i++)
    {
      inStream >> activeSegments[i].segment.idx;
      inStream >> activeSegments[i].segment.cell.idx;
      inStream >> activeSegments[i].overlap;
    }
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++)
  {
    inStream >> cellIndex;
    winnerCells.push_back(Cell(cellIndex));
  }

  if (version < 2)
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments[i].segment.idx;
      inStream >> matchingSegments[i].segment.cell.idx;
      matchingSegments[i].overlap = 0; // Unknown
    }
  }
  else
  {
    UInt numMatchingSegments;
    inStream >> numMatchingSegments;
    matchingSegments.resize(numMatchingSegments);
    for (UInt i = 0; i < numMatchingSegments; i++)
    {
      inStream >> matchingSegments[i].segment.idx;
      inStream >> matchingSegments[i].segment.cell.idx;
      inStream >> matchingSegments[i].overlap;
    }
  }

  if (version < 2)
  {
    UInt numMatchingCells;
    inStream >> numMatchingCells;
    for (UInt i = 0; i < numMatchingCells; i++) {
      inStream >> cellIndex; // Ignore
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
    << "version                   = " << version_ << std::endl
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
  for (UInt i = 0; i != state.size(); ++i) {
    if (i > 0 && i % 10 == 0) {
      std::cout << "\n   ";
    }
    std::cout << state[i] << " ";
  }
  std::cout << "]\n";
}

void TemporalMemory::printState(vector<Real> &state)
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
