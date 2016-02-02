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
#include <boost/tuple/tuple.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

#define EPSILON 0.0000001

TemporalMemory::TemporalMemory()
{
  version_ = 1;
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
  predictiveCells.clear();
  matchingSegments.clear();
  matchingCells.clear();
}

void TemporalMemory::compute(UInt activeColumnsSize, UInt activeColumns[], bool learn)
{
  set<Cell> prevPredictiveCells(predictiveCells.begin(), predictiveCells.end());
  vector<Segment> prevActiveSegments(activeSegments);
  set<Cell> prevActiveCells(activeCells);
  set<Cell> prevWinnerCells(winnerCells);
  vector<Segment> prevMatchingSegments(matchingSegments);
  set<Cell> prevMatchingCells(matchingCells.begin(), matchingCells.end());

  set<UInt> _activeColumns;
  set<UInt> _predictedColumns;
  set<Cell> _predictedInactiveCells;
  set<Cell> _activeCells;
  set<Cell> _winnerCells;
  vector<Segment> _learningSegments;

  for (UInt i = 0; i < activeColumnsSize; i++)
  {
    _activeColumns.insert(activeColumns[i]);
  }

  activeCells.clear();
  winnerCells.clear();

  tie(
    _activeCells,
    _winnerCells,
    _predictedColumns,
    _predictedInactiveCells) =
    activateCorrectlyPredictiveCells(
      prevPredictiveCells,
      prevMatchingCells,
      _activeColumns);

  for (Cell cell : _activeCells)
    activeCells.insert(cell);
  for (Cell cell : _winnerCells)
    winnerCells.insert(cell);

  tie(
    _activeCells,
    _winnerCells,
    _learningSegments) = burstColumns(
      _activeColumns,
      _predictedColumns,
      prevActiveCells,
      prevWinnerCells,
      connections);

  for (Cell cell : _activeCells)
    activeCells.insert(cell);
  for (Cell cell : _winnerCells)
    winnerCells.insert(cell);

  if (learn)
  {
    learnOnSegments(
      prevActiveSegments,
      _learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      connections,
      _predictedInactiveCells,
      prevMatchingSegments);
  }

  vector<Segment> _activeSegments;
  set<Cell> _predictiveCells;
  vector<Segment> _matchingSegments;
  set<Cell> _matchingCells;

  tie(_activeSegments, _predictiveCells,
    _matchingSegments, _matchingCells) =
    computePredictiveCells(activeCells, connections);

  activeSegments = _activeSegments;
  predictiveCells.clear();
  for (Cell c : _predictiveCells)
    predictiveCells.push_back(c);
  matchingSegments = _matchingSegments;
  matchingCells.clear();
  for (Cell c : _matchingCells)
    matchingCells.push_back(c);
}

void TemporalMemory::reset(void)
{
  activeCells.clear();
  predictiveCells.clear();
  activeSegments.clear();
  winnerCells.clear();
}

// ==============================
//  Phases
// ==============================

tuple<set<Cell>, set<Cell>, set<UInt>, set<Cell>>
TemporalMemory::activateCorrectlyPredictiveCells(
  set<Cell>& prevPredictiveCells,
  set<Cell>& prevMatchingCells,
  set<UInt>& activeColumns)
{
  set<Cell> _activeCells;
  set<Cell> _winnerCells;
  set<UInt> _predictedColumns;
  set<Cell> _predictedInactiveCells;

  for (Cell cell : prevPredictiveCells)
  {
    UInt column = columnForCell(cell);

    if (find(activeColumns.begin(), activeColumns.end(), column)
      != activeColumns.end())
    {
      _activeCells.insert(cell);
      _winnerCells.insert(cell);
      _predictedColumns.insert(column);
    }
  }

  if (predictedSegmentDecrement_ > 0.0)
  {
    for (Cell cell : prevMatchingCells)
    {
      UInt column = columnForCell(cell);

      if (find(activeColumns.begin(), activeColumns.end(), column)
        == activeColumns.end())
      {
        _predictedInactiveCells.insert(cell);
      }
    }
  }

  return make_tuple(_activeCells, _winnerCells,
    _predictedColumns, _predictedInactiveCells);
}

tuple<set<Cell>, set<Cell>, vector<Segment>> TemporalMemory::burstColumns(
  set<UInt>& activeColumns,
  set<UInt>& predictedColumns,
  set<Cell>& prevActiveCells,
  set<Cell>& prevWinnerCells,
  Connections& _connections)
{
  set<Cell> _activeCells;
  set<Cell> _winnerCells;
  vector<Segment> _learningSegments;

  vector<UInt> _unpredictedColumns(activeColumns.begin(), activeColumns.end());

  if (predictedColumns.size() > 0)
  {
    // Resize to the worst case usage
    _unpredictedColumns.resize(activeColumns.size() + predictedColumns.size());

    // Remove the predicted columns from the 
    // currently active columns
    vector<UInt>::iterator it = set_difference(
      activeColumns.begin(), activeColumns.end(),
      predictedColumns.begin(), predictedColumns.end(),
      _unpredictedColumns.begin());

    // Trim remainer of set
    _unpredictedColumns.resize(it - _unpredictedColumns.begin());
  }

  // Sort unpredictedActiveColumns before iterating for python compatibility
  sort(_unpredictedColumns.begin(), _unpredictedColumns.end());

  for (Int column : _unpredictedColumns)
  {
    Segment bestSegment;
    Cell bestCell;
    bool foundCell = false;
    bool foundSegment = false;

    vector<Cell> cells = cellsForColumnCell(column);

    for (auto cell : cells)
      _activeCells.insert(cell);

    tie(foundCell, bestCell, foundSegment, bestSegment) =
      bestMatchingCell(cells, prevActiveCells, _connections);

    if (foundCell)
    {
      _winnerCells.insert(bestCell);
    }

    if (!foundSegment && prevWinnerCells.size())
    {
      bestSegment = _connections.createSegment(bestCell);
      foundSegment = true;
    }

    if (foundSegment)
    {
      _learningSegments.push_back(bestSegment);
    }
  }

  return make_tuple(_activeCells, _winnerCells, _learningSegments);
}

bool sortSegmentsByCells(Segment i, Segment j) {
  if (i.cell.idx == j.cell.idx) {
    // secondary sort on segment idx
    return i.idx < j.idx;
  }
  else {
    // primary sort on cell idx
    return i.cell.idx < j.cell.idx;
  }
}

void TemporalMemory::learnOnSegments(
  vector<Segment>& prevActiveSegments,
  vector<Segment>& learningSegments,
  set<Cell>& prevActiveCells,
  set<Cell>& winnerCells,
  set<Cell>& prevWinnerCells,
  Connections& _connections,
  set<Cell>& predictedInactiveCells,
  vector<Segment>& prevMatchingSegments)
{
  vector<Segment> _allSegments;

  for (auto segment : prevActiveSegments)
    _allSegments.push_back(segment);
  for (auto segment : learningSegments)
    _allSegments.push_back(segment);

  // Sort segments before iterating for python compatibility
  sort(_allSegments.begin(), _allSegments.end(), sortSegmentsByCells);

  for (Segment segment : _allSegments)
  {
    bool isLearningSegment = (find(
      learningSegments.begin(), learningSegments.end(),
      segment) != learningSegments.end());

    bool isFromWinnerCell = (find(
      winnerCells.begin(), winnerCells.end(),
      segment.cell) != winnerCells.end());

    vector<Synapse> activeSynapses(activeSynapsesForSegment(
      segment, prevActiveCells, _connections));

    if (isLearningSegment || isFromWinnerCell)
    {
      adaptSegment(segment, activeSynapses, _connections,
        permanenceIncrement_, permanenceDecrement_);
    }

    if (isLearningSegment)
    {
      Int n = maxNewSynapseCount_ - Int(activeSynapses.size());

      set<Cell> learningCells = pickCellsToLearnOn(
        n, segment,
        prevWinnerCells, _connections);

      for (Cell presynapticCell : learningCells)
      {
        _connections.createSynapse(
          segment, presynapticCell, initialPermanence_);
      }
    }
  }

  if (predictedSegmentDecrement_ > 0.0)
  {
    for (Segment segment : prevMatchingSegments)
    {
      bool isPredictedInactiveCell = (find(
        predictedInactiveCells.begin(), predictedInactiveCells.end(),
        segment.cell) != predictedInactiveCells.end());

      vector<Synapse> activeSynapses(activeSynapsesForSegment(
        segment, prevActiveCells, _connections));

      if (isPredictedInactiveCell)
      {
        adaptSegment(segment, activeSynapses, _connections,
          -predictedSegmentDecrement_, 0.0);
      }
    }
  }
}

tuple<vector<Segment>, set<Cell>, vector<Segment>, set<Cell>>
TemporalMemory::computePredictiveCells(
  set<Cell>& _activeCells, Connections& _connections)
{
  map<Segment, int> numActiveConnectedSynapsesForSegment;
  map<Segment, int> numActiveSynapsesForSegment;
  vector<Cell> activeCells(_activeCells.begin(), _activeCells.end());

  Activity activity = _connections.computeActivity(activeCells, connectedPermanence_, activationThreshold_);

  vector<Segment> _activeSegments = _connections.activeSegments(activity);
  vector<Cell> predictiveCellsVec = _connections.activeCells(activity);
  set<Cell> _predictiveCells(predictiveCellsVec.begin(), predictiveCellsVec.end());

  Activity matchingActivity = _connections.computeActivity(activeCells, 0.0, minThreshold_, false);

  vector<Segment> _matchingSegments = _connections.activeSegments(matchingActivity);
  vector<Cell> matchingCellsVec = _connections.activeCells(matchingActivity);
  set<Cell> _matchingCells(matchingCellsVec.begin(), matchingCellsVec.end());

  return make_tuple(_activeSegments, _predictiveCells, 
                    _matchingSegments, _matchingCells);
}

// ==============================
//  Helper functions
// ==============================

tuple<bool, Cell, bool, Segment>
TemporalMemory::bestMatchingCell(
  vector<Cell>& cells,
  set<Cell>& activeCells,
  Connections& _connections)
{
  Int maxSynapses = 0;
  Int numActiveSynapses;
  Cell bestCell;
  Segment bestSegment;
  bool foundCell = false;
  bool foundSegment = false;

  Segment segment;
  for (Cell cell : cells)
  {
    bool found;
    tie(found, segment, numActiveSynapses) =
      bestMatchingSegment(cell, activeCells, _connections);

    if (found && numActiveSynapses > maxSynapses)
    {
      maxSynapses = numActiveSynapses;
      foundCell = true;
      bestCell = cell;
      foundSegment = true;
      bestSegment = segment;
    }
  }

  if (!foundCell)
  {
    bestCell = leastUsedCell(cells, _connections);
    foundCell = true;
    foundSegment = false;
  }

  return make_tuple(foundCell, bestCell, foundSegment, bestSegment);
}

tuple<bool, Segment, Int>
TemporalMemory::bestMatchingSegment(
  Cell& cell,
  set<Cell>& activeCells,
  Connections& _connections)
{
  Int maxSynapses = minThreshold_;
  Int bestNumActiveSynapses = 0;
  Segment bestSegment;
  bool found = false;

  for (Segment segment : _connections.segmentsForCell(cell))
  {
    Int numActiveSynapses = 0;

    for (auto synapse : _connections.synapsesForSegment(segment))
    {
      SynapseData synapseData = _connections.dataForSynapse(synapse);

      if (find(activeCells.begin(), activeCells.end(),
        synapseData.presynapticCell) != activeCells.end())
      {
        numActiveSynapses += 1;
      }
    }

    if (numActiveSynapses >= maxSynapses)
    {
      maxSynapses = numActiveSynapses;
      bestSegment = segment;
      bestNumActiveSynapses = numActiveSynapses;
      found = true;
    }
  }

  return make_tuple(found, bestSegment, bestNumActiveSynapses);
}

Cell TemporalMemory::leastUsedCell(
  vector<Cell>& cells,
  Connections& _connections)
{
  vector<Cell> leastUsedCells;
  Int minNumSegments = INT_MAX;

  for (Cell cell : cells)
  {
    Int numSegments = (Int)_connections.segmentsForCell(cell).size();

    if (numSegments < minNumSegments)
    {
      minNumSegments = numSegments;
      leastUsedCells.clear();
    }

    if (numSegments == minNumSegments)
      leastUsedCells.push_back(cell);
  }

  Int i = _rng.getUInt32((UInt32)leastUsedCells.size());
  return leastUsedCells[i];
}

vector<Synapse> TemporalMemory::activeSynapsesForSegment(
  Segment& segment,
  set<Cell>& activeCells,
  Connections& _connections)
{
  vector<Synapse> synapses;

  for (Synapse synapse : _connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = _connections.dataForSynapse(synapse);

    if (find(activeCells.begin(), activeCells.end(),
      synapseData.presynapticCell) != activeCells.end())
    {
      synapses.push_back(synapse);
    }
  }
  return synapses;
}

void TemporalMemory::adaptSegment(
  Segment& segment,
  vector<Synapse>& activeSynapses,
  Connections& _connections,
  Permanence _permanenceIncrement,
  Permanence _permanenceDecrement)
{
  vector<Synapse> synapses = _connections.synapsesForSegment(segment);
  for (Synapse synapse : synapses)
  {
    SynapseData synapseData = _connections.dataForSynapse(synapse);
    Permanence permanence = synapseData.permanence;

    if (find(activeSynapses.begin(), activeSynapses.end(),
      synapse) != activeSynapses.end())
      permanence += _permanenceIncrement;
    else
      permanence -= _permanenceDecrement;

    // Keep permanence within min / max bounds
    if (permanence > 1.0)
      permanence = 1.0;
    if (permanence < 0.0)
      permanence = 0.0;

    if (permanence < EPSILON)
      _connections.destroySynapse(synapse);
    else
      _connections.updateSynapsePermanence(synapse, permanence);
  }
}

set<Cell> TemporalMemory::pickCellsToLearnOn(
  Int iN,
  Segment& segment,
  set<Cell>& _winnerCells,
  Connections& _connections)
{
  vector<Cell> candidates(_winnerCells.begin(), _winnerCells.end());

  // Remove cells that are already synapsed on by this segment
  for (auto synapse : _connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = _connections.dataForSynapse(synapse);
    Cell presynapticCell = synapseData.presynapticCell;

    if (find(candidates.begin(), candidates.end(),
      presynapticCell) != candidates.end())
    {
      candidates.erase(find(candidates.begin(), candidates.end(), presynapticCell));
    }
  }

  // Pick n cells randomly
  Int n = min(iN, (Int)candidates.size());
  sort(candidates.begin(), candidates.end());

  set<Cell> cells;
  for (int c = 0; c < n; c++)
  {
    Int i = _rng.getUInt32((UInt32)candidates.size());
    cells.insert(candidates[i]);
    candidates.erase(find(candidates.begin(), candidates.end(), candidates[i]));
  }

  return cells;
}

Int TemporalMemory::columnForCell(Cell& cell)
{
  _validateCell(cell);

  return cell.idx / cellsPerColumn_;
}

vector<Cell> TemporalMemory::cellsForColumnCell(Int column)
{
  _validateColumn(column);

  Int start = cellsPerColumn_ * column;
  Int end = start + cellsPerColumn_;

  vector<Cell> cellsInColumn;
  for (Int i = start; i < end; i++)
  {
    cellsInColumn.push_back(Cell(i));
  }

  return cellsInColumn;
}

vector<CellIdx> TemporalMemory::cellsForColumn(Int column)
{
  return _cellsToIndices(cellsForColumnCell(column));
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
  return _cellsToIndices(predictiveCells);
}

vector<CellIdx> TemporalMemory::getWinnerCells() const
{
  return _cellsToIndices(winnerCells);
}

vector<CellIdx> TemporalMemory::getMatchingCells() const
{
  return _cellsToIndices(matchingCells);
}

UInt TemporalMemory::numberOfColumns() const
{
  return numColumns_;
}

map<Int, set<Cell>> TemporalMemory::mapCellsToColumns(set<Cell>& cells)
{
  map<Int, set<Cell>> cellsForColumns;

  for (Cell cell : cells)
  {
    Int column = columnForCell(cell);
    cellsForColumns[column].insert(cell);
  }

  return cellsForColumns;
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

bool TemporalMemory::_validateColumn(UInt column)
{
  if (column < numberOfColumns())
    return true;

  NTA_THROW << "Invalid column " << column;
  return false;
}

bool TemporalMemory::_validateCell(Cell& cell)
{
  if (cell.idx < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell.idx;
  return false;
}

bool TemporalMemory::_validateSegment(Segment& segment)
{
  if (activeSegments.size() == 0 && segment.idx <= MAX_SEGMENTS_PER_CELL &&
    _validateCell(segment.cell))
    return true;

  if (find(activeSegments.begin(), activeSegments.end(), segment)
    != activeSegments.end())
    return true;

  NTA_THROW << "Invalid segment" << segment.idx;
  return false;
}

bool TemporalMemory::_validatePermanence(Real permanence)
{
  if (permanence < 0.0 || permanence > 1.0)
  {
    NTA_THROW << "Invalid permanence " << permanence;
    return false;
  }
  return true;
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

  outStream << predictiveCells.size() << " ";
  for (Cell elem : predictiveCells) {
    outStream << elem.idx << " ";
  }
  outStream << endl;

  outStream << winnerCells.size() << " ";
  for (Cell elem : winnerCells) {
    outStream << elem.idx << " ";
  }
  outStream << endl;

  outStream << activeSegments.size() << " ";
  for (Segment elem : activeSegments) {
    outStream << elem.idx << " ";
    outStream << elem.cell.idx << " ";
  }
  outStream << endl;

  outStream << matchingSegments.size() << " ";
  for (Segment elem : matchingSegments) {
    outStream << elem.idx << " ";
    outStream << elem.cell.idx << " ";
  }
  outStream << endl;

  outStream << matchingCells.size() << " ";
  for (Cell elem : matchingCells) {
    outStream << elem.idx << " ";
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

  auto _predictiveCells = proto.initPredictiveCells(predictiveCells.size());
  i = 0;
  for (Cell c : predictiveCells)
  {
    _predictiveCells.set(i++, c.idx);
  }

  auto _activeSegments = proto.initActiveSegments(activeSegments.size());
  for (UInt i = 0; i < activeSegments.size(); ++i)
  {
    _activeSegments.set(i, activeSegments[i].cell.idx);
  }

  auto _winnerCells = proto.initWinnerCells(winnerCells.size());
  i = 0;
  for (Cell c : winnerCells)
  {
    _winnerCells.set(i++, c.idx);
  }

  auto _matchingSegments = proto.initMatchingSegments(matchingSegments.size());
  for (UInt i = 0; i < matchingSegments.size(); ++i)
  {
    _matchingSegments.set(i, matchingSegments[i].cell.idx);
  }

  auto _matchingCells = proto.initMatchingCells(matchingCells.size());
  i = 0;
  for (Cell c : matchingCells)
  {
    _matchingCells.set(i++, c.idx);
  }
}

// Implementation note: this method sets up the instance using data from
// proto. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void TemporalMemory::read(TemporalMemoryProto::Reader& proto)
{
  UInt index;

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
    activeCells.insert(Cell(value));
  }

  predictiveCells.clear();
  for (auto value : proto.getPredictiveCells())
  {
    predictiveCells.push_back(Cell(value));
  }

  activeSegments.clear();
  index = 0;
  for (auto value : proto.getActiveSegments())
  {
    activeSegments.push_back(Segment(index++, value));
  }

  winnerCells.clear();
  for (auto value : proto.getWinnerCells())
  {
    winnerCells.insert(Cell(value));
  }

  matchingSegments.clear();
  index = 0;
  for (auto value : proto.getMatchingSegments())
  {
    matchingSegments.push_back(Segment(index++, value));
  }

  matchingCells.clear();
  for (auto value : proto.getMatchingCells())
  {
    matchingCells.push_back(Cell(value));
  }
}

void TemporalMemory::load(istream& inStream)
{
  // Current version
  version_ = 1;

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
  for (UInt i = 0; i < numColumnDimensions; i++) {
    inStream >> columnDimensions_[i];
  }

  CellIdx cellIndex;

  UInt numActiveCells;
  inStream >> numActiveCells;
  for (UInt i = 0; i < numActiveCells; i++) {
    inStream >> cellIndex;
    activeCells.insert(Cell(cellIndex));
  }

  UInt numPredictiveCells;
  inStream >> numPredictiveCells;
  for (UInt i = 0; i < numPredictiveCells; i++) {
    inStream >> cellIndex;
    predictiveCells.push_back(Cell(cellIndex));
  }

  UInt numActiveSegments;
  inStream >> numActiveSegments;
  activeSegments.resize(numActiveSegments);
  for (UInt i = 0; i < numActiveSegments; i++) {
    inStream >> activeSegments[i].idx;
    inStream >> activeSegments[i].cell.idx;
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  for (UInt i = 0; i < numWinnerCells; i++) {
    inStream >> cellIndex;
    winnerCells.insert(Cell(cellIndex));
  }

  UInt numMatchingSegments;
  inStream >> numMatchingSegments;
  matchingSegments.resize(numMatchingSegments);
  for (UInt i = 0; i < numMatchingSegments; i++) {
    inStream >> matchingSegments[i].idx;
    inStream >> matchingSegments[i].cell.idx;
  }

  UInt numMatchingCells;
  inStream >> numMatchingCells;
  for (UInt i = 0; i < numMatchingCells; i++) {
    inStream >> cellIndex;
    matchingCells.push_back(Cell(cellIndex));
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
