/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
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

TemporalMemory::TemporalMemory()
{
  version_ = 1;
}

TemporalMemory::TemporalMemory(
  vector<UInt> columnDimensions,
  Int cellsPerColumn,
  Int activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  Int minThreshold,
  Int maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Int seed)
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
    seed);
}

TemporalMemory::~TemporalMemory()
{
}

/**
 Initialize the temporal memory using the given parameters.

 @param columnDimensions     Dimensions of the column space
 @param cellsPerColumn       Number of cells per column
 @param activationThreshold  If the number of active connected synapses on a segment is at least this threshold, the segment is said to be active.
 @param initialPermanence    Initial permanence of a new synapse.
 @param connectedPermanence  If the permanence value for a synapse is greater than this value, it is said to be connected.
 @param minThreshold         If the number of synapses active on a segment is at least this threshold, it is selected as the best matching cell in a bursting column.
 @param maxNewSynapseCount   The maximum number of synapses added to a segment during learning.
 @param permanenceIncrement  Amount by which permanences of synapses are incremented during learning.
 @param permanenceDecrement  Amount by which permanences of synapses are decremented during learning.
 @param seed                 Seed for the random number generator.
 */
void TemporalMemory::initialize(
  vector<UInt> columnDimensions,
  Int cellsPerColumn,
  Int activationThreshold,
  Permanence initialPermanence,
  Permanence connectedPermanence,
  Int minThreshold,
  Int maxNewSynapseCount,
  Permanence permanenceIncrement,
  Permanence permanenceDecrement,
  Int seed)
{
  // Error checking
  if (columnDimensions.size() <= 0)
    NTA_THROW << "Number of column dimensions must be greater than 0";

  if (cellsPerColumn <= 0)
    NTA_THROW << "Number of cells per column must be greater than 0";

  // TODO: Validate all parameters (and add validation tests)

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

  // Initialize member variables
  connections = Connections(numberOfCells());
  seed_((UInt64)(seed < 0 ? rand() : seed));

  activeCells.clear();
  predictiveCells.clear();
  activeSegments.clear();
  winnerCells.clear();
}

// ==============================
//  Main functions
// ==============================

/*
 * Feeds input record through TM, performing inference and learning.
 * Updates member variables with new state.
 *
 * @param activeColumns   Indices of active columns in `t`
 * @param learn           Whether or not learning is enabled
 */
void TemporalMemory::compute(UInt activeColumnsSize, UInt activeColumns[], bool learn)
{
  vector<Cell> _activeCells;
  vector<Cell> _winnerCells;
  vector<Segment> _activeSegments;
  vector<Cell> _predictiveCells;
  vector<UInt> _predictedColumns;

  tie(_activeCells, _winnerCells, _activeSegments,
    _predictiveCells, _predictedColumns)
    = computeFn(
      activeColumnsSize,
      activeColumns,
      predictiveCells,
      activeSegments,
      activeCells,
      winnerCells,
      connections,
      learn);

  activeCells = _activeCells;
  winnerCells = _winnerCells;
  activeSegments = _activeSegments;
  predictiveCells = _predictiveCells;
  predictedColumns = _predictedColumns;
}

/*
 * 'Functional' version of compute.
 * Returns new state.
 *
 * @param activeColumns         Indices of active columns in time `t`
 * @param prevPredictiveCells   Indices of predictive cells in `t-1`
 * @param prevActiveSegments    Indices of active segments in `t-1`
 * @param prevActiveCells       Indices of active cells in `t-1`
 * @param prevWinnerCells       Indices of winner cells in `t-1`
 * @param connections           Connectivity of layer
 * @param learn                 Whether or not learning is enabled
 *
 * @return (tuple)Contains:
 *  `activeCells`     (set),
 *  `winnerCells`     (set),
 *  `activeSegments`  (set),
 *  `predictiveCells` (set)
 */
tuple<vector<Cell>, vector<Cell>, vector<Segment>, vector<Cell>, vector<UInt>>
TemporalMemory::computeFn(
  UInt activeColumnsSize,
  UInt activeColumns[],
  vector<Cell>& prevPredictiveCells,
  vector<Segment>& prevActiveSegments,
  vector<Cell> prevActiveCells,
  vector<Cell> prevWinnerCells,
  Connections& _connections,
  bool learn)
{
  vector<UInt> _activeColumns;
  vector<Cell> _predictiveCells;
  vector<Cell> _activeCells;
  vector<Cell> _winnerCells;
  vector<Segment> _learningSegments;
  vector<Segment> _activeSegments;

  for (UInt i = 0; i < activeColumnsSize; i++)
  {
    _activeColumns.push_back(activeColumns[i]);
  }

  activeCells.clear();
  winnerCells.clear();

  tie(
    _activeCells,
    _winnerCells,
    predictedColumns) = activateCorrectlyPredictiveCells(
      prevPredictiveCells,
      _activeColumns);

  for (Cell cell : _activeCells)
    activeCells.push_back(cell);
  for (Cell cell : _winnerCells)
    winnerCells.push_back(cell);

  tie(
    _activeCells,
    _winnerCells,
    _learningSegments) = burstColumns(
    _activeColumns,
    predictedColumns,
    prevActiveCells,
    prevWinnerCells,
    _connections);

  for (auto cell : _activeCells)
  {
    bool found = false;
    for (Cell ac : activeCells)
    {
      if (ac.idx == cell.idx)
      {
        found = true;
        continue;
      }
    }
    if (!found)
      activeCells.push_back(cell);
  }
  for (auto cell : _winnerCells)
  {
    bool found = false;
    for (Cell wc : winnerCells)
    {
      if (wc.idx == cell.idx)
      {
        found = true;
        continue;
      }
    }
    if (!found)
      winnerCells.push_back(cell);
  }

  if (learn)
  {
    learnOnSegments(
      prevActiveSegments,
      _learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      _connections);
  }

  tie(_activeSegments, _predictiveCells) =
    computePredictiveCells(activeCells, _connections);

  return make_tuple(
    activeCells,
    winnerCells,
    activeSegments,
    predictiveCells,
    predictedColumns);
}

/*
 * Indicates the start of a new sequence.
 * Resets sequence state of the TM.
*/
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

/*
 * Phase 1 : Activate the correctly predictive cells.
 *
 * Pseudocode :
 *
 * - for each prev predictive cell
 *   - if in active column
 *     - mark it as active
 *     - mark it as winner cell
 *  	 - mark column as predicted
 *
 * @param prevPredictiveCells   Indices of predictive cells in `t-1`
 * @param activeColumns         Indices of active columns in `t`
 *
 * @return (tuple)Contains:
 *  `activeCells`      (set),
 *  `winnerCells`      (set),
 *  `predictedColumns` (set)
 */
tuple<vector<Cell>, vector<Cell>, vector<UInt>>
TemporalMemory::activateCorrectlyPredictiveCells(
  vector<Cell>& prevPredictiveCells,
  vector<UInt>& activeColumns)
{
  vector<Cell> _activeCells;
  vector<Cell> _winnerCells;
  vector<UInt> _predictedColumns;

  for (Cell cell : prevPredictiveCells)
  {
    UInt column = columnForCell(cell);

    if (find(activeColumns.begin(), activeColumns.end(), column)
      != activeColumns.end())
    {
      _activeCells.push_back(cell);
      _winnerCells.push_back(cell);

      if (find(_predictedColumns.begin(), _predictedColumns.end(), column)
        == _predictedColumns.end())
      {
        _predictedColumns.push_back(column);
      }
    }
  }

  return make_tuple(_activeCells, _winnerCells, _predictedColumns);
}

/*
 * Phase 2 : Burst unpredicted columns.
 *
 * Pseudocode :
 *
 * - for each unpredicted active column
 *  - mark all cells as active
 *  - mark the best matching cell as winner cell
 *   - (learning)
 *    - if it has no matching segment
 *     - (optimization) if there are prev winner cells
 *      - add a segment to it
 *    - mark the segment as learning
 *
 * @param activeColumns(set)       Indices of active columns in `t`
 * @param predictedColumns(set)    Indices of predicted columns in `t`
 * @param prevActiveCells(set)     Indices of active cells in `t-1`
 * @param prevWinnerCells(set)     Indices of winner cells in `t-1`
 * @param connections(Connections) Connectivity of layer
 *
 * @return (tuple)Contains:
 *  `activeCells`      (set),
 *  `winnerCells`      (set),
 *  `learningSegments` (set)
 */
tuple<vector<Cell>, vector<Cell>, vector<Segment>> TemporalMemory::burstColumns(
  vector<UInt>& activeColumns,
  vector<UInt>& predictedColumns,
  vector<Cell>& prevActiveCells,
  vector<Cell>& prevWinnerCells,
  Connections& _connections)
{
  vector<Cell> _activeCells;
  vector<Cell> _winnerCells;
  vector<Segment> _learningSegments;

  vector<UInt> _unpredictedColumns; // = activeColumns - predictedColumns
  {
    // Resize to the worst case usage
    _unpredictedColumns.resize(activeColumns.size() + predictedColumns.size());

    if (predictedColumns.size() == 0)
      _unpredictedColumns = activeColumns;
    else
      if (activeColumns.size() == 0)
      _unpredictedColumns = predictedColumns;
      else
      {
        // Remove the predicted columns from the 
        // currently active columns
        vector<UInt>::iterator it = set_difference(
          activeColumns.begin(), activeColumns.end(),
          predictedColumns.begin(), predictedColumns.end(),
          _unpredictedColumns.begin());

        // Trim remainer of set
        _unpredictedColumns.resize(it - _unpredictedColumns.begin());
      }
  }

  for (Int column : _unpredictedColumns)
  {
    Segment* bestSegment = NULL;
    Cell* bestCell = NULL;

    vector<Cell> cells = cellsForColumn(column);

    for (auto cell : cells)
      _activeCells.push_back(cell);

    tie(bestCell, bestSegment) =
      bestMatchingCell(cells, prevActiveCells, _connections);

    if (bestCell) _winnerCells.push_back(*bestCell);

    if (bestSegment == NULL && prevWinnerCells.size())
    {
      bestSegment = new Segment(_connections.createSegment(*bestCell));
    }

    if (bestSegment != NULL)
    {
      _learningSegments.push_back(*bestSegment);
    }
  }

  return make_tuple(_activeCells, _winnerCells, _learningSegments);
}

/*
 * Phase 3 : Perform learning by adapting segments.
 *
 * Pseudocode :
 *
 *	-(learning) for each prev active or learning segment
 * 	 - if learning segment or from winner cell
 *	  - strengthen active synapses
 *	  - weaken inactive synapses
 *	 - if learning segment
 *	  - add some synapses to the segment
 *	    - subsample from prev winner cells
 *
 * @param prevActiveSegments(set)   Indices of active segments in `t-1`
 * @param learningSegments(set)     Indices of learning segments in `t`
 * @param prevActiveCells(set)      Indices of active cells in `t-1`
 * @param winnerCells(set)          Indices of winner cells in `t`
 * @param prevWinnerCells(set)      Indices of winner cells in `t-1`
 * @param connections(Connections)  Connectivity of layer
 */
void TemporalMemory::learnOnSegments(
  vector<Segment>& prevActiveSegments,
  vector<Segment>& learningSegments,
  vector<Cell>& prevActiveCells,
  vector<Cell>& winnerCells,
  vector<Cell>& prevWinnerCells,
  Connections& _connections)
{
  vector<Segment> _allSegments;

  for (auto segment : prevActiveSegments)
    _allSegments.push_back(segment);
  for (auto segment : learningSegments)
    _allSegments.push_back(segment);

  for (Segment segment : _allSegments)
  {
    bool isLearningSegment = false;
    bool isFromWinnerCell = (find(winnerCells.begin(), winnerCells.end(), segment.cell) != winnerCells.end());

    for (Segment s : learningSegments)
    {
      if (s == segment)
      {
        isLearningSegment = true;
        break;
      }
    }

    vector<Synapse> activeSynapses(activeSynapsesForSegment(
      segment, prevActiveCells, _connections));

    if (isLearningSegment || isFromWinnerCell)
    {
      adaptSegment(segment, activeSynapses, _connections);
    }

    if (isLearningSegment)
    {
      Int n = maxNewSynapseCount_ - Int(activeSynapses.size());

      for (Cell presynapticCell : pickCellsToLearnOn(n, segment, prevWinnerCells, _connections))
      {
        _connections.createSynapse(segment, presynapticCell, initialPermanence_);
      }
    }
  }
}

/*
 * Phase 4 : Compute predictive cells due to lateral input on distal dendrites.
 *
 * Pseudocode :
 *	- for each distal dendrite segment with activity >= activationThreshold
 *		- mark the segment as active
 *		- mark the cell as predictive
 *
 *		Forward propagates activity from active cells to the synapses that touch
 *		them, to determine which synapses are active.
 *
 *	@param activeCells(set)         Indices of active cells in `t`
 *	@param connections(Connections) Connectivity of layer
 *
 *	@return (tuple) Contains:
 *   `activeSegments`  (set)
 *   `predictiveCells` (set)
 */
tuple<vector<Segment>, vector<Cell>> TemporalMemory::computePredictiveCells(
  vector<Cell>& _activeCells,
  Connections& _connections)
{
  map<Segment, Int> _numActiveConnectedSynapsesForSegment;

  vector<Segment> _activeSegments;
  vector<Cell> _predictiveCells;

  for (Cell cell : _activeCells)
  {
    vector<Segment> segmentsForCell = _connections.segmentsForCell(cell);

    for (Segment segment : segmentsForCell)
    {
      for (Synapse synapse : _connections.synapsesForSegment(segment))
      {
        SynapseData synapseData = _connections.dataForSynapse(synapse);
        Segment segment = synapse.segment;
        Real permanence = synapseData.permanence;

        if (permanence >= connectedPermanence_)
        {
          _numActiveConnectedSynapsesForSegment[segment] += 1;

          if (_numActiveConnectedSynapsesForSegment[segment] >= 
              activationThreshold_)
          {
            _activeSegments.push_back(segment);
            _predictiveCells.push_back(Cell(segment.cell.idx));
          }
        }
      }
    }
  }

  return make_tuple(_activeSegments, _predictiveCells);
}

// ==============================
//  Helper functions
// ==============================

/*
 * Gets the cell with the best matching segment
 * (see `TM.bestMatchingSegment`) that has the largest number of active
 * synapses of all best matching segments.
 *
 * If none were found, pick the least used cell(see `TM.leastUsedCell`).
 *
 *	@param cells        Indices of cells
 *	@param activeCells  Indices of active cells
 *	@param connections  Connectivity of layer
 *
 *	@return (tuple)Contains:
 *   `cell`        (int),
 *   `bestSegment` (int)
 */

tuple<Cell*, Segment*>
TemporalMemory::bestMatchingCell(
  vector<Cell>& cells,
  vector<Cell>& activeCells,
  Connections& _connections)
{
  Int maxSynapses = 0;
  Cell* bestCell = NULL;
  Segment* bestSegment = NULL;

  for (Cell cell : cells)
  {
    Int numActiveSynapses;
    Segment* segment = NULL;

    tie(segment, numActiveSynapses) = bestMatchingSegment(
      cell, activeCells, _connections);

    if (segment != NULL && numActiveSynapses > maxSynapses)
    {
      maxSynapses = numActiveSynapses;
      bestCell = new Cell(cell);
      bestSegment = segment;
    }
    else
    {
      if (segment)
        delete segment;
    }
  }

  if (bestCell == NULL)
    bestCell = new Cell(leastUsedCell(cells, _connections));

  return make_tuple(bestCell, bestSegment);
}

/*
 * Gets the segment on a cell with the largest number of activate synapses,
 * including all synapses with non - zero permanences.
 *
 * @param cell          Cell index
 * @param activeCells   Indices of active cells
 * @param connections   Connectivity of layer
 *
 * @return (tuple)Contains:
 *  `segment`                 (int),
 *  `connectedActiveSynapses` (set)
 */
tuple<Segment*, Int>
TemporalMemory::bestMatchingSegment(
  Cell& cell,
  vector<Cell>& activeCells,
  Connections& _connections)
{
  Int maxSynapses = minThreshold_;
  Segment* bestSegment = NULL;
  Int bestNumActiveSynapses = 0;

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
      if (bestSegment != NULL)
      {
        delete bestSegment;
      }

      maxSynapses = numActiveSynapses;
      bestSegment = new Segment(segment);
      bestNumActiveSynapses = numActiveSynapses;
    }
  }

  return make_tuple(bestSegment, bestNumActiveSynapses);
}

/*
 * Gets the cell with the smallest number of segments.
 * Break ties randomly.
 *
 * @param cells         Indices of cells
 * @param connections   Connectivity of layer
 *
 * @return (int) Cell index
 */
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

  vector<Cell> sortedCells(leastUsedCells.begin(), leastUsedCells.end());
  sort(sortedCells.begin(), sortedCells.end());
  return sortedCells[i];
}

/*
 * Returns the synapses on a segment that are active due to lateral input
 * from active cells.
 *
 * @param segment       Segment index
 * @param activeCells   Indices of active cells
 * @param connections   Connectivity of layer
 *
 * @return (set) Indices of active synapses on segment
 */
vector<Synapse> TemporalMemory::activeSynapsesForSegment(
  Segment& segment,
  vector<Cell>& activeCells,
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

/*
 * Updates synapses on segment.
 * Strengthens active synapses; weakens inactive synapses.
 *
 * @param segment        Segment index
 * @param activeSynapses Indices of active synapses
 * @param connections    Connectivity of layer
 */
void TemporalMemory::adaptSegment(
  Segment& segment,
  vector<Synapse>& activeSynapses,
  Connections& _connections)
{
  for (Synapse synapse : _connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = _connections.dataForSynapse(synapse);
    Permanence permanence = synapseData.permanence;

    if (find(activeSynapses.begin(), activeSynapses.end(),
      synapse) != activeSynapses.end())
      permanence += permanenceIncrement_;
    else
      permanence -= permanenceDecrement_;

    // Keep permanence within min / max bounds
    //permanence = max(0.0, min(1.0, permanence));
    if (permanence > 1.0)
      permanence = 1.0;
    if (permanence < 0.0)
      permanence = 0.0;

    _connections.updateSynapsePermanence(synapse, permanence);
  }
}

/*
 * Pick cells to form distal connections to.
 *
 * TODO : Respect topology and learningRadius
 *
 *	   @param n             Number of cells to pick
 *	   @param segment       Segment index
 *	   @param winnerCells   Indices of winner cells in `t`
 *	   @param connections   Connectivity of layer
 *
 *	   @return (set) Indices of cells picked
 */
vector<Cell> TemporalMemory::pickCellsToLearnOn(
  Int iN,
  Segment& segment,
  vector<Cell>& _winnerCells,
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

  vector<Cell> cells;
  for (int c = 0; c < n; c++)
  {
    Int i = _rng.getUInt32((UInt32)candidates.size());
    cells.push_back(candidates[i]);
    candidates.erase(find(candidates.begin(), candidates.end(), candidates[i]));
  }
  return cells;
}

/*
 * Returns the index of the column that a cell belongs to.
 *
 * @param cell(int) Cell index
 *
 * @return (int)Column index
 */
Int TemporalMemory::columnForCell(Cell& cell)
{
  _validateCell(cell);

  return cell.idx / cellsPerColumn_;
}

/*
 * Returns the indices of cells that belong to a column.
 *
 * @param column(int) Column index
 *
 * @return (set)Cell indices
 */

vector<Cell> TemporalMemory::cellsForColumn(Int column)
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

/*
 * Returns the number of columns in this layer.
 *
 * @return (int)Number of columns
 */
Int TemporalMemory::numberOfColumns(void)
{
  Int acc(1);

  for (Int column : columnDimensions_)
  {
    acc *= column;
  }
  return acc;
}

/*
 * Returns the number of cells in this layer.
 *
 * @return (int)Number of cells
 */
UInt TemporalMemory::numberOfCells(void)
{
  return numberOfColumns() * cellsPerColumn_;
}

/*
 * Maps cells to the columns they belong to
 *
 * @param cells(set) Cells
 *
 * @return (dict)Mapping from columns to their cells in `cells`
 */
map<Int, vector<Cell>> TemporalMemory::mapCellsToColumns(vector<Cell>& cells)
{
  map<Int, vector<Cell>> cellsForColumns;

  for (Cell cell : cells)
  {
    Int column = columnForCell(cell);
    cellsForColumns[column].push_back(cell);
  }

  return cellsForColumns;
}

/*
 * Raises an error if column index is invalid.
 *
 * @param column(int) Column index
 */
bool TemporalMemory::_validateColumn(Int column)
{
  if (column >= 0 && column < numberOfColumns())
    return true;

  NTA_THROW << "Invalid column " << column;
  return false;
}

/*
 * Raises an error if cell index is invalid.
 *
 * @param cell(int) Cell index
 */
bool TemporalMemory::_validateCell(Cell& cell)
{
  if (cell.idx < numberOfCells())
    return true;

  NTA_THROW << "Invalid cell " << cell.idx;
  return false;
}

/*
 * Raises an error if segment is invalid.
 *
 * \note { Relies on activeSegments_ being iteratable }
 *
 * @param segment segment index
 */
bool TemporalMemory::_validateSegment(Segment& segment)
{
  if (activeSegments.size() == 0 && segment.idx <= SEGMENT_MAX &&
    _validateCell(segment.cell))
    return true;

  if (find(activeSegments.begin(), activeSegments.end(), segment)
    != activeSegments.end())
    return true;

  NTA_THROW << "Invalid segment" << segment.idx;
  return false;
}

/*
 * Raises an error if permanence is invalid.
 *
 * @param permanence (float) Permanence
 */
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

Int TemporalMemory::getNumColumns() const
{
  return numColumns_;
}

Int TemporalMemory::getCellsPerColumn() const
{
  return cellsPerColumn_;
}

Int TemporalMemory::getActivationThreshold() const
{
  return activationThreshold_;
}

void TemporalMemory::setActivationThreshold(Int activationThreshold)
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

Int TemporalMemory::getMinThreshold() const
{
  return minThreshold_;
}

void TemporalMemory::setMinThreshold(Int minThreshold)
{
  minThreshold_ = minThreshold;
}

Int TemporalMemory::getMaxNewSynapseCount() const
{
  return maxNewSynapseCount_;
}

void TemporalMemory::setMaxNewSynapseCount(Int maxNewSynapseCount)
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

/* create a RNG with given seed */
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
    << endl;

  outStream << columnDimensions_.size() << " ";
  for (auto & elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << predictiveCells.size() << " ";
  for (Cell elem : predictiveCells) {
    outStream << elem.idx << " ";
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
  for (Segment elem : activeSegments) {
    outStream << elem.idx << " ";
    outStream << elem.cell.idx << " ";
  }
  outStream << endl;

//  connections.save(outStream);
//  outStream << endl;

//  _rng.save(outStream);
//  outStream << endl;

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

  auto _connections = proto.initConnections();
  connections.write(_connections);

  auto random = proto.initRandom();
  _rng.write(random);

  auto _activeCells = proto.initActiveCells(activeCells.size());
  for (UInt i = 0; i < activeCells.size(); ++i)
  {
    _activeCells.set(i, activeCells[i].idx);
  }

  auto _predictiveCells = proto.initPredictiveCells(predictiveCells.size());
  for (UInt i = 0; i < predictiveCells.size(); ++i)
  {
    _predictiveCells.set(i, predictiveCells[i].idx);
  }

  auto _activeSegments = proto.initActiveSegments(activeSegments.size());
  for (UInt i = 0; i < activeSegments.size(); ++i)
  {
    _activeSegments.set(i, activeSegments[i].cell.idx);
  }

  auto _winnerCells = proto.initWinnerCells(winnerCells.size());
  for (UInt i = 0; i < winnerCells.size(); ++i)
  {
    _winnerCells.set(i, winnerCells[i].idx);
  }
}

void TemporalMemory::write(ostream& stream) const
{
  capnp::MallocMessageBuilder message;
  TemporalMemoryProto::Builder proto = message.initRoot<TemporalMemoryProto>();
  write(proto);

  kj::std::StdOutputStream out(stream);
  capnp::writeMessage(out, message);
}

// Implementation note: this method sets up the instance using data from
// inStream. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
void TemporalMemory::read(istream& stream)
{
  kj::std::StdInputStream in(stream);

  capnp::InputStreamMessageReader message(in);
  TemporalMemoryProto::Reader proto = message.getRoot<TemporalMemoryProto>();
  read(proto);
}

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

  auto _connections = proto.getConnections();
  connections.read(_connections);
  
  auto random = proto.getRandom();
  _rng.read(random);

  activeCells.clear();
  for (auto value : proto.getActiveCells())
  {
    activeCells.push_back(Cell(value));
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
    winnerCells.push_back(Cell(value));
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
  Int version;
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
    >> permanenceDecrement_;

  // Retrieve vectors.
  UInt numColumnDimensions;
  inStream >> numColumnDimensions;
  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++) {
    inStream >> columnDimensions_[i];
  }

  UInt numPredictiveCells;
  inStream >> numPredictiveCells;
  predictiveCells.resize(numPredictiveCells);
  for (UInt i = 0; i < numPredictiveCells; i++) {
    inStream >> predictiveCells[i].idx;
  }

  UInt numActiveCells;
  inStream >> numActiveCells;
  activeCells.resize(numActiveCells);
  for (UInt i = 0; i < numActiveCells; i++) {
    inStream >> activeCells[i].idx;
  }

  UInt numWinnerCells;
  inStream >> numWinnerCells;
  winnerCells.resize(numWinnerCells);
  for (UInt i = 0; i < numWinnerCells; i++) {
    inStream >> winnerCells[i].idx;
  }

  UInt numActiveSegments;
  inStream >> numActiveSegments;
  activeSegments.resize(numActiveSegments);
  for (UInt i = 0; i < numActiveSegments; i++) {
    inStream >> activeSegments[i].idx;
    inStream >> activeSegments[i].cell.idx;
  }

//  connections.load(inStream);
//  _rng.load(inStream);

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
    << "version                     = " << version_ << std::endl
    << "numColumns                  = " << getNumColumns() << std::endl
    << "cellsPerColumn              = " << getCellsPerColumn() << std::endl
    << "activationThreshold         = " << getActivationThreshold() << std::endl
    << "initialPermanence           = " << getInitialPermanence() << std::endl
    << "connectedPermanence         = " << getConnectedPermanence() << std::endl
    << "minThreshold                = " << getMinThreshold() << std::endl
    << "maxNewSynapseCount          = " << getMaxNewSynapseCount() << std::endl
    << "permanenceIncrement         = " << getPermanenceIncrement() << std::endl
    << "permanenceDecrement         = " << getPermanenceDecrement() << std::endl;
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
