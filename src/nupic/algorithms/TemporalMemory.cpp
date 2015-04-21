/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;


TemporalMemory::TemporalMemory()
{
  version_ = 1;
  connections_ = nullptr;
}

TemporalMemory::~TemporalMemory()
{
  if (connections_)
    delete connections_;
}

/**
 Initialize the temporal memory using the given parameters.

 @param columnDimensions     Dimensions of the column space
 @param cellsPerColumn       Number of cells per column
 @param activationThreshold  If the number of active connected synapses on a segment is at least this threshold, the segment is said to be active.
 @param learningRadius       Radius around cell from which it can sample to form distal dendrite connections.
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
  Int learningRadius,
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
  learningRadius_ = learningRadius;
  initialPermanence_ = initialPermanence;
  connectedPermanence_ = connectedPermanence;
  minThreshold_ = minThreshold;
  maxNewSynapseCount_ = maxNewSynapseCount;
  permanenceIncrement_ = permanenceIncrement;
  permanenceDecrement_ = permanenceDecrement;

  // Initialize member variables

  activeCells_.clear();
  winnerCells_.clear();

  activeSegments_.clear();
  learningSegments_.clear();

  seed_((UInt64)(seed < 0 ? rand() : seed));

  connections_ = new Connections(numberOfCells());
}


// ==============================
//  Main functions
// ==============================

/*
* Indicates the start of a new sequence.Resets sequence state of the TM.
*/
void TemporalMemory::reset(void)
{
  activeCells_.clear();
  winnerCells_.clear();

  activeSegments_.clear();
  learningSegments_.clear();
}

/*
* Feeds input record through TM, performing inference and learning.
* Updates member variables with new state.
*
* @param activeColumns   Indices of active columns in `t`
* @param learn           Whether or not learning is enabled
*/
void TemporalMemory::compute(UInt activeColumns[], bool learn)
{
  set<Cell> activeCells;
  set<Cell> winnerCells;

  set<Segment> activeSegments;

  set<Cell> predictiveCells;
  set<UInt> predictedColumns;

  tie(activeCells, winnerCells, activeSegments, predictiveCells, predictedColumns)
    = computeFn(
    activeColumns,
    predictiveCells_,
    activeSegments_,
    activeCells_,
    winnerCells_,
    *connections_,
    learn);

  activeCells_ = activeCells;
  winnerCells_ = winnerCells;
  activeSegments_ = activeSegments;
  predictiveCells_ = predictiveCells;

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

  outStream << numColumns_ << " " << endl;

  outStream << columnDimensions_.size() << " ";
  for (const UInt & elem : columnDimensions_) {
    outStream << elem << " ";
  }
  outStream << endl;

  outStream << cellsPerColumn_ << " "
    << activationThreshold_ << " "
    << learningRadius_ << " "
    << initialPermanence_ << " "
    << connectedPermanence_ << " "
    << minThreshold_ << " "
    << maxNewSynapseCount_ << " "
    << permanenceIncrement_ << " "
    << permanenceDecrement_ << " " << endl;

  outStream << random_ << " " << endl;

  outStream << endl;
  outStream << "~TemporalMemory" << endl;

}

// Implementation note: this method sets up the instance using data from
// inStream. This method does not call initialize. As such we have to be careful
// that everything in initialize is handled properly here.
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
  inStream >> numColumns_;

  // Retrieve vectors.
  UInt numColumnDimensions;
  inStream >> numColumnDimensions;

  columnDimensions_.resize(numColumnDimensions);
  for (UInt i = 0; i < numColumnDimensions; i++) {
    inStream >> columnDimensions_[i];
  }

  inStream >> cellsPerColumn_;
  inStream >> activationThreshold_;
  inStream >> learningRadius_;
  inStream >> initialPermanence_;
  inStream >> connectedPermanence_;
  inStream >> minThreshold_;
  inStream >> maxNewSynapseCount_;
  inStream >> permanenceIncrement_;
  inStream >> permanenceDecrement_;

  inStream >> random_;

  inStream >> marker;
  NTA_CHECK(marker == "~TemporalMemory");

}

/* create a RNG with given seed */
void TemporalMemory::seed_(UInt64 seed)
{
  random_ = Random(seed);
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

tuple<set<Cell>, set<Cell>, set<Segment>, set<Cell>, set<UInt>>
TemporalMemory::computeFn(
UInt activeColumns[],
set<Cell>& prevPredictiveCells,
set<Segment>& prevActiveSegments,
set<Cell>& prevActiveCells,
set<Cell>& prevWinnerCells,
Connections& connections,
bool learn)
{
  set<UInt> _activeColumns(activeColumns, activeColumns + sizeof activeColumns / sizeof activeColumns[0]);
  set<UInt> predictedColumns;

  set<Cell> activeCells;
  set<Cell> winnerCells;

  tie(activeCells, winnerCells, predictedColumns) = activateCorrectlyPredictiveCells(
      prevPredictiveCells, 
      _activeColumns);

  activeCells.insert(activeCells_.begin(), activeCells_.end());
  winnerCells.insert(winnerCells_.begin(), winnerCells_.end());

  set<Segment> learningSegments;

  tie(activeCells, winnerCells, learningSegments) = burstColumns(
    _activeColumns, 
    predictedColumns, 
    prevActiveCells, 
    prevWinnerCells, 
    connections);

  activeCells.insert(activeCells_.begin(), activeCells_.end());
  winnerCells.insert(winnerCells_.begin(), winnerCells_.end());

  if (learn)
  {
    learnOnSegments(
      prevActiveSegments,
      learningSegments,
      prevActiveCells,
      winnerCells,
      prevWinnerCells,
      connections);
  }

  set<Segment> activeSegments;
  set<Cell> predictiveCells;

  tie(activeSegments, predictiveCells) =
    computePredictiveCells(activeCells, connections);

  return make_tuple(
    activeCells,
    winnerCells,
    activeSegments,
    predictiveCells,
    predictedColumns);
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
tuple<set<Cell>, set<Cell>, set<UInt>>
TemporalMemory::activateCorrectlyPredictiveCells(
set<Cell>& prevPredictiveCells,
set<UInt>& activeColumns)
{
  set<Cell> activeCells;
  set<Cell> winnerCells;

  set<UInt> predictedColumns;

  for (Cell cell : prevPredictiveCells)
  {
    UInt column = columnForCell(cell);

    if (activeColumns.find(column) != activeColumns.end())
    {
      activeCells.insert(cell);
      winnerCells.insert(cell);
      predictedColumns.insert(column);
    }
  }

  return make_tuple(activeCells, winnerCells, predictedColumns);
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
tuple<set<Cell>, set<Cell>, set<Segment>> TemporalMemory::burstColumns(
  set<UInt>& activeColumns,
  set<UInt>& predictedColumns,
  set<Cell>& prevActiveCells,
  set<Cell>& prevWinnerCells,
  Connections& connections)
{
  set<Cell> activeCells;
  set<Cell> winnerCells;

  set<Segment> learningSegments;

  vector<UInt> unpredictedColumns;

  // Resize to the worst case usage
  unpredictedColumns.resize(activeColumns.size() + predictedColumns.size());

  // Remove the predicted columns from the currently active columns
  vector<UInt>::iterator it = set_difference(
    activeColumns.begin(), activeColumns.end(),
    predictedColumns.begin(), predictedColumns.end(),
    unpredictedColumns.begin());

  // Trim remainer of set
  unpredictedColumns.resize(it - unpredictedColumns.begin());

  for (Int column : unpredictedColumns)
  {
    Segment bestSegment;
    Cell bestCell;

    set<Cell> cells = cellsForColumn(column);

    activeCells.insert(cells.begin(), cells.end());

    tie(bestCell, bestSegment) =
      bestMatchingCell(cells, prevActiveCells, connections);

    winnerCells.insert(bestCell);

    if (bestSegment.idx < 0 && prevWinnerCells.size())
      bestSegment = connections.createSegment(bestCell);

    if (bestSegment.idx >= 0)
      learningSegments.insert(bestSegment);
  }

  return make_tuple(activeCells, winnerCells, learningSegments);
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
  set<Segment>& prevActiveSegments,
  set<Segment>& learningSegments,
  set<Cell>& prevActiveCells,
  set<Cell>& winnerCells,
  set<Cell>& prevWinnerCells,
  Connections& connections)
{
  set<Segment> allSegments;
  allSegments.insert(prevActiveSegments.begin(), prevActiveSegments.end());
  allSegments.insert(learningSegments.begin(), learningSegments.end());

  for (Segment segment : allSegments)
  {
    bool isLearningSegment = (learningSegments.find(segment) != learningSegments.end());
    bool isFromWinnerCell = (winnerCells.find(segment.cell) != winnerCells.end());

    vector<Synapse> activeSynapses(activeSynapsesForSegment(
      segment, prevActiveCells, connections));

    if (isLearningSegment || isFromWinnerCell)
      adaptSegment(segment, activeSynapses, connections);

    if (isLearningSegment)
    {
      Int n = maxNewSynapseCount_ - Int(activeSynapses.size());

      for (Cell presynapticCell : pickCellsToLearnOn(n, segment, prevWinnerCells, connections))
      {
        connections.createSynapse(segment, presynapticCell, initialPermanence_);
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
 *	@return (tuple)Contains:
 *   `activeSegments`  (set),
 *   `predictiveCells` (set)
 */

tuple<set<Segment>, set<Cell>> TemporalMemory::computePredictiveCells(
  set<Cell>& activeCells,
  Connections& connections)
{
  map<Segment, Int> numActiveConnectedSynapsesForSegment;

  set<Segment> activeSegments;
  set<Cell> predictiveCells;

  for (Cell cell : activeCells)
  {
    vector<Segment> segmentsForCell = connections.segmentsForCell(cell);
    for (Segment segment : segmentsForCell)
    {
      for (Synapse synapse : connections.synapsesForSegment(segment))
      {
        SynapseData synapseData = connections.dataForSynapse(synapse);
        Segment segment = synapse.segment;
        Real permanence = synapseData.permanence;

        if (permanence >= connectedPermanence_)
        {
          numActiveConnectedSynapsesForSegment[segment] += 1;

          if (numActiveConnectedSynapsesForSegment[segment] >= activationThreshold_)
          {
            activeSegments.insert(segment);
            predictiveCells.insert(Cell(segment.cell.idx));
          }
        }
      }
    }
  }

  return make_tuple(activeSegments, predictiveCells);
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

tuple<Cell, Segment>
TemporalMemory::bestMatchingCell(
set<Cell>& cells,
set<Cell>& activeCells,
Connections& connections)
{
  Int maxSynapses = 0;
  Cell bestCell;
  Segment bestSegment(-1, Cell(0));

  for (Cell cell : cells)
  {
    Int numActiveSynapses;
    Segment segment;

    tie(segment, numActiveSynapses) = bestMatchingSegment(cell, activeCells, connections);

    if (numActiveSynapses > maxSynapses)
    {
      maxSynapses = numActiveSynapses;
      bestCell = cell;
      bestSegment = segment;
    }
  }

  if (maxSynapses == 0)
    bestCell = leastUsedCell(cells, connections);

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
tuple<Segment, Int>
TemporalMemory::bestMatchingSegment(
Cell& cell,
set<Cell>& activeCells,
Connections& connections)
{
  Int maxSynapses = minThreshold_;
  Segment bestSegment(-1, Cell(0));
  Int bestNumActiveSynapses = 0;

  for (Segment segment : connections.segmentsForCell(cell))
  {
    Int numActiveSynapses = 0;

    for (auto synapse : connections.synapsesForSegment(segment))
    {
      SynapseData synapseData = connections.dataForSynapse(synapse);

      if (activeCells.find(synapseData.presynapticCell) != activeCells.end())
        numActiveSynapses += 1;
    }

    if (numActiveSynapses >= maxSynapses)
    {
      maxSynapses = numActiveSynapses;
      bestSegment = segment;
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
  set<Cell>& cells,
  Connections& connections)
{
  vector<Cell> leastUsedCells;
  Int minNumSegments = INT_MAX;

  for (Cell cell : cells)
  {
    Int numSegments = (Int)connections.segmentsForCell(cell).size();

    if (numSegments < minNumSegments)
    {
      minNumSegments = numSegments;
      leastUsedCells.clear();
    }

    if (numSegments == minNumSegments)
      leastUsedCells.push_back(cell);
  }

  Int i = random_.getUInt32((UInt32)leastUsedCells.size());
  Cell leastUsedCell = leastUsedCells[i];

  return leastUsedCell;
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
  set<Cell>& activeCells,
  Connections& connections)
{
  vector<Synapse> synapses;

  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = connections.dataForSynapse(synapse);

    if (find(activeCells.begin(), activeCells.end(), synapseData.presynapticCell) != activeCells.end())
      synapses.push_back(synapse);
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
  Connections& connections)
{
  for (Synapse synapse : connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = connections.dataForSynapse(synapse);
    Permanence permanence = synapseData.permanence;

    if (find(activeSynapses.begin(), activeSynapses.end(), synapse) != activeSynapses.end())
      permanence += permanenceIncrement_;
    else
      permanence -= permanenceDecrement_;

    // Keep permanence within min / max bounds
    //permanence = max(0.0, min(1.0, permanence));
    if (permanence > 1.0)
      permanence = 1.0;
    if (permanence < 0.0)
      permanence = 0.0;

    connections.updateSynapsePermanence(synapse, permanence);
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
set<Cell> TemporalMemory::pickCellsToLearnOn(
  Int iN,
  Segment& segment,
  set<Cell>& winnerCells,
  Connections& connections)
{
  vector<Cell> candidates(winnerCells.begin(), winnerCells.end());

  // Remove cells that are already synapsed on by this segment
  for (auto synapse : connections.synapsesForSegment(segment))
  {
    SynapseData synapseData = connections.dataForSynapse(synapse);
    Cell presynapticCell = synapseData.presynapticCell;

    if (find(candidates.begin(), candidates.end(), presynapticCell) != candidates.end())
      candidates.erase(find(candidates.begin(), candidates.end(), presynapticCell));
  }

  // Pick n cells randomly
  Int n = min(iN, (Int)candidates.size());
  set<Cell> cells;

  for (int c = 0; c < n; c++)
  {
    Int i = random_.getUInt32((UInt32)candidates.size());

    cells.insert(candidates[i]);

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

set<Cell> TemporalMemory::cellsForColumn(Int column)
{
  _validateColumn(column);

  Int start = cellsPerColumn_ * column;
  Int end = start + cellsPerColumn_;

  set<Cell> cellsInColumn;
  for (Int i = start; i < end; i++)
  {
    cellsInColumn.insert(Cell(i));
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
map<Int, vector<Cell>> TemporalMemory::mapCellsToColumns(set<Cell>& cells)
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
  if (activeSegments_.size() == 0 && segment.idx <= SEGMENT_MAX && _validateCell(segment.cell))
    return true;
  if (find(activeSegments_.begin(), activeSegments_.end(), segment) != activeSegments_.end())
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

Int TemporalMemory::getLearningRadius() const
{
  return learningRadius_;
}

void TemporalMemory::setLearningRadius(Int learningRadius)
{
  learningRadius_ = learningRadius;
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

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main SP creation parameters
void TemporalMemory::printParameters()
{
  std::cout << "------------CPP TemporalMemory Parameters ------------------\n";
  std::cout
    << "version                     = " << version_ << std::endl
    << "numColumns                  = " << getNumColumns() << std::endl
    << "cellsPerColumn              = " << getCellsPerColumn() << std::endl
    << "activationThreshold         = " << getActivationThreshold() << std::endl
    << "learningRadius              = " << getLearningRadius() << std::endl
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
