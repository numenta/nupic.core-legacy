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
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

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
	UInt cellsPerColumn, 
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

	// TODO : Validate all parameters(and add validation tests)

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
  
  connections_ = Connections(numberOfCells());
	
  _random = Random(seed);

	activeCells_.clear();
	predictiveCells_.clear();
	activeSegments_.clear();
	winnerCells_.clear();

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
  predictiveCells_.clear();
  activeSegments_.clear();
  winnerCells_.clear();
}

/*
 * Feeds input record through TM, performing inference and learning.
 * Updates member variables with new state.
 *
 * @param activeColumns   Indices of active columns in `t`
 * @param learn           Whether or not learning is enabled
 */
void TemporalMemory::compute(vector<Int>& activeColumns, bool learn)
{
//  computeFn(activeColumns, ..., learn);
}

/*
 * 'Functional' version of compute.
 * Returns new state.
 *
 * @param activeColumns         Indices of active columns in `t`
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
//tuple<vector<Cell>, vector<Cell>, vector<Segment>, vector<Cell>>
void TemporalMemory::computeFn(
  vector<Int>& activeColumns,
  vector<Cell>& prevPredictiveCells,
  vector<Segment>& prevActiveSegments,
  vector<Cell>& prevActiveCells,
  vector<Cell>& prevWinnerCells,
  Connections& connections,
  bool learn)
{
	activateCorrectlyPredictiveCells(prevPredictiveCells, activeColumns);
	burstColumns(activeColumns, predictedColumns_, prevActiveCells, prevWinnerCells, connections);

  if (learn)
  {
    learnOnSegments(prevActiveSegments, learningSegments_, prevActiveCells, winnerCells_, prevWinnerCells, connections);
  }

	computePredictiveCells(activeCells_, connections);

  return;// make_tuple(activeCells_, winnerCells_, activeSegments_, predictiveCells_, predictedColumns_);
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
//tuple<vector<Cell>, vector<Cell>, vector<Int>>
void TemporalMemory::activateCorrectlyPredictiveCells(
  vector<Cell>& prevPredictiveCells,
  vector<Int>& activeColumns)
{
	for (Cell cell : prevPredictiveCells)
	{
		Int column = columnForCell(cell);

    if (find(activeColumns.begin(), activeColumns.end(), column) != activeColumns.end())
    {
      activeCells_.push_back(cell);
      winnerCells_.push_back(cell);
      predictedColumns_.push_back(column);
    }
	}

  return;// make_tuple(activeCells, winnerCells, predictedColumns);
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
//tuple<vector<Cell>, vector<Cell>, vector<Segment>>
void TemporalMemory::burstColumns(
  vector<Int>& activeColumns,
  vector<Int>& predictedColumns,
  vector<Cell>& prevActiveCells,
  vector<Cell>& prevWinnerCells,
  Connections& connections)
{
  vector<Int> unpredictedColumns;

  // Resize to the worst case usage
  unpredictedColumns.resize(activeColumns.size() + predictedColumns.size());

  // Remove the predicted columns from the currently active columns
  vector<Int>::iterator it = set_difference(
    activeColumns.begin(), activeColumns.end(),
    predictedColumns.begin(), predictedColumns.end(),
    unpredictedColumns.begin());

  // Trim remainer of set
  unpredictedColumns.resize(it - unpredictedColumns.begin());

  for (auto column : unpredictedColumns)
  {
    vector<Cell> cells = cellsForColumn(column);
    activeCells_.insert(activeCells_.end(), cells.begin(), cells.end());

    Segment bestSegment;
    Cell bestCell;

    tie(bestCell, bestSegment) = 
      bestMatchingCell(cells, prevActiveCells, connections);

		winnerCells_.push_back(bestCell);

    if (prevWinnerCells.size())
    {
      bestSegment = connections.createSegment(bestCell);
      learningSegments_.push_back(bestSegment);
    }
	}

  return;// make_tuple(activeCells_, winnerCells_, learningSegments_);
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
  Connections& connections)
{
  for (Segment segment : prevActiveSegments)
  {
    bool isFromWinnerCell = false;// (find(winnerCells_.begin(), winnerCells_.end(), connections.cellForSegment(segment)) != winnerCells.end());

    vector<Synapse> activeSynapses = activeSynapsesForSegment(segment, prevActiveCells, connections);

    if (isFromWinnerCell)
      adaptSegment(segment, activeSynapses, connections);
  }

  for (auto segment : learningSegments)
	{
    vector<Synapse> activeSynapses = activeSynapsesForSegment(segment, prevActiveCells, connections);

  	adaptSegment(segment, activeSynapses, connections);

		Int n = maxNewSynapseCount_ - Int(activeSynapses.size());

		for (auto presynapticCell : 
      pickCellsToLearnOn(n, segment, prevWinnerCells, connections))
		{
			connections.createSynapse(segment, presynapticCell, initialPermanence_);
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
//tuple<vector<Int>, vector<Cell>>
void TemporalMemory::computePredictiveCells(
  vector<Cell>& activeCells,
  Connections& connections)
{
	map<Segment, Int> numActiveConnectedSynapsesForSegment;

	for (Cell cell : activeCells)
	{
/*
    for (Synapse synapse : connections.synapsesForPresynapticCell(cell).values())
		{
			Segment segment = synapse.segment;
			Real permanence = connections.dataForSynapse(synapse).permanence;

			if (permanence >= connectedPermanence_)
			{
				numActiveConnectedSynapsesForSegment[segment] += 1;

				if (numActiveConnectedSynapsesForSegment[segment] >= activationThreshold_)
				{
					activeSegments_.push_back(segment);
					predictiveCells_.push_back(connections.cellForSegment(segment));
				}
			}
		}
*/
	}

  return;// make_tuple(activeSegments, predictiveCells);
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
    vector<Cell>& cells,
    vector<Cell>& activeCells,
    Connections& connections)
{
	Int maxSynapses = 0;
	Cell bestCell(-1);
	Segment bestSegment(-1,bestCell);

	for(Cell cell : cells)
	{
    Int numActiveSynapses;
    Segment segment;

		tie(segment, numActiveSynapses) = bestMatchingSegment(cell, activeCells, connections);

		if (_validateSegment(segment) && numActiveSynapses > maxSynapses)
		{
			maxSynapses = numActiveSynapses;
			bestCell = cell;
			bestSegment = segment;
		}
	}

	if (_validateCell(bestCell))
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
    vector<Cell>& activeCells,
    Connections& connections)
{
	Int maxSynapses = minThreshold_;
  Segment bestSegment(-1,Cell(-1));
	Int bestNumActiveSynapses;

	for (Segment segment : connections.segmentsForCell(cell))
	{
		Int numActiveSynapses = 0;

		for (auto synapse : connections.synapsesForSegment(segment))
		{
			SynapseData synapseData = connections.dataForSynapse(synapse);

			if (find(activeCells.begin(), activeCells.end(), synapseData.presynapticCell) != activeCells.end())
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
  vector<Cell>& cells,
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

  Int i = _random.getUInt32((UInt32)leastUsedCells.size());
  return leastUsedCells[i];
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
		permanence = max((Real)0.0, min((Real)1.0, permanence));

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
vector<Cell> TemporalMemory::pickCellsToLearnOn(
  Int iN,
  Segment& segment,
  vector<Cell>& winnerCells,
  Connections& connections)
{
  vector<Cell> candidates(winnerCells);

	// Remove cells that are already synapsed on by this segment
	for (auto synapse : connections.synapsesForSegment(segment))
	{
    SynapseData synapseData = connections.dataForSynapse(synapse);
		Cell presynapticCell = synapseData.presynapticCell;

		if (find(candidates.begin(), candidates.end(), presynapticCell) != candidates.end())
			candidates.erase(find(candidates.begin(), candidates.end(),presynapticCell));
	}

	Int n = min(iN, (Int)candidates.size());

  sort(candidates.begin(), candidates.end());
	
  vector<Cell> cells;

	// Pick n cells randomly
  for (int c = 0; c < n; c++)
	{
		Int i = _random.getUInt32((UInt32)candidates.size());
		
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

  return Int(cell.idx / cellsPerColumn_);
}

/*
 * Returns the indices of cells that belong to a column.
 *
 * @param column(int) Column index
 *
 * @return (set)Cell indices
 */

// Cell number index generator:
struct _cell_number_iterator {
  CellIdx current;
  _cell_number_iterator(Int start) { current = start; }
  Cell operator()() { return Cell(current++); }
};

vector<Cell> TemporalMemory::cellsForColumn(Int column)
{
  _validateColumn(column);

  Int start = cellsPerColumn_ * column;
  Int end = start + cellsPerColumn_;

  vector<Cell> cellsInColumn;
  cellsInColumn.resize(end - start);

  _cell_number_iterator intIter(start);
  std::generate(cellsInColumn.begin(), cellsInColumn.end(), intIter);

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
    acc *= column;

	return acc;
}


/*
	* Returns the number of cells in this layer.
	*
	* @return (int)Number of cells
	*/
Int TemporalMemory::numberOfCells(void)
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

	for(Cell cell : cells)
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
	if (column >= numberOfColumns() || column < 0)
	{
		NTA_THROW << "Invalid column " << column;
		return false;
	}
	return true;
}

/*
 * Raises an error if cell index is invalid.
 *
 * @param cell(int) Cell index
 */
bool TemporalMemory::_validateCell(Cell& cell)
{
  if (cell.idx >= (UInt32)numberOfCells() || cell.idx < 0)
  {
    NTA_THROW << "Invalid cell " << cell.idx;
    return false;
  }
  return true;
}

/*
 * Raises an error if segment is invalid.
 *
 * @param segment segment index
 */
bool TemporalMemory::_validateSegment(Segment& segment)
{
  if (find(activeSegments_.begin(), activeSegments_.end(), segment) != activeSegments_.end())
    return true;

  NTA_THROW << "Invalid segment" << segment.idx;
  return false;
}

/*
 * Raises an error if segment is invalid.
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

UInt TemporalMemory::persistentSize()
{
	// TODO: this won't scale!
	stringstream s;
	s.flags(ios::scientific);
	s.precision(numeric_limits<double>::digits10 + 1);
	
	save(s);
	
	return (UInt)s.str().size();
}

void TemporalMemory::save(ostream& outStream)
{
	// Write a starting marker and version.
	outStream << "TemporalMemory" << endl;
	outStream << version_ << endl;

	// Store the simple variables first.
	outStream << numColumns_ << " "
		<< endl;

	outStream << columnDimensions_.size() << " ";
	for (auto & elem : columnDimensions_) {
		outStream << elem << " ";
	}
	outStream << endl;

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

	inStream >> marker;
	NTA_CHECK(marker == "~TemporalMemory");

}

//----------------------------------------------------------------------
// Debugging helpers
//----------------------------------------------------------------------

// Print the main SP creation parameters
void TemporalMemory::printParameters()
{
	std::cout << "------------CPP TemporalMemory Parameters ------------------\n";
//  std::cout
//    << "version                     = " << version() << std::endl
//    << "numColumns                  = " << getNumColumns() << std::endl;
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
