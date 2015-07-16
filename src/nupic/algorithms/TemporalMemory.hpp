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
  * Definitions for the Temporal Memory in C++
  */

#ifndef NTA_TEMPORAL_MEMORY_HPP
#define NTA_TEMPORAL_MEMORY_HPP

#include <vector>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Random.hpp>
#include <nupic/algorithms/Connections.hpp>
#include <nupic/proto/TemporalMemoryProto.capnp.h>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;

namespace nupic {
  namespace algorithms {
    namespace temporal_memory {

      /**
       * CLA temporal memory implementation in C++.
       *
       * The primary public interfaces to this function are the "initialize"
       * and "compute" methods.
       *
       * Example usage:
       *
       *     SpatialPooler sp;
       *     sp.initialize(inputDimensions, columnDimensions, <parameters>);
       *
       *     TemporalMemory tm;
       *     tm.initialize(columnDimensions, <parameters>);
       *
       *     while (true) {
       *        <get input vector, streaming spatiotemporal information>
       *        sp.compute(inputVector, learn, activeColumns)
       *        tm.compute(number of activeColumns, activeColumns, learn)
       *        <do something with output, e.g. add classifiers>
       *     }
       *
       */
      class TemporalMemory {
      public:
        TemporalMemory();

        /**
        * Initialize the temporal memory (TM) using the given parameters.
        *
        * @param columnDimensions     Dimensions of the column space
        * @param cellsPerColumn       Number of cells per column
        * @param activationThreshold  If the number of active connected synapses on a segment is at least this threshold, the segment is said to be active.
        * @param initialPermanence    Initial permanence of a new synapse.
        * @param connectedPermanence  If the permanence value for a synapse is greater than this value, it is said to be connected.
        * @param minThreshold         If the number of synapses active on a segment is at least this threshold, it is selected as the best matching cell in a bursting column.
        * @param maxNewSynapseCount   The maximum number of synapses added to a segment during learning.
        * @param permanenceIncrement  Amount by which permanences of synapses are incremented during learning.
        * @param permanenceDecrement  Amount by which permanences of synapses are decremented during learning.
        * @param seed                 Seed for the random number generator.
        */
        TemporalMemory(
          vector<UInt> columnDimensions,
          UInt cellsPerColumn = 32,
          UInt activationThreshold = 13,
          Permanence initialPermanence = 0.21,
          Permanence connectedPermanence = 0.50,
          UInt minThreshold = 10,
          UInt maxNewSynapseCount = 20,
          Permanence permanenceIncrement = 0.10,
          Permanence permanenceDecrement = 0.10,
          Int seed = 1);

        virtual void initialize(
          vector<UInt> columnDimensions = { 2048 },
          UInt cellsPerColumn = 32,
          UInt activationThreshold = 13,
          Permanence initialPermanence = 0.21,
          Permanence connectedPermanence = 0.50,
          UInt minThreshold = 10,
          UInt maxNewSynapseCount = 20,
          Permanence permanenceIncrement = 0.10,
          Permanence permanenceDecrement = 0.10,
          Int seed = 1);

        virtual ~TemporalMemory();

        //----------------------------------------------------------------------
        //  Main functions
        //----------------------------------------------------------------------

        /**
         * Get the version number of for the TM implementation.
         *
         * @returns Integer version number.
         */
        virtual UInt version() const {
          return version_;
        };

        /**
        * This *only* updates _rng to a new Random using seed.
        *
        * @returns Integer version number.
        */
        void seed_(UInt64 seed);

        /**
         * Indicates the start of a new sequence.
         * Resets sequence state of the TM.
         */
        virtual void reset();

        /**
         * Feeds input record through TM, performing inference and learning.
         * Updates member variables with new state.
         *
         * @param activeColumnsSize Number of active columns
         * @param activeColumns     Indices of active columns in `t`
         * @param learn             Whether or not learning is enabled
         */
        virtual void compute(
          UInt activeColumnsSize, UInt activeColumns[], bool learn = true);

        /**
         * 'Functional' version of compute. Returns new state.
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
         *  `activeCells`       (set),
         *  `winnerCells`       (set),
         *  `activeSegments`    (set),
         *  `predictiveCells`   (set),
         *  'predictedColumns'  (set)
         */
        tuple<set<Cell>, set<Cell>, vector<Segment>, set<Cell>, set<UInt>>
          computeFn(
            UInt activeColumnsSize,
            UInt activeColumns[],
            set<Cell>& prevPredictiveCells,
            vector<Segment>& prevActiveSegments,
            set<Cell>& prevActiveCells,
            set<Cell>& prevWinnerCells,
            Connections& connections,
            bool learn = true);

        // ==============================
        //  Phases
        // ==============================

        /**
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
        virtual tuple<set<Cell>, set<Cell>, set<UInt>>
          activateCorrectlyPredictiveCells(
            set<Cell>& prevPredictiveCells,
            set<UInt>& activeColumns);

        /**
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
        virtual tuple<set<Cell>, set<Cell>, vector<Segment>> burstColumns(
          set<UInt>& activeColumns,
          set<UInt>& predictedColumns,
          set<Cell>& prevActiveCells,
          set<Cell>& prevWinnerCells,
          Connections& connections);

        /**
         * Phase 3 : Perform learning by adapting segments.
         *
         * Pseudocode :
         *
         *		- (learning) for each prev active or learning segment
         * 		 - if learning segment or from winner cell
         *		  - strengthen active synapses
         *		  - weaken inactive synapses
         *		 - if learning segment
         *		  - add some synapses to the segment
         *		    - subsample from prev winner cells
         *
         * @param prevActiveSegments(set)   Indices of active segments in `t-1`
         * @param learningSegments(set)     Indices of learning segments in `t`
         * @param prevActiveCells(set)      Indices of active cells in `t-1`
         * @param winnerCells(set)          Indices of winner cells in `t`
         * @param prevWinnerCells(set)      Indices of winner cells in `t-1`
         * @param connections(Connections)  Connectivity of layer
         */
        virtual void learnOnSegments(
          vector<Segment>& prevActiveSegments,
          vector<Segment>& learningSegments,
          set<Cell>& prevActiveCells,
          set<Cell>& winnerCells,
          set<Cell>& prevWinnerCells,
          Connections& connections);

        /**
         * Phase 4 : Compute predictive cells due to lateral input 
         * on distal dendrites.
         *
         * Pseudocode :
         *
         *		- for each distal dendrite segment with activity >= activationThreshold
         *		  - mark the segment as active
         *		  - mark the cell as predictive
         *
         * Forward propagates activity from active cells to the synapses
         * that touch them, to determine which synapses are active.
         *
         *	@param activeCells(set)         Indices of active cells in `t`
         *	@param connections(Connections) Connectivity of layer
         *
         *	@return (tuple)Contains:
         *   `activeSegments`  (set),
         *   `predictiveCells` (set)
         */
        virtual tuple<vector<Segment>, set<Cell>> computePredictiveCells(
          set<Cell>& activeCells,
          Connections& connections);


        // ==============================
        //  Helper functions
        // ==============================

        /**
         * Gets the cell with the best matching segment
         * (see `TM.bestMatchingSegment`) that has the 
         * largest number of active synapses of all 
         * best matching segments. If none were found,
         * pick the least used cell (see `TM.leastUsedCell`).
         *
         *	@param cells        Indices of cells
         *	@param activeCells  Indices of active cells
         *	@param connections  Connectivity of layer
         *
         *	@return (tuple)Contains:
         *   `foundCell`    (bool),
         *   `bestCell`     (int),
         *   `foundSegment` (bool),
         *   `bestSegment`  (int)
         */
        tuple<bool, Cell, bool, Segment> bestMatchingCell(
          vector<Cell>& cells,
          set<Cell>& activeCells,
          Connections& connections);

        /**
         * Gets the segment on a cell with the largest number of activate
         * synapses, including all synapses with non - zero permanences.
         *
         * @param cell          Cell index
         * @param activeCells   Indices of active cells
         * @param connections   Connectivity of layer
         *
         * @return (tuple)Contains:
         *  `segment`                 (int),
         *  `connectedActiveSynapses` (set)
         */
        tuple<bool, Segment, Int> bestMatchingSegment(
          Cell& cell,
          set<Cell>& activeCells,
          Connections& connections);

        /**
         * Gets the cell with the smallest number of segments.
         * Break ties randomly.
         *
         * @param cells         Indices of cells
         * @param connections   Connectivity of layer
         *
         * @return (int) Cell index
         */
        Cell leastUsedCell(
          vector<Cell>& cells,
          Connections& connections);

        /**
         * Returns the synapses on a segment that are active due to 
         * lateral input from active cells.
         *
         * @param segment       Segment index
         * @param activeCells   Indices of active cells
         * @param connections   Connectivity of layer
         *
         * @return (set) Indices of active synapses on segment
         */
        vector<Synapse> activeSynapsesForSegment(
          Segment& segment,
          set<Cell>& activeCells,
          Connections& connections);

        /**
         * Updates synapses on segment.
         * Strengthens active synapses; weakens inactive synapses.
         *
         * @param segment        Segment index
         * @param activeSynapses Indices of active synapses
         * @param connections    Connectivity of layer
         */
        void adaptSegment(
          Segment& segment,
          vector<Synapse>& activeSynapses,
          Connections& connections);

        /**
         * Pick cells to form distal connections to.
         *
         * TODO : Respect topology and learningRadius
         *
         * @param n             Number of cells to pick
         * @param segment       Segment index
         * @param winnerCells   Indices of winner cells in `t`
         * @param connections   Connectivity of layer
         *
         * @return (set) Indices of cells picked
         */
        set<Cell> pickCellsToLearnOn(
          Int n,
          Segment& segment,
          set<Cell>& winnerCells,
          Connections& connections);

        /**
         * Returns the index of the column that a cell belongs to.
         *
         * @param cell Cell index
         *
         * @return (int) Column index
         */
        Int columnForCell(Cell& cell);

        /**
         * Returns the indices of cells that belong to a column.
         *
         * @param column Column index
         *
         * @return (set) Cell indices
         */
        vector<Cell> cellsForColumn(Int column);

        /**
         * Returns the number of cells in this layer.
         *
         * @return (int) Number of cells
         */
        UInt numberOfCells(void);

        /**
         * Maps cells to the columns they belong to
         *
         * @param cells Cells
         *
         * @return (dict) Mapping from columns to their cells in `cells`
         */
        map<Int, set<Cell>> mapCellsToColumns(set<Cell>& cells);

        /**
         * Returns the dimensions of the columns in the region.
         *
         * @returns Integer number of column dimension
         */
        vector<UInt> getColumnDimensions() const;

        /**
         * Returns the total number of columns.
         *
         * @returns Integer number of column numbers
         */
        UInt numberOfColumns() const;

        /**
         * Returns the number of cells per column.
         *
         * @returns Integer number of cells per column
         */
        UInt getCellsPerColumn() const;

        /**
         * Returns the activation threshold.
         *
         * @returns Integer number of the activation threshold
         */
        UInt getActivationThreshold() const;
        void setActivationThreshold(UInt);

        /**
         * Returns the initial permanence.
         *
         * @returns Initial permanence
         */
        Permanence getInitialPermanence() const;
        void setInitialPermanence(Permanence);

        /**
         * Returns the connected permanance.
         *
         * @returns Returns the connected permanance
         */
        Permanence getConnectedPermanence() const;
        void setConnectedPermanence(Permanence);

        /**
         * Returns the minimum threshold.
         *
         * @returns Integer number of minimum threshold
         */
        UInt getMinThreshold() const;
        void setMinThreshold(UInt);

        /**
         * Returns the maximum new synapse count.
         *
         * @returns Integer number of maximum new synapse count
         */
        UInt getMaxNewSynapseCount() const;
        void setMaxNewSynapseCount(UInt);

        /**
         * Returns the permanence increment.
         *
         * @returns Returns the Permanence increment
         */
        Permanence getPermanenceIncrement() const;
        void setPermanenceIncrement(Permanence);

        /**
         * Returns the permanence decrement.
         *
         * @returns Returns the Permanence decrement
         */
        Permanence getPermanenceDecrement() const;
        void setPermanenceDecrement(Permanence);

        /**
         * Raises an error if column index is invalid.
         *
         * @param column Column index
         */
        bool _validateColumn(UInt column);

        /**
         * Raises an error if cell index is invalid.
         *
         * @param cell Cell index
         */
        bool _validateCell(Cell& cell);

        /**
         * Raises an error if segment is invalid.
         *
         * @param segment segment index
         */
        bool _validateSegment(Segment& segment);

        /**
         * Raises an error if segment is invalid.
         *
         * @param permanence (float) Permanence
         */
        bool _validatePermanence(Real permanence);

        /**
         * Save (serialize) the current state of the spatial pooler to the
         * specified file.
         *
         * @param fd A valid file descriptor.
         */
        virtual void save(ostream& outStream) const;

        virtual void write(ostream& stream) const;
        virtual void write(TemporalMemoryProto::Builder& proto) const;

        /**
         * Load (deserialize) and initialize the spatial pooler from the
         * specified input stream.
         *
         * @param inStream A valid istream.
         */
        virtual void load(istream& inStream);

        virtual void read(istream& stream);
        virtual void read(TemporalMemoryProto::Reader& proto);

        /**
         * Returns the number of bytes that a save operation would result in.
         * Note: this method is currently somewhat inefficient as it just does
         * a full save into an ostream and counts the resulting size.
         *
         * @returns Integer number of bytes
         */
        virtual UInt persistentSize() const;

        //----------------------------------------------------------------------
        // Debugging helpers
        //----------------------------------------------------------------------

        /**
         * Print the main TM creation parameters
         */
        void printParameters();

        /**
         * Print the given UInt array in a nice format
         */
        void printState(vector<UInt> &state);

        /**
         * Print the given Real array in a nice format
         */
        void printState(vector<Real> &state);

      protected:
        UInt numColumns_;
        vector<UInt> columnDimensions_;
        UInt cellsPerColumn_;
        UInt activationThreshold_;
        UInt minThreshold_;
        UInt maxNewSynapseCount_;
        Permanence initialPermanence_;
        Permanence connectedPermanence_;
        Permanence permanenceIncrement_;
        Permanence permanenceDecrement_;

        UInt version_;
        Random _rng;

      public:
        set<Cell> activeCells;
        set<Cell> winnerCells;
        vector<Segment> activeSegments;
        vector<Cell> predictiveCells;
        set<UInt> predictedColumns;
        Connections connections;
      };

    } // end namespace temporal_memory
  } // end namespace algorithms
} // end namespace nta

#endif //NTA_TEMPORAL_MEMORY_HPP
