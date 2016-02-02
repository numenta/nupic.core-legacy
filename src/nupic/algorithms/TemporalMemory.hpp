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
 * Definitions for the Temporal Memory in C++
 */

#ifndef NTA_TEMPORAL_MEMORY_HPP
#define NTA_TEMPORAL_MEMORY_HPP

#include <vector>
#include <nupic/types/Serializable.hpp>
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
      class TemporalMemory : public Serializable<TemporalMemoryProto> {
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
         * @param predictedSegmentDecrement Amount by which active permanences of synapses of previously predicted but inactive segments are decremented.
         * @param seed                 Seed for the random number generator.
         *
         * Notes:
         *
         * predictedSegmentDecrement: A good value is just a bit larger than
         * (the column-level sparsity * permanenceIncrement). So, if column-level
         * sparsity is 2% and permanenceIncrement is 0.01, this parameter should be
         * something like 4% * 0.01 = 0.0004).
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
          Permanence predictedSegmentDecrement = 0.0,
          Int seed = 42,
          UInt maxSegmentsPerCell=MAX_SEGMENTS_PER_CELL,
          UInt maxSynapsesPerSegment=MAX_SYNAPSES_PER_SEGMENT);

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
          Permanence predictedSegmentDecrement = 0.0,
          Int seed = 42,
          UInt maxSegmentsPerCell=MAX_SEGMENTS_PER_CELL,
          UInt maxSynapsesPerSegment=MAX_SYNAPSES_PER_SEGMENT);

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
         *
         * @param activeColumnsSize Number of active columns
         * @param activeColumns     Indices of active columns
         * @param learn             Whether or not learning is enabled
         *
         * Updates member variables:
         * - `activeCells`       (set)
         * - `winnerCells`       (set)
         * - `activeSegments`    (set)
         * - `predictiveCells`   (set)
         * - `predictedColumns`  (set)
         * - `matchingSegments`  (set)
         * - `matchingCells`     (set)
         */
        virtual void compute(
          UInt activeColumnsSize, UInt activeColumns[], bool learn = true);


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
         * - if orphan decay active
         *   - for each prev matching cell
         *     - if not in active column
         *       - mark it as an predicted but inactive cell
         *
         * @param prevPredictiveCells   Indices of predictive cells in `t-1`
         * @param prevMatchingCells     Indices of matching cells in `t-1`
         * @param activeColumns         Indices of active columns in `t`
         *
         * @return (tuple)Contains:
         *  `activeCells`               (set),
         *  `winnerCells`               (set),
         *  `predictedColumns`          (set),
         *  `predictedInactiveCells`    (set)
         */
        virtual tuple<set<Cell>, set<Cell>, set<UInt>, set<Cell>>
          activateCorrectlyPredictiveCells(
            set<Cell>& prevPredictiveCells,
            set<Cell>& prevMatchingCells,
            set<UInt>& activeColumns);

        /**
         * Phase 2 : Burst unpredicted columns.
         *
         * Pseudocode :
         *
         * - for each unpredicted active column
         *   - mark all cells as active
         *   - mark the best matching cell as winner cell
         *     - (learning)
         *       - if it has no matching segment
         *         - (optimization) if there are prev winner cells
         *           - add a segment to it
         *       - mark the segment as learning
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
         * Pseudocode:
         *
         *   - (learning) for each prev active or learning segment
         *     - if learning segment or from winner cell
         *       - strengthen active synapses
         *       - weaken inactive synapses
         *     - if learning segment
         *       - add some synapses to the segment
         *         - subsample from prev winner cells
         *
         *   - if predictedSegmentDecrement > 0
         *     - for each previously matching segment
         *       - if cell is a predicted inactive cell
         *         - weaken active synapses but don't touch inactive synapses
         *
         * @param prevActiveSegments(set)   Indices of active segments in `t-1`
         * @param learningSegments(set)     Indices of learning segments in `t`
         * @param prevActiveCells(set)      Indices of active cells in `t-1`
         * @param winnerCells(set)          Indices of winner cells in `t`
         * @param prevWinnerCells(set)      Indices of winner cells in `t-1`
         * @param connections(Connections)  Connectivity of layer
         * @param predictedInactiveCells    Indices of predicted inactive cells
         * @param prevMatchingSegments      Indices of matching segments in `t-1`
         */
        virtual void learnOnSegments(
          vector<Segment>& prevActiveSegments,
          vector<Segment>& learningSegments,
          set<Cell>& prevActiveCells,
          set<Cell>& winnerCells,
          set<Cell>& prevWinnerCells,
          Connections& _connections,
          set<Cell>& predictedInactiveCells,
          vector<Segment>& prevMatchingSegments);

        /**
         * Phase 4 : Compute predictive cells due to lateral input
         * on distal dendrites.
         *
         * Pseudocode:
         *
         *   - for each distal dendrite segment with activity >= activationThreshold
         *     - mark the segment as active
         *     - mark the cell as predictive
         *
         *   - if predictedSegmentDecrement > 0
         *     - for each distal dendrite segment with unconnected
         *       activity >=  minThreshold
         *       - mark the segment as matching
         *       - mark the cell as matching
         *
         * Forward propagates activity from active cells to the synapses
         * that touch them, to determine which synapses are active.
         *
         *  @param activeCells(set)         Indices of active cells in `t`
         *  @param connections(Connections) Connectivity of layer
         *
         *  @return (tuple)Contains:
         *   `activeSegments`   (set),
         *   `predictiveCells`  (set),
         *   `matchingSegments` (set),
         *   `matchingCells`    (set)
         */
        virtual tuple<vector<Segment>, set<Cell>, vector<Segment>, set<Cell>>
          computePredictiveCells(
            set<Cell>& activeCells, Connections& connections);


        // ==============================
        //  Helper functions
        // ==============================

        /**
         * Gets the cell with the best matching segment
         * (see `TM.bestMatchingSegment`) that has the
         * largest number of active synapses of all
         * best matching segments.
         *
         * If none were found, pick the least used cell
         * (see `TM.leastUsedCell`)
         *
         * @param cells        Indices of cells
         * @param activeCells  Indices of active cells
         * @param connections  Connectivity of layer
         *
         * @return (tuple)Contains:
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
         * @param segment               Segment index
         * @param activeSynapses        Indices of active synapses
         * @param connections           Connectivity of layer
         * @param permanenceIncrement   Amount to increment active synapses
         *@param permanenceDecrement   Amount to decrement inactive synapses
         */
        void adaptSegment(
          Segment& segment,
          vector<Synapse>& activeSynapses,
          Connections& connections,
          Permanence permanenceIncrement,
          Permanence permanenceDecrement);

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
        * Returns the Cell objects that belong to a column.
        *
        * @param column Column index
        *
        * @return (vector<Cell>) Cell objects
        */
       vector<Cell> cellsForColumnCell(Int column);

        /**
         * Returns the indices of cells that belong to a column.
         *
         * @param column Column index
         *
         * @return (vector<CellIdx>) Cell indices
         */
        vector<CellIdx> cellsForColumn(Int column);

        /**
         * Returns the number of cells in this layer.
         *
         * @return (int) Number of cells
         */
        UInt numberOfCells(void);

        /**
        * Returns the indices of the active cells.
        *
        * @returns (std::vector<CellIdx>) Vector of indices of active cells.
        */
        vector<CellIdx> getActiveCells() const;

        /**
        * Returns the indices of the predictive cells.
        *
        * @returns (std::vector<CellIdx>) Vector of indices of predictive cells.
        */
        vector<CellIdx> getPredictiveCells() const;

        /**
        * Returns the indices of the winner cells.
        *
        * @returns (std::vector<CellIdx>) Vector of indices of winner cells.
        */
        vector<CellIdx> getWinnerCells() const;

        /**
        * Returns the indices of the matching cells.
        *
        * @returns (std::vector<CellIdx>) Vector of indices of matching cells.
        */
        vector<CellIdx> getMatchingCells() const;

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
         * Returns the predicted Segment decrement.
         *
         * @returns Returns the segment decrement
         */
        Permanence getPredictedSegmentDecrement() const;
        void setPredictedSegmentDecrement(Permanence);

       /**
        * Extracts a vector<CellIdx> from a Iterable of Cells.
        *
        * @param Iterable<Cell> Iterable of Cells
        *                       (.e.g. set<Cell>, vector<Cell>).
        *
        * @returns vector<CellIdx> The indices of the Cells in the Iterable.
        */
        template <typename Iterable>
        vector<CellIdx> _cellsToIndices(const Iterable &cellSet) const;

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

        using Serializable::write;
        virtual void write(TemporalMemoryProto::Builder& proto) const override;

        /**
         * Load (deserialize) and initialize the spatial pooler from the
         * specified input stream.
         *
         * @param inStream A valid istream.
         */
        virtual void load(istream& inStream);

        using Serializable::read;
        virtual void read(TemporalMemoryProto::Reader& proto) override;

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
        Permanence predictedSegmentDecrement_;

        UInt version_;
        Random _rng;

      public:
        set<Cell> activeCells;
        set<Cell> winnerCells;
        vector<Segment> activeSegments;
        vector<Cell> predictiveCells;
        set<UInt> predictedColumns;
        vector<Segment> matchingSegments;
        vector<Cell> matchingCells;
        Connections connections;
      };

    } // end namespace temporal_memory
  } // end namespace algorithms
} // end namespace nta

#endif // NTA_TEMPORAL_MEMORY_HPP

