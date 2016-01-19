/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * Definitions for the Connections class in C++
 */

#ifndef NTA_CONNECTIONS_HPP
#define NTA_CONNECTIONS_HPP

#include <climits>
#include <utility>
#include <vector>

#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/math/Math.hpp>
#include <nupic/proto/ConnectionsProto.capnp.h>

namespace nupic
{

  namespace algorithms
  {

    namespace connections
    {
      typedef UInt32 CellIdx;
      typedef UInt16 SegmentIdx;
      typedef UInt16 SynapseIdx;
      typedef Real32 Permanence;
      typedef UInt64 Iteration;

      // Defaults
      static const UInt16 MAX_SEGMENTS_PER_CELL = 255;
      static const UInt16 MAX_SYNAPSES_PER_SEGMENT = 255;

      /**
       * Cell class used in Connections.
       *
       * @b Description
       * The Cell class is a data structure that points to a particular cell.
       *
       * @param idx Index of cell.
       *
       */
      struct Cell
      {
        CellIdx idx;

        Cell(CellIdx idx) : idx(idx) {}
        Cell() {}

        bool operator==(const Cell &other) const;
        bool operator<=(const Cell &other) const;
        bool operator<(const Cell &other) const;
        bool operator>=(const Cell &other) const;
        bool operator>(const Cell &other) const;
      };

      /**
       * Segment class used in Connections.
       *
       * @b Description
       * The Segment class is a data structure that points to a particular
       * segment on a particular cell.
       *
       * @param idx     Index of segment.
       * @param cellIdx Index of cell.
       *
       */
      struct Segment
      {
        SegmentIdx idx;
        Cell cell;

        Segment(SegmentIdx idx, Cell cell) : idx(idx), cell(std::move(cell)) {}
        Segment() {}

        bool operator==(const Segment &other) const;
        bool operator<=(const Segment &other) const;
        bool operator<(const Segment &other) const;
        bool operator>=(const Segment &other) const;
        bool operator>(const Segment &other) const;
      };

      /**
       * Synapse class used in Connections.
       *
       * @b Description
       * The Synapse class is a data structure that points to a particular
       * synapse on a particular segment on a particular cell.
       *
       * @param idx        Index of synapse in segment.
       * @param segmentIdx Index of segment in cell.
       * @param cellIdx    Index of cell.
       *
       */
      struct Synapse
      {
        SynapseIdx idx;
        Segment segment;

        Synapse(SynapseIdx idx, Segment segment) : idx(idx), segment(std::move(segment)) {}
        Synapse() {}

        bool operator==(const Synapse &other) const;
      };

      /**
       * SynapseData class used in Connections.
       *
       * @b Description
       * The SynapseData class is a data structure that contains the data for a
       * synapse on a segment.
       *
       * @param presynapticCellIdx Cell that this synapse gets input from.
       * @param permanence         Permanence of synapse.
       * @param destroyed          Whether this synapse has been destroyed.
       *
       */
      struct SynapseData
      {
        Cell presynapticCell;
        Permanence permanence;
        bool destroyed;
      };

      /**
       * SegmentData class used in Connections.
       *
       * @b Description
       * The SegmentData class is a data structure that contains the data for a
       * segment on a cell.
       *
       * @param synapses          Data for synapses that this segment contains.
       * @param destroyed         Whether this segment has been destroyed.
       * @param lastUsedIteration The iteration that this segment was last used at.
       *
       */
      struct SegmentData
      {
        std::vector<SynapseData> synapses;
        bool destroyed;
        Iteration lastUsedIteration;
      };

      /**
       * CellData class used in Connections.
       *
       * @b Description
       * The CellData class is a data structure that contains the data for a
       * cell.
       *
       * @param segments Data for segments that this cell contains.
       *
       */
      struct CellData
      {
        std::vector<SegmentData> segments;
      };

      /**
       * Activity class used in Connections.
       *
       * @b Description
       * The Activity class is a data structure that represents the
       * activity of a collection of cells, as computed by propagating
       * input through connections.
       *
       */
      struct Activity
      {
        std::map< Cell, std::vector<Segment> > activeSegmentsForCell;
        std::map<Segment, SynapseIdx> numActiveSynapsesForSegment;
      };

      /**
       * Connections implementation in C++.
       *
       * @b Description
       * The Connections class is a data structure that represents the
       * connections of a collection of cells. It is used in the HTM
       * learning algorithms to store and access data related to the
       * connectivity of cells.
       *
       * It's main utility is to provide a common, optimized data structure
       * that all HTM learning algorithms can use. It is flexible enough to
       * support any learning algorithm that operates on a collection of cells.
       *
       * Each type of connection (proximal, distal, apical) should be
       * represented by a different instantiation of this class. This class
       * will help compute the activity along those connections due to active
       * input cells. The responsibility for what effect that activity has on
       * the cells and connections lies in the user of this class.
       *
       * This class is optimized to store connections between cells, and
       * compute the activity of cells due to input over the connections.
       *
       */
      class Connections : public Serializable<ConnectionsProto>
      {
      public:
        static const UInt16 VERSION = 1;

        /**
         * Connections empty constructor.
         * (Does not call `initialize`.)
         */
        Connections() {};

        /**
         * Connections constructor.
         *
         * @param numCells              Number of cells.
         * @param maxSegmentsPerCell    Maximum number of segments per cell.
         * @param maxSynapsesPerSegment Maximum number of synapses per segment.
         */
        Connections(CellIdx numCells,
                    SegmentIdx maxSegmentsPerCell=MAX_SEGMENTS_PER_CELL,
                    SynapseIdx maxSynapsesPerSegment=MAX_SYNAPSES_PER_SEGMENT);

        virtual ~Connections() {}

        /**
         * Initialize connections.
         *
         * @param numCells              Number of cells.
         * @param maxSegmentsPerCell    Maximum number of segments per cell.
         * @param maxSynapsesPerSegment Maximum number of synapses per segment.
         */
        void initialize(CellIdx numCells,
                        SegmentIdx maxSegmentsPerCell,
                        SynapseIdx maxSynapsesPerSegment);

        /**
         * Creates a segment on the specified cell.
         *
         * @param cell Cell to create segment on.
         *
         * @retval Created segment.
         */
        Segment createSegment(const Cell& cell);

        /**
         * Creates a synapse on the specified segment.
         *
         * @param segment         Segment to create synapse on.
         * @param presynapticCell Cell to synapse on.
         * @param permanence      Initial permanence of new synapse.
         *
         * @reval Created synapse.
         */
        Synapse createSynapse(const Segment& segment,
                              const Cell& presynapticCell,
                              Permanence permanence);

        /**
         * Destroys segment.
         *
         * @param segment Segment to destroy.
         */
        void destroySegment(const Segment& segment);

        /**
         * Destroys synapse.
         *
         * @param synapse Synapse to destroy.
         */
        void destroySynapse(const Synapse& synapse);

        /**
         * Updates a synapse's permanence.
         *
         * @param synapse    Synapse to update.
         * @param permanence New permanence.
         */
        void updateSynapsePermanence(const Synapse& synapse,
                                     Permanence permanence);

        /**
         * Gets the segments for a cell.
         *
         * @param cell Cell to get segments for.
         *
         * @retval Segments on cell.
         */
        std::vector<Segment> segmentsForCell(const Cell& cell) const;

        /**
         * Gets the synapses for a segment.
         *
         * @param segment Segment to get synapses for.
         *
         * @retval Synapses on segment.
         */
        std::vector<Synapse> synapsesForSegment(const Segment& segment);

        /**
         * Gets the data for a segment.
         *
         * @param segment Segment to get data for.
         *
         * @retval Segment data.
         */
        SegmentData dataForSegment(const Segment& segment) const;

        /**
         * Gets the data for a synapse.
         *
         * @param synapse Synapse to get data for.
         *
         * @retval Synapse data.
         */
        SynapseData dataForSynapse(const Synapse& synapse) const;

        /**
         * Returns the synapses for the source cell that they synapse on.
         *
         * @param presynapticCell(int) Source cell index
         *
         * @return (set)Synapse indices
         */
        std::vector<Synapse> synapsesForPresynapticCell(const Cell& presynapticCell) const;

        /**
         * Gets the segment with the most active synapses due to given input,
         * from among all the segments on all the given cells.
         *
         * @param cells            Cells to look among.
         * @param input            Active cells in the input.
         * @param synapseThreshold Only consider segments with number of active synapses greater than this threshold.
         * @param retSegment       Segment to return.
         *
         * @retval Segment found?
         */
        bool mostActiveSegmentForCells(const std::vector<Cell>& cells,
                                       std::vector<Cell> input,
                                       SynapseIdx synapseThreshold,
                                       Segment& retSegment) const;

        /**
         * Gets the segment that was least recently used from among all the
         * segments on the given cell.
         *
         * @param cell       Cell whose segments to consider.
         * @param retSegment Segment to return.
         *
         * @retval False if cell has no segments.
         */
        bool leastRecentlyUsedSegment(const Cell& cell,
                                      Segment& retSegment) const;

         /**
          * Gets the synapse with the lowest permanence on the segment.
          *
          * @param segment       Segment whose synapses to consider.
          * @param retSynapse    Synapse with the lowest permanence.
          *
          * @retval False if segment has no synapses.
          */
         bool minPermanenceSynapse(const Segment& segment,
                                   Synapse& retSynapse) const;

        /**
         * Forward-propagates input to synapses, dendrites, and cells, to
         * compute their activity.
         *
         * @param input               Active cells in the input.
         * @param permanenceThreshold Only consider synapses with permanences greater than this threshold.
         * @param synapseThreshold    Only consider segments with number of active synapses greater than this threshold.
         *
         * @retval Activity to return.
         */
        Activity computeActivity(const std::vector<Cell>& input,
                                 Permanence permanenceThreshold,
                                 SynapseIdx synapseThreshold,
                                 bool recordIteration=true);

        /**
         * Gets the active segments from activity.
         *
         * @param activity Activity.
         *
         * @retval Active segments.
         */
        std::vector<Segment> activeSegments(const Activity& activity);

        /**
         * Gets the active cells from activity.
         *
         * @param activity Activity.
         *
         * @retval Active cells.
         */
        std::vector<Cell> activeCells(const Activity& activity);

        // Serialization

        /**
         * Saves serialized data to output stream.
         */
        virtual void save(std::ostream& outStream) const;

        /**
         * Writes serialized data to output stream.
         */
        using Serializable::write;

        /**
         * Writes serialized data to proto object.
         */
        virtual void write(ConnectionsProto::Builder& proto) const override;

        /**
         * Loads serialized data from input stream.
         */
        virtual void load(std::istream& inStream);

        /**
         * Reads serialized data from input stream.
         */
        using Serializable::read;

        /**
         * Reads serialized data from proto object.
         */
        virtual void read(ConnectionsProto::Reader& proto) override;

        // Debugging

        /**
         * Gets the number of segments.
         *
         * @retval Number of segments.
         */
        UInt numSegments() const;

        /**
         * Gets the number of synapses.
         *
         * @retval Number of synapses.
         */
        UInt numSynapses() const;

        /**
         * Comparison operator.
         */
        bool operator==(const Connections &other) const;

      private:
        std::vector<CellData> cells_;
        // Mapping (presynaptic cell => synapses) used in forward propagation
        std::map< Cell, std::vector<Synapse> > synapsesForPresynapticCell_;
        UInt numSegments_;
        UInt numSynapses_;
        SegmentIdx maxSegmentsPerCell_;
        SynapseIdx maxSynapsesPerSegment_;
        Iteration iteration_;
      }; // end class Connections

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nupic

#endif // NTA_CONNECTIONS_HPP
