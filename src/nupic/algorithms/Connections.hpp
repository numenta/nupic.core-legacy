/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
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
        CellIdx cell;

        Segment(SegmentIdx idx, CellIdx cell) : idx(idx), cell(cell) {}
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

        Synapse(SynapseIdx idx, Segment segment) : idx(idx), segment(segment) {}
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
        CellIdx presynapticCell;
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
       * @param flatIdx           This segment's index in flattened lists of all segments
       *
       */
      struct SegmentData
      {
        std::vector<SynapseData> synapses;
        UInt32 numDestroyedSynapses;
        bool destroyed;
        Iteration lastUsedIteration;
        UInt32 flatIdx;
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
        UInt32 numDestroyedSegments;
      };

      /**
       * A segment + overlap pair.
       *
       * @b Description
       * The "overlap" describes the number of active synapses on the segment.
       * Depending on the use case, these synapses may have a permanence below
       * the connected threshold.
       */
      struct SegmentOverlap
      {
        Segment segment;
        UInt32 overlap;
      };

      /**
       * A base class for Connections event handlers.
       *
       * @b Description
       * This acts as a plug-in point for visualizations.
       */
      class ConnectionsEventHandler
      {
      public:
        virtual ~ConnectionsEventHandler() {}

        /**
         * Called after a segment is created.
         */
        virtual void onCreateSegment(Segment segment) {}

        /**
         * Called before a segment is destroyed.
         */
        virtual void onDestroySegment(Segment segment) {}

        /**
         * Called after a synapse is created.
         */
        virtual void onCreateSynapse(Synapse synapse) {}

        /**
         * Called before a synapse is destroyed.
         */
        virtual void onDestroySynapse(Synapse synapse) {}

        /**
         * Called before a synapse's permanence is changed.
         */
        virtual void onUpdateSynapsePermanence(Synapse synapse,
                                               Permanence permanence) {}
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
       * Its main utility is to provide a common, optimized data structure
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
        friend class SegmentExcitationTally;

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
                    SegmentIdx maxSegmentsPerCell=255,
                    SynapseIdx maxSynapsesPerSegment=255);

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
        Segment createSegment(CellIdx cell);

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
                              CellIdx presynapticCell,
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
        std::vector<Segment> segmentsForCell(CellIdx cell) const;

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
         * Do a reverse-lookup of a segment from its flatIdx.
         *
         * @param flatIdx the flatIdx of the segment
         *
         * @retval Segment
         */
        Segment segmentForFlatIdx(UInt32 flatIdx) const;

        /**
         * Returns the synapses for the source cell that they synapse on.
         *
         * @param presynapticCell(int) Source cell index
         *
         * @return (set)Synapse indices
         */
        std::vector<Synapse> synapsesForPresynapticCell(CellIdx presynapticCell)
          const;

        /**
         * Compute the segment excitations for a vector of cells.
         *
         * @param input
         * Active cells in the input.
         *
         * @param activePermanenceThreshold
         * Minimum permanence for a synapse to contribute to an active segment
         *
         * @param activeSynapseThreshold
         * Minimum number of synapses to mark a segment as "active"
         *
         * @param matchingPermanenceThreshold
         * Minimum permanence for a synapse to contribute to an matching segment
         *
         * @param matchingSynapseThreshold
         * Minimum number of synapses to mark a segment as "matching"
         *
         * @param outActiveSegments
         * An output vector.
         * On return, filled with active segments and overlaps.
         *
         * @param outActiveSegments
         * An output vector.
         * On return, filled with matching segments and overlaps.
         */
        void computeActivity(const std::vector<CellIdx>& input,
                             Permanence activePermanenceThreshold,
                             SynapseIdx activeSynapseThreshold,
                             Permanence matchingPermanenceThreshold,
                             SynapseIdx matchingSynapseThreshold,
                             std::vector<SegmentOverlap>& activeSegmentsOut,
                             std::vector<SegmentOverlap>& matchingSegmentsOut)
          const;

        /**
         * Record the fact that a segment had some activity. This information is
         * used during segment cleanup.
         *
         * @param segment
         * The segment that had some activity.
         */
        void recordSegmentActivity(Segment segment);

        /**
         * Mark the passage of time. This information is used during segment
         * cleanup.
         */
        void startNewIteration();

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
         * Gets the number of cells.
         *
         * @retval Number of cells.
         */
        CellIdx numCells() const;

        /**
         * Gets the number of segments.
         *
         * @retval Number of segments.
         */
        UInt numSegments() const;

        /**
         * Gets the number of segments on a cell.
         *
         * @retval Number of segments.
         */
        UInt numSegments(CellIdx cell) const;

        /**
         * Gets the number of synapses.
         *
         * @retval Number of synapses.
         */
        UInt numSynapses() const;

        /**
         * Gets the number of synapses on a segment.
         *
         * @retval Number of synapses.
         */
        UInt numSynapses(const Segment& segment) const;

        /**
         * Comparison operator.
         */
        bool operator==(const Connections &other) const;

        /**
         * Add a connections events handler.
         *
         * The Connections instance takes ownership of the eventHandlers
         * object. Don't delete it. When calling from Python, call
         * eventHandlers.__disown__() to avoid garbage-collecting the object
         * while this instance is still using it. It will be deleted on
         * `unsubscribe`.
         *
         * @param handler
         * An object implementing the ConnectionsEventHandler interface
         *
         * @retval Unsubscribe token
         */
        UInt32 subscribe(ConnectionsEventHandler* handler);

        /**
         * Remove an event handler.
         *
         * @param token
         * The return value of `subscribe`.
         */
        void unsubscribe(UInt32 token);

      protected:

        /**
         * Gets the segment that was least recently used from among all the
         * segments on the given cell.
         *
         * @param cell Cell whose segments to consider.
         *
         * @retval The least recently used segment.
         */
        Segment leastRecentlyUsedSegment_(CellIdx cell) const;

         /**
          * Gets the synapse with the lowest permanence on the segment.
          *
          * @param segment Segment whose synapses to consider.
          *
          * @retval Synapse with the lowest permanence.
          */
        Synapse minPermanenceSynapse_(const Segment& segment) const;

        /**
         * Gets a reference to the data for a segment.
         *
         * @param segment Segment to get data for.
         *
         * @retval Editable segment data.
         */
        SegmentData& dataForSegment_(const Segment& segment);

        /**
         * Gets a reference to the data for a segment.
         *
         * @param segment Segment to get data for.
         *
         * @retval Read-only segment data.
         */
        const SegmentData& dataForSegment_(const Segment& segment) const;

        /**
         * Gets a reference to the data for a synapse.
         *
         * @param synapse Synapse to get data for.
         *
         * @retval Editable synapse data.
         */
        SynapseData& dataForSynapse_(const Synapse& synapse);

        /**
         * Gets a reference to the data for a synapse.
         *
         * @param synapse Synapse to get data for.
         *
         * @retval Read-only synapse data.
         */
        const SynapseData& dataForSynapse_(const Synapse& synapse) const;

      private:
        std::vector<CellData> cells_;
        // Mapping (presynaptic cell => synapses) used in forward propagation
        std::map<CellIdx, std::vector<Synapse> > synapsesForPresynapticCell_;
        UInt numSegments_;
        UInt numSynapses_;
        std::vector<Segment> segmentForFlatIdx_;
        UInt nextFlatIdx_;
        SegmentIdx maxSegmentsPerCell_;
        SynapseIdx maxSynapsesPerSegment_;
        Iteration iteration_;
        UInt32 nextEventToken_;
        std::map<UInt32, ConnectionsEventHandler*> eventHandlers_;
      }; // end class Connections

      /**
       * Takes a Connections instance and a set of active cells, and calculates
       * the excited columns.
       *
       * This is essentially part of the Connections class. It accesses
       * Connections internals to perform this calculation quickly.
       *
       * This is implemented as a class rather than a Connections method because
       * this allows callers to provide the "active presynaptic cells" without
       * imposing requirements on how these cells are stored. The cells might be
       * stored in a vector, in multiple vectors, as a list of indices, as a
       * list of zeros and ones, etc., so we expose a "Tally" class and require
       * the caller to do the iteration.
       */
      class SegmentExcitationTally
      {
      public:
        SegmentExcitationTally(const Connections& connections,
                               Permanence activePermanenceThreshold,
                               Permanence matchingPermanenceThreshold);
        void addActivePresynapticCell(CellIdx cellIdx);
        void getResults(SynapseIdx activeSynapseThreshold,
                        SynapseIdx matchingSynapseThreshold,
                        std::vector<SegmentOverlap>& activeSegmentsOut,
                        std::vector<SegmentOverlap>& matchingSegmentsOut)
          const;
      private:
        const Connections& connections_;
        const Permanence activePermanenceThreshold_;
        const Permanence matchingPermanenceThreshold_;
        std::vector<UInt32> numActiveSynapsesForSegment_;
        std::vector<UInt32> numMatchingSynapsesForSegment_;
      };

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nupic

#endif // NTA_CONNECTIONS_HPP
