/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
 * Definitions for the Connections class in C++
 */

#ifndef NTA_CONNECTIONS_HPP
#define NTA_CONNECTIONS_HPP

#include <vector>
#include <utility>
#include <nta/types/Types.hpp>
#include <nta/math/Math.hpp>

namespace nta
{

  namespace algorithms
  {

    namespace connections
    {
      typedef UInt32 CellIdx;
      typedef Byte SegmentIdx;
      typedef Byte SynapseIdx;
      typedef Real32 Permanence;

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

        Segment(SegmentIdx idx, Cell cell) : idx(idx), cell(cell) {}
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
       * 
       */
      struct SynapseData
      {
        Cell presynapticCell;
        Permanence permanence;
      };

      /**
       * SegmentData class used in Connections.
       *
       * @b Description
       * The SegmentData class is a data structure that contains the data for a
       * segment on a cell.
       *
       * @param synapses Data for synapses that this segment contains.
       * 
       */
      struct SegmentData
      {
        std::vector<SynapseData> synapses;
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
        std::map<Segment, UInt> numActiveSynapsesForSegment;
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
      class Connections
      {
      public:
        Connections(CellIdx numCells);

        virtual ~Connections() {}

        /**
         Creates a segment on the specified cell.

         @param cell Cell to create segment on.

         @retval Created segment.
        */
        Segment createSegment(const Cell& cell);

        /**
         Creates a synapse on the specified segment.

         @param segment         Segment to create synapse on.
         @param presynapticCell Cell to synapse on.
         @param permanence      Initial permanence of new synapse.

         @reval Created synapse.
        */
        Synapse createSynapse(const Segment& segment,
                              const Cell& presynapticCell,
                              Permanence permanence);

        /**
         Updates a synapse's permanence.

         @param synapse    Synapse to update.
         @param permanence New permanence.
        */
        void updateSynapsePermanence(const Synapse& synapse,
                                     Permanence permanence);

        /**
         Gets the segments for a cell.

         @param cell Cell to get segments for.

         @retval Segments on cell.
        */
        std::vector<Segment> segmentsForCell(const Cell& cell);

        /**
         Gets the synapses for a segment.

         @param segment Segment to get synapses for.

         @retval Synapses on segment.
        */
        std::vector<Synapse> synapsesForSegment(const Segment& segment);

        /**
         Gets the data for a synapse.

         @param synapse Synapse to get data for.

         @retval Synapse data.
        */
        SynapseData dataForSynapse(const Synapse& synapse) const;

        /**
         Gets the segment with the most active synapses due to given input,
         from among all the segments on all the given cells.

         @param cells            Cells to look among.
         @param input            Active cells in the input.
         @param synapseThreshold Only consider segments with number of active synapses greater than this threshold.
         @param segment          Segment to return.

         @retval Segment found?
        */
        bool mostActiveSegmentForCells(const std::vector<Cell>& cells,
                                       std::vector<Cell> input,
                                       UInt synapseThreshold,
                                       Segment& retSegment) const;

        /**
         Forward-propagates input to synapses, dendrites, and cells, to
         compute their activity.

         @param input               Active cells in the input.
         @param permanenceThreshold Only consider synapses with permanences greater than this threshold.
         @param synapseThreshold    Only consider segments with number of active synapses greater than this threshold.

         @retval Activity to return.
        */
        Activity computeActivity(const std::vector<Cell>& input,
                                 Permanence permanenceThreshold,
                                 UInt synapseThreshold) const;

        /**
         Gets the active segments from activity.

         @param activity Activity.

         @retval Active segments.
        */
        std::vector<Segment> activeSegments(const Activity& activity);

        /**
         Gets the active cells from activity.

         @param activity Activity.

         @retval Active cells.
        */
        std::vector<Cell> activeCells(const Activity& activity);

      private:
        std::vector<CellData> cells_;
        // Mapping (presynaptic cell => synapses) used in forward propagation
        std::map< Cell, std::vector<Synapse> > synapsesForPresynapticCell_;
      }; // end class Connections

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nta

#endif // NTA_CONNECTIONS_HPP
