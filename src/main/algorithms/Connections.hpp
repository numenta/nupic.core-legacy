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
        SegmentIdx segmentIdx;
        CellIdx cellIdx;
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
        CellIdx cellIdx;
      };

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
        CellIdx presynapticCellIdx;
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
        std::map<Cell, UInt> numActiveSegmentsForCell;
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
       */
      class Connections
      {
      public:
        Connections(CellIdx numCells);

        virtual ~Connections() {}

      private:
        std::vector<CellData> cells_;
        std::map< Cell, std::vector<Synapse> > synapsesForPresynapticCell_;
      }; // end class Connections

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nta

#endif // NTA_CONNECTIONS_HPP
