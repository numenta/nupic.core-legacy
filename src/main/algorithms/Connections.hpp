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
#include <nta/types/Types.hpp>
#include <nta/math/Math.hpp>

namespace nta
{

  namespace algorithms
  {

    namespace connections
    {

      /**
       * CellActivity class used in Connections.
       *
       * @b Description
       * The CellActivity class is a data structure that represents the
       * activity of a collection of cells, as computed by propagating
       * input through connections.
       * 
       */
      struct CellActivity
      {
      };

      // Forward declaration
      struct Synapse;

      /**
       * Segment class used in Connections.
       *
       * @b Description
       * The Segment class is a data structure that represents a segment
       * on a cell.
       *
       * @param cell     Index of cell this segment belongs to.
       * @param synapses List of synapses that this segment contains.
       * 
       */
      struct Segment
      {
        UInt cell;
        std::vector<Synapse*> synapses;
      };

      /**
       * Synapse class used in Connections.
       *
       * @b Description
       * The Synapse class is a data structure that represents a synapse
       * on a segment.
       *
       * @param segment         Segment that this synapse belongs to.
       * @param presynapticCell Cell that this synapse gets input from.
       * @param permanence      Permanence of synapse.
       * 
       */
      struct Synapse
      {
        Segment* segment;
        UInt presynapticCell;
        Real permanence;
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
        Connections();

        virtual ~Connections() {}

        /**
         Creates a segment on the specified cell.

         @param cell    Index of cell to create segment on.
         @param segment Segment to return.
        */
        void createSegment(UInt cell, Segment& segment);

        /**
         Creates a synapse on the specified segment.

         @param segment         Segment to create synapse on.
         @param presynapticCell Cell to synapse on.
         @param permanence      Initial permanence of new synapse.
         @param synapse         Synapse to return.
        */
        void createSynapse(Segment& segment,
                           UInt presynapticCell,
                           Real permanence,
                           Synapse &synapse);

        /**
         Updates a synapse's permanence.

         @param synapse    Synapse to update.
         @param permanence New permanence.
        */
        void updateSynapsePermanence(Synapse& synapse, Real permanence);

        /**
         Forward-propagates input to synapses, dendrites, and cells, to
         compute their activity.

         @param input Indices of active bits in the input.

         @retval CellActivity due to connected synapses.
        */
        void computeActivity(std::vector<UInt> input,
                             Real permanenceThreshold,
                             UInt synapseThreshold,
                             CellActivity& activity);

      }; // end class Connections

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nta

#endif // NTA_CONNECTIONS_HPP
