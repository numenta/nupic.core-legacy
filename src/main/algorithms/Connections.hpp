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

         @param cell Index of cell to create segment on.

         @retval Segment index.
        */
        UInt createSegment(UInt cell);

        /**
         Forward-propagates input to synapses, dendrites, and cells, to
         compute their activity.

         @param input Indices of active bits in the input.

         @retval CellActivity due to connected synapses.
        */
        CellActivity computeActivity(std::vector<UInt> input,
                                     Real permanenceThreshold,
                                     UInt synapseThreshold);

      }; // end class Connections

    } // end namespace connections

  } // end namespace algorithms

} // end namespace nta

#endif // NTA_CONNECTIONS_HPP
