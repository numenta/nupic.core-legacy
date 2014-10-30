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
 * Definitions for the Activity class in C++
 */

#ifndef NTA_ACTIVITY_HPP
#define NTA_ACTIVITY_HPP

#include <vector>
#include <nta/types/Types.hpp>

namespace nta
{

  namespace algorithms
  {

    namespace activity
    {

      /**
       * Activity implementation used in Connections class.
       *
       * @b Description
       * The Activity class is a data structure that represents the
       * activity of cells in the Connections class, after propagating
       * an input through the connections.
       * 
       */
      class Activity
      {
      public:
        Activity();

        virtual ~Activity() {}

      }; // end class Activity

    } // end namespace activity

  } // end namespace algorithms

} // end namespace nta

#endif // NTA_ACTIVITY_HPP
