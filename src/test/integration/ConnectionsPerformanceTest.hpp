/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

/** @file
 * Definitions for ConnectionsPerformanceTest
 */

//----------------------------------------------------------------------

#ifndef NTA_CONNECTIONS_PERFORMANCE_TEST
#define NTA_CONNECTIONS_PERFORMANCE_TEST

#include <nupic/algorithms/Connections.hpp>

using namespace nupic::algorithms::connections;

namespace nupic
{

  class ConnectionsPerformanceTest
  {
  public:
    ConnectionsPerformanceTest() {}
    virtual ~ConnectionsPerformanceTest() {}

    // Run all appropriate tests
    virtual void RunTests();

  private:
    void testTemporalMemoryUsage();

    void setupSampleConnections(Connections &connections);
    Activity computeSampleActivity(Connections &connections);

  }; // end class ConnectionsPerformanceTest

} // end namespace nupic

#endif // NTA_CONNECTIONS_PERFORMANCE_TEST
