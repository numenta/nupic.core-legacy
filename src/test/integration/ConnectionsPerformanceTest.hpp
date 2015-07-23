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
#include <nupic/algorithms/TemporalMemory.hpp>

using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;

namespace nupic
{

  class ConnectionsPerformanceTest
  {
  public:
    ConnectionsPerformanceTest() {}
    virtual ~ConnectionsPerformanceTest() {}

    // Run all appropriate tests
    virtual void RunTests();

    void testTemporalMemoryUsage();
    void testLargeTemporalMemoryUsage();
    void testSpatialPoolerUsage();
    void testTemporalPoolerUsage();

  private:
    void runTemporalMemoryTest(UInt numColumns,
                               UInt w,
                               int numSequences,
                               int numElements,
                               string label);
    void runSpatialPoolerTest(UInt numCells,
                              UInt numInputs,
                              UInt w,
                              UInt numWinners,
                              string label);

    void checkpoint(clock_t timer, std::string text);
    std::vector<Cell> randomSDR(UInt n, UInt w);
    void feedTM(TemporalMemory &tm, vector<Cell> sdr, bool learn = true);
    std::vector<Cell> computeSPWinnerCells(UInt numCells, Activity& activity);

  }; // end class ConnectionsPerformanceTest

} // end namespace nupic

#endif // NTA_CONNECTIONS_PERFORMANCE_TEST
