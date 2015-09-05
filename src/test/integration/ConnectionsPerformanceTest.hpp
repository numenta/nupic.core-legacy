/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

/** @file
 * Definitions for ConnectionsPerformanceTest
 */

//----------------------------------------------------------------------

#ifndef NTA_CONNECTIONS_PERFORMANCE_TEST
#define NTA_CONNECTIONS_PERFORMANCE_TEST

#include <vector>

#include <nupic/types/Types.hpp>

namespace nupic
{

  namespace algorithms
  {
    namespace temporal_memory
    {
      class TemporalMemory;
    }

    namespace connections
    {
      struct Activity;
      struct Cell;
    }
  }

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
                               std::string label);
    void runSpatialPoolerTest(UInt numCells,
                              UInt numInputs,
                              UInt w,
                              UInt numWinners,
                              std::string label);

    void checkpoint(clock_t timer, std::string text);
    std::vector<algorithms::connections::Cell> randomSDR(UInt n, UInt w);
    void feedTM(algorithms::temporal_memory::TemporalMemory &tm,
                std::vector<algorithms::connections::Cell> sdr,
                bool learn = true);
    std::vector<algorithms::connections::Cell> computeSPWinnerCells(
      UInt numCells, algorithms::connections::Activity& activity);

  }; // end class ConnectionsPerformanceTest

} // end namespace nupic

#endif // NTA_CONNECTIONS_PERFORMANCE_TEST
