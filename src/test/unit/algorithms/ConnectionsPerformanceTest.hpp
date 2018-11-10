/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

namespace testing {

using nupic::algorithms::temporal_memory::TemporalMemory; 	

namespace nupic {
  namespace algorithms {
    namespace connections {
      typedef UInt32 Segment;
} } }

class ConnectionsPerformanceTest {
public:
  ConnectionsPerformanceTest() {}
  virtual ~ConnectionsPerformanceTest() {}
  float runTemporalMemoryTest(UInt numColumns, UInt w, int numSequences,
                             int numElements, std::string label);
  float runSpatialPoolerTest(UInt numCells, UInt numInputs, UInt w,
                            UInt numWinners, std::string label);

private:
  std::vector<UInt32> randomSDR(UInt n, UInt w);
  void feedTM(algorithms::temporal_memory::TemporalMemory &tm,
              std::vector<CellIdx> sdr, bool learn = true);
  std::vector<CellIdx> computeSPWinnerCells(Connections &connections, UInt numCells,
                       const vector<UInt> &numActiveSynapsesForSegment);

}; // end class ConnectionsPerformanceTest

} // end namespace testing

#endif // NTA_CONNECTIONS_PERFORMANCE_TEST
