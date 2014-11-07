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
 * ---------------------------------------------------------------------
 */

/** @file
 * Definitions for ConnectionsTest
 */

//----------------------------------------------------------------------

#ifndef NTA_CONNECTIONS_TEST
#define NTA_CONNECTIONS_TEST

#include <nta/test/Tester.hpp>
#include <nta/algorithms/Connections.hpp>

using namespace nta::algorithms::connections;

namespace nta
{

  class ConnectionsTest : public Tester
  {
  public:
    ConnectionsTest() {}
    virtual ~ConnectionsTest() {}

    // Run all appropriate tests
    virtual void RunTests();

  private:
    void testCreateSegment();
    void testCreateSynapse();
    void testUpdateSynapsePermanence();
    void testMostActiveSegmentForCells();
    void testMostActiveSegmentForCellsNone();
    void testComputeActivity();
    void testActiveSegments();
    void testActiveCells();

    void setupSampleConnections(Connections &connections);
    Activity computeSampleActivity(Connections &connections);

  }; // end class ConnectionsTest

} // end namespace nta

#endif // NTA_CONNECTIONS_TEST
