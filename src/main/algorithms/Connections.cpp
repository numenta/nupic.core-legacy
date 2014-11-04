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
 * Implementation of Connections
 */

#include <iostream>
#include <nta/algorithms/Connections.hpp>

using namespace std;
using namespace nta;
using namespace nta::algorithms::connections;

Connections::Connections()
{
}

Segment Connections::createSegment(UInt cell)
{
  vector<Synapse*> synapses;
  Segment segment = {cell, synapses};

  return segment;
}

Synapse Connections::createSynapse(Segment& segment,
                                   UInt presynapticCell,
                                   Real permanence)
{
  Synapse synapse = {segment, presynapticCell, permanence};
  segment.synapses.push_back(&synapse);

  return synapse;
}

CellActivity Connections::computeActivity(vector<UInt> input,
                                          Real permanenceThreshold,
                                          UInt synapseThreshold)
{
  CellActivity connectedActivity;
  return connectedActivity;
}
