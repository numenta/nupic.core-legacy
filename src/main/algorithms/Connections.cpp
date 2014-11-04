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

void Connections::createSegment(UInt cell, Segment& segment)
{
  vector<Synapse*> synapses;

  segment.cell = cell;
  segment.synapses = synapses;
}

void Connections::createSynapse(Segment& segment,
                                UInt presynapticCell,
                                Real permanence,
                                Synapse& synapse)
{
  synapse.segment = &segment;
  synapse.presynapticCell = presynapticCell;
  synapse.permanence = permanence;

  segment.synapses.push_back(&synapse);
}

void Connections::updateSynapsePermanence(Synapse& synapse, Real permanence)
{
  synapse.permanence = permanence;
}

bool Connections::getMostActiveSegmentForCells(std::vector<UInt> cells,
                                               std::vector<UInt> input,
                                               UInt synapseThreshold,
                                               Segment& segment)
{
  return false;
}

void Connections::computeActivity(vector<UInt> input,
                                  Real permanenceThreshold,
                                  UInt synapseThreshold,
                                  CellActivity& activity)
{
}
