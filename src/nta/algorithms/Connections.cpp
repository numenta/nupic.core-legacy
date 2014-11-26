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

#include <climits>
#include <nupic/algorithms/Connections.hpp>

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::connections;

Connections::Connections(CellIdx numCells) : cells_(numCells) {
  numSegments_ = 0;
  numSynapses_ = 0;
}

Segment Connections::createSegment(const Cell& cell)
{
  vector<SegmentData>& segments = cells_[cell.idx].segments;
  if (segments.size() == UCHAR_MAX)
  {
    throw runtime_error("Cannot create segment: cell has reached maximum number of segments.");
  }
  Segment segment(segments.size(), cell);

  SegmentData segmentData;
  segments.push_back(segmentData);
  numSegments_++;

  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   const Cell& presynapticCell,
                                   Permanence permanence)
{
  vector<SynapseData>& synapses = cells_[segment.cell.idx].segments[segment.idx].synapses;
  if (synapses.size() == UCHAR_MAX)
  {
    throw runtime_error("Cannot create synapse: segment has reached maximum number of synapses.");
  }
  Synapse synapse(synapses.size(), segment);

  SynapseData synapseData = {presynapticCell, permanence};
  synapses.push_back(synapseData);
  numSynapses_++;

  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);

  return synapse;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  cells_[cell.idx].segments[segment.idx].synapses[synapse.idx].permanence = permanence;
}

vector<Segment> Connections::segmentsForCell(const Cell& cell)
{
  vector<Segment> segments;
  Segment segment;

  for (SegmentIdx i = 0; i < cells_[cell.idx].segments.size(); i++)
  {
    segment.idx = i;
    segment.cell = cell;
    segments.push_back(segment);
  }

  return segments;
}

vector<Synapse> Connections::synapsesForSegment(const Segment& segment)
{
  const Cell& cell = segment.cell;
  vector<Synapse> synapses;
  Synapse synapse;

  for (SynapseIdx i = 0; i < cells_[cell.idx].segments[segment.idx].synapses.size(); i++)
  {
    synapse.idx = i;
    synapse.segment = segment;
    synapses.push_back(synapse);
  }

  return synapses;
}

SynapseData Connections::dataForSynapse(const Synapse& synapse) const
{
  const Segment& segment = synapse.segment;
  const Cell& cell = segment.cell;

  return cells_[cell.idx].segments[segment.idx].synapses[synapse.idx];
}

bool Connections::mostActiveSegmentForCells(const vector<Cell>& cells,
                                            vector<Cell> input,
                                            UInt synapseThreshold,
                                            Segment& retSegment) const
{
  UInt numSynapses, maxSynapses = synapseThreshold;
  vector<SegmentData> segments;
  vector<SynapseData> synapses;
  SegmentIdx segmentIdx = 0;
  bool found = false;

  sort(input.begin(), input.end());  // for binary search

  for (auto cell : cells)
  {
    segments = cells_[cell.idx].segments;
    segmentIdx = 0;

    for (auto segment : segments)
    {
      synapses = segment.synapses;
      numSynapses = 0;

      for (auto synapse : synapses)
      {
        if (binary_search(input.begin(), input.end(), synapse.presynapticCell))
        {
          numSynapses++;
        }
      }

      if (numSynapses >= maxSynapses)
      {
        maxSynapses = numSynapses;
        retSegment.idx = segmentIdx;
        retSegment.cell = cell;
        found = true;
      }

      segmentIdx++;
    }
  }

  return found;
}

Activity Connections::computeActivity(const vector<Cell>& input,
                                      Permanence permanenceThreshold,
                                      UInt synapseThreshold) const
{
  Activity activity;
  vector<Synapse> synapses;
  SynapseData synapseData;

  for (auto cell : input)
  {
    if (!synapsesForPresynapticCell_.count(cell)) continue;
    synapses = synapsesForPresynapticCell_.at(cell);

    for (auto synapse : synapses)
    {
      synapseData = dataForSynapse(synapse);

      if (synapseData.permanence >= permanenceThreshold)
      {
        activity.numActiveSynapsesForSegment[synapse.segment] += 1;

        if (activity.numActiveSynapsesForSegment[synapse.segment] == synapseThreshold)
        {
          activity.activeSegmentsForCell[synapse.segment.cell].push_back(synapse.segment);
        }
      }
    }
  }

  return activity;
}

vector<Segment> Connections::activeSegments(const Activity& activity)
{
  vector<Segment> segments;

  for (auto i : activity.activeSegmentsForCell)
  {
    segments.insert(segments.end(), i.second.begin(), i.second.end());
  }

  return segments;
}

vector<Cell> Connections::activeCells(const Activity& activity)
{
  vector<Cell> cells;

  for (auto i : activity.activeSegmentsForCell)
  {
    cells.push_back(i.first);
  }

  return cells;
}

UInt Connections::numSegments() const
{
  return numSegments_;
}

UInt Connections::numSynapses() const
{
  return numSynapses_;
}

bool Cell::operator==(const Cell &other) const
{
  return idx == other.idx;
}

bool Cell::operator<=(const Cell &other) const
{
  return idx <= other.idx;
}

bool Cell::operator<(const Cell &other) const
{
  return idx < other.idx;
}

bool Cell::operator>=(const Cell &other) const
{
  return idx >= other.idx;
}

bool Cell::operator>(const Cell &other) const
{
  return idx > other.idx;
}

bool Segment::operator==(const Segment &other) const
{
  return idx == other.idx && cell == other.cell;
}

bool Segment::operator<=(const Segment &other) const
{
  return idx == other.idx ? cell <= other.cell : idx <= other.idx;
}

bool Segment::operator<(const Segment &other) const
{
  return idx == other.idx ? cell < other.cell : idx < other.idx;
}

bool Segment::operator>=(const Segment &other) const
{
  return idx == other.idx ? cell >= other.cell : idx >= other.idx;
}

bool Segment::operator>(const Segment &other) const
{
  return idx == other.idx ? cell > other.cell : idx > other.idx;
}

bool Synapse::operator==(const Synapse &other) const
{
  return idx == other.idx && segment == other.segment;
}
