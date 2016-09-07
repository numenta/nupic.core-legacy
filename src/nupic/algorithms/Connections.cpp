/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
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
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of Connections
 */

#include <climits>
#include <iostream>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/std/iostream.h>

#include <nupic/algorithms/Connections.hpp>


using std::vector;
using std::string;
using std::endl;
using namespace nupic;
using namespace nupic::algorithms::connections;

static const Permanence EPSILON = 0.00001;

Connections::Connections(CellIdx numCells,
                         SegmentIdx maxSegmentsPerCell,
                         SynapseIdx maxSynapsesPerSegment)
{
  initialize(numCells, maxSegmentsPerCell, maxSynapsesPerSegment);
}

void Connections::initialize(CellIdx numCells,
                             SegmentIdx maxSegmentsPerCell,
                             SynapseIdx maxSynapsesPerSegment)
{
  cells_ = vector<CellData>(numCells);
  numSegments_ = 0;
  numSynapses_ = 0;
  nextFlatIdx_ = 0;
  maxSegmentsPerCell_ = maxSegmentsPerCell;
  maxSynapsesPerSegment_ = maxSynapsesPerSegment;
  iteration_ = 0;
  nextEventToken_ = 0;
}

UInt32 Connections::subscribe(ConnectionsEventHandler* handler)
{
  UInt32 token = nextEventToken_++;
  eventHandlers_[token] = handler;
  return token;
}

void Connections::unsubscribe(UInt32 token)
{
  delete eventHandlers_.at(token);
  eventHandlers_.erase(token);
}

Segment Connections::createSegment(CellIdx cell)
{
  NTA_CHECK(maxSegmentsPerCell_ > 0);
  while (numSegments(cell) >= maxSegmentsPerCell_)
  {
    destroySegment(leastRecentlyUsedSegment_(cell));
  }

  CellData& cellData = cells_[cell];
  Segment segment = {cell, (SegmentIdx)-1, (UInt32)-1};
  if (cellData.numDestroyedSegments > 0)
  {
    bool found = false;
    for (SegmentIdx i = 0; i < cellData.segments.size(); i++)
    {
      if (cellData.segments[i].destroyed)
      {
        segment.idx = i;
        found = true;
      }
    }

    NTA_CHECK(found);

    cellData.segments[segment.idx].destroyed = false;
    cellData.numDestroyedSegments--;
  }
  else
  {
    const UInt32 flatIdx = nextFlatIdx_++;
    segment.idx = cellData.segments.size();
    segment.flatIdx = flatIdx;
    segmentForFlatIdx_.push_back(segment);

    cellData.segments.push_back(SegmentData());
    cellData.segments[segment.idx].flatIdx = flatIdx;
  }

  cellData.segments[segment.idx].lastUsedIteration = iteration_;
  numSegments_++;

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSegment(segment);
  }

  return segment;
}

Synapse Connections::createSynapse(const Segment& segment,
                                   CellIdx presynapticCell,
                                   Permanence permanence)
{
  NTA_CHECK(maxSynapsesPerSegment_ > 0);
  NTA_CHECK(permanence > 0);
  while (numSynapses(segment) >= maxSynapsesPerSegment_)
  {
    destroySynapse(minPermanenceSynapse_(segment));
  }

  SegmentData& segmentData = dataForSegment_(segment);
  SynapseIdx synapseIdx = (SynapseIdx)-1;
  if (segmentData.numDestroyedSynapses > 0)
  {
    bool found = false;
    for(SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
    {
      if (segmentData.synapses[i].destroyed)
      {
        synapseIdx = i;
        found = true;
        break;
      }
    }

    NTA_CHECK(found);

    segmentData.synapses[synapseIdx].destroyed = false;
    segmentData.numDestroyedSynapses--;
  }
  else
  {
    synapseIdx = segmentData.synapses.size();
    segmentData.synapses.push_back(SynapseData());
  }

  SynapseData& synapseData = segmentData.synapses[synapseIdx];
  synapseData.presynapticCell = presynapticCell;
  synapseData.permanence = permanence;

  Synapse synapse = {synapseIdx, segment};
  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
  numSynapses_++;

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSynapse(synapse);
  }

  return synapse;
}

void Connections::destroySegment(const Segment& segment)
{
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySegment(segment);
  }

  SegmentData& segmentData = dataForSegment_(segment);

  NTA_CHECK(!segmentData.destroyed) << "Segment already destroyed.";

  for (SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
  {
    Synapse synapse = {i, segment};
    const SynapseData& synapseData = dataForSynapse_(synapse);

    if (!synapseData.destroyed)
    {
      vector<Synapse>& presynapticSynapses =
        synapsesForPresynapticCell_.at(synapseData.presynapticCell);

      auto it = std::find(presynapticSynapses.begin(), presynapticSynapses.end(),
                          synapse);
      NTA_CHECK(it != presynapticSynapses.end());
      presynapticSynapses.erase(it);

      if (presynapticSynapses.size() == 0)
      {
        synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
      }
      numSynapses_--;
    }
  }
  segmentData.synapses.clear();
  segmentData.numDestroyedSynapses = 0;

  segmentData.destroyed = true;
  cells_[segment.cell].numDestroyedSegments++;
  numSegments_--;
}

void Connections::destroySynapse(const Synapse& synapse)
{
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySynapse(synapse);
  }

  SynapseData& synapseData = dataForSynapse_(synapse);

  NTA_CHECK(!synapseData.destroyed) << "Synapse already destroyed.";

  vector<Synapse>& presynapticSynapses =
    synapsesForPresynapticCell_.at(synapseData.presynapticCell);

  auto it = std::find(presynapticSynapses.begin(), presynapticSynapses.end(),
                      synapse);
  NTA_CHECK(it != presynapticSynapses.end());
  presynapticSynapses.erase(it);

  if (presynapticSynapses.size() == 0)
  {
    synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
  }

  synapseData.destroyed = true;
  dataForSegment_(synapse.segment).numDestroyedSynapses++;
  numSynapses_--;
}

void Connections::updateSynapsePermanence(const Synapse& synapse,
                                          Permanence permanence)
{
  for (auto h : eventHandlers_)
  {
    h.second->onUpdateSynapsePermanence(synapse, permanence);
  }

  dataForSynapse_(synapse).permanence = permanence;
}

vector<Segment> Connections::segmentsForCell(CellIdx cell) const
{
  vector<Segment> segments;
  segments.reserve(numSegments(cell));

  const vector<SegmentData>& allSegments = cells_[cell].segments;

  for (SegmentIdx i = 0; i < (SegmentIdx)allSegments.size(); i++)
  {
    if (!allSegments[i].destroyed)
    {
      segments.push_back({cell, i, allSegments[i].flatIdx});
    }
  }

  return segments;
}

Segment Connections::getSegment(CellIdx cell, SegmentIdx idx) const
{
  const UInt32 flatIdx = cells_[cell].segments[idx].flatIdx;
  return {cell, idx, flatIdx};
}

vector<Synapse> Connections::synapsesForSegment(const Segment& segment) const
{
  vector<Synapse> synapses;
  synapses.reserve(numSynapses(segment));

  const SegmentData& segmentData = dataForSegment_(segment);
  if (segmentData.destroyed)
  {
    NTA_THROW << "Attempting to access destroyed segment's synapses.";
  }

  for (SynapseIdx i = 0; i < segmentData.synapses.size(); i++)
  {
    if (!segmentData.synapses[i].destroyed)
    {
      synapses.push_back({i, segment});
    }
  }

  return synapses;
}

SegmentData Connections::dataForSegment(const Segment& segment) const
{
  return cells_[segment.cell].segments[segment.idx];
}

SegmentData& Connections::dataForSegment_(const Segment& segment)
{
  return cells_[segment.cell].segments[segment.idx];
}

const SegmentData& Connections::dataForSegment_(const Segment& segment) const
{
  return cells_[segment.cell].segments[segment.idx];
}

SynapseData Connections::dataForSynapse(const Synapse& synapse) const
{
  return cells_[synapse.segment.cell]
    .segments[synapse.segment.idx]
    .synapses[synapse.idx];
}

SynapseData& Connections::dataForSynapse_(const Synapse& synapse)
{
  SegmentData& segmentData = dataForSegment_(synapse.segment);
  return segmentData.synapses[synapse.idx];
}

const SynapseData& Connections::dataForSynapse_(const Synapse& synapse) const
{
  const SegmentData& segmentData = dataForSegment_(synapse.segment);
  return segmentData.synapses[synapse.idx];
}

Segment Connections::segmentForFlatIdx(UInt32 flatIdx) const
{
  return segmentForFlatIdx_[flatIdx];
}

UInt32 Connections::segmentFlatListLength() const
{
  return nextFlatIdx_;
}

vector<Synapse> Connections::synapsesForPresynapticCell(
  CellIdx presynapticCell) const
{
  if (synapsesForPresynapticCell_.find(presynapticCell) ==
      synapsesForPresynapticCell_.end())
    return vector<Synapse>{};

  return synapsesForPresynapticCell_.at(presynapticCell);
}

Segment Connections::leastRecentlyUsedSegment_(CellIdx cell) const
{
  const vector<SegmentData>& segments = cells_[cell].segments;
  bool found = false;
  SegmentIdx minIdx;
  Iteration minIteration;
  for (SegmentIdx i = 0; i < segments.size(); i++)
  {
    if (!segments[i].destroyed && (!found ||
                                   segments[i].lastUsedIteration < minIteration))
    {
      minIdx = i;
      minIteration = segments[i].lastUsedIteration;
      found = true;
    }
  }

  NTA_CHECK(found);

  return Segment(cell, minIdx, segments[minIdx].flatIdx);
}

Synapse Connections::minPermanenceSynapse_(const Segment& segment) const
{
  const vector<SynapseData>& synapses =
    cells_[segment.cell].segments[segment.idx].synapses;
  bool found = false;
  SynapseIdx minIdx;
  Permanence minPermanence;
  for (SynapseIdx i = 0; i < synapses.size(); i++)
  {
    if(!synapses[i].destroyed &&
       (!found || synapses[i].permanence < minPermanence - EPSILON))
    {
      minIdx = i;
      minPermanence = synapses[i].permanence;
      found = true;
    }
  }

  NTA_CHECK(found);

  return Synapse(minIdx, segment);
}

void Connections::computeActivity(
  vector<UInt32>& numActiveConnectedSynapsesForSegment,
  vector<UInt32>& numActivePotentialSynapsesForSegment,
  CellIdx activePresynapticCell,
  Permanence connectedPermanence) const
{
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == nextFlatIdx_);
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == nextFlatIdx_);

  if (synapsesForPresynapticCell_.count(activePresynapticCell))
  {
    for (const Synapse& synapse :
           synapsesForPresynapticCell_.at(activePresynapticCell))
    {
      ++numActivePotentialSynapsesForSegment[synapse.segment.flatIdx];

      const SynapseData& synapseData = dataForSynapse_(synapse);
      NTA_ASSERT(synapseData.permanence > 0);
      if (synapseData.permanence >= connectedPermanence - EPSILON)
      {
        ++numActiveConnectedSynapsesForSegment[synapse.segment.flatIdx];
      }
    }
  }
}

void Connections::computeActivity(
  vector<UInt32>& numActiveConnectedSynapsesForSegment,
  vector<UInt32>& numActivePotentialSynapsesForSegment,
  const vector<CellIdx>& activePresynapticCells,
  Permanence connectedPermanence) const
{
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == nextFlatIdx_);
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == nextFlatIdx_);

  for (CellIdx cell : activePresynapticCells)
  {
    if (synapsesForPresynapticCell_.count(cell))
    {
      for (const Synapse& synapse : synapsesForPresynapticCell_.at(cell))
      {
        ++numActivePotentialSynapsesForSegment[synapse.segment.flatIdx];

        const SynapseData& synapseData = dataForSynapse_(synapse);
        NTA_ASSERT(synapseData.permanence > 0);
        if (synapseData.permanence >= connectedPermanence - EPSILON)
        {
          ++numActiveConnectedSynapsesForSegment[synapse.segment.flatIdx];
        }
      }
    }
  }
}

void Connections::recordSegmentActivity(Segment segment)
{
  dataForSegment_(segment).lastUsedIteration = iteration_;
}

void Connections::startNewIteration()
{
  iteration_++;
}

void Connections::save(std::ostream& outStream) const
{
  // Write a starting marker.
  outStream << "Connections" << endl;
  outStream << Connections::VERSION << endl;

  outStream << cells_.size() << " "
            << maxSegmentsPerCell_ << " "
            << maxSynapsesPerSegment_ << " "
            << endl;

  for (CellData cellData : cells_)
  {
    auto segments = cellData.segments;
    outStream << segments.size() << " ";

    for (SegmentData segment : segments)
    {
      outStream << segment.destroyed << " ";
      outStream << segment.lastUsedIteration << " ";

      auto synapses = segment.synapses;
      outStream << synapses.size() << " ";

      for (SynapseData synapse : synapses)
      {
        outStream << synapse.presynapticCell << " ";
        outStream << synapse.permanence << " ";
        outStream << synapse.destroyed << " ";
      }
      outStream << endl;
    }
    outStream << endl;
  }
  outStream << endl;

  outStream << iteration_ << " " << endl;

  outStream << "~Connections" << endl;
}

void Connections::write(ConnectionsProto::Builder& proto) const
{
  proto.setVersion(Connections::VERSION);

  auto protoCells = proto.initCells(cells_.size());

  for (CellIdx i = 0; i < cells_.size(); ++i)
  {
    auto segments = cells_[i].segments;
    auto protoSegments = protoCells[i].initSegments(segments.size());

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j)
    {
      auto synapses = segments[j].synapses;
      auto protoSynapses = protoSegments[j].initSynapses(synapses.size());
      protoSegments[j].setDestroyed(segments[j].destroyed);
      protoSegments[j].setLastUsedIteration(segments[j].lastUsedIteration);

      for (SynapseIdx k = 0; k < synapses.size(); ++k)
      {
        protoSynapses[k].setPresynapticCell(synapses[k].presynapticCell);
        protoSynapses[k].setPermanence(synapses[k].permanence);
        protoSynapses[k].setDestroyed(synapses[k].destroyed);
      }
    }
  }

  proto.setMaxSegmentsPerCell(maxSegmentsPerCell_);
  proto.setMaxSynapsesPerSegment(maxSynapsesPerSegment_);
  proto.setIteration(iteration_);
}

void Connections::load(std::istream& inStream)
{
  // Check the marker
  string marker;
  inStream >> marker;
  NTA_CHECK(marker == "Connections");

  // Check the saved version.
  UInt version;
  inStream >> version;
  NTA_CHECK(version <= Connections::VERSION);

  // Retrieve simple variables
  UInt numCells;
  inStream >> numCells
           >> maxSegmentsPerCell_
           >> maxSynapsesPerSegment_;

  initialize(numCells, maxSegmentsPerCell_, maxSynapsesPerSegment_);

  cells_.resize(numCells);
  for (UInt cell = 0; cell < numCells; cell++)
  {
    CellData& cellData = cells_[cell];

    UInt numSegments;
    inStream >> numSegments;

    cellData.segments.resize(numSegments);
    for (SegmentIdx j = 0; j < numSegments; j++)
    {
      inStream >> cellData.segments[j].destroyed;
      inStream >> cellData.segments[j].lastUsedIteration;
      const UInt32 flatIdx = nextFlatIdx_++;
      cellData.segments[j].flatIdx = flatIdx;
      const Segment segment = {cell, j, flatIdx};
      segmentForFlatIdx_.push_back(segment);

      UInt numSynapses;
      inStream >> numSynapses;

      auto& synapses = cellData.segments[j].synapses;
      synapses.resize(numSynapses);
      for (UInt k = 0; k < numSynapses; k++)
      {
        inStream >> synapses[k].presynapticCell;
        inStream >> synapses[k].permanence;
        inStream >> synapses[k].destroyed;

        if (synapses[k].destroyed)
        {
          cellData.segments[j].numDestroyedSynapses++;
        }
        else
        {
          numSynapses_++;

          Synapse synapse = Synapse(k, segment);
          synapsesForPresynapticCell_[synapses[k].presynapticCell].push_back(synapse);
        }
      }

      if (cellData.segments[j].destroyed)
      {
        cellData.numDestroyedSegments++;
      }
      else
      {
        numSegments_++;
      }
    }
  }

  inStream >> iteration_;

  inStream >> marker;
  NTA_CHECK(marker == "~Connections");
}

void Connections::read(ConnectionsProto::Reader& proto)
{
  // Check the saved version.
  UInt version = proto.getVersion();
  NTA_CHECK(version <= Connections::VERSION);

  auto protoCells = proto.getCells();

  initialize(protoCells.size(),
             proto.getMaxSegmentsPerCell(),
             proto.getMaxSynapsesPerSegment());

  for (CellIdx cell = 0; cell < protoCells.size(); ++cell)
  {
    auto protoSegments = protoCells[cell].getSegments();
    vector<SegmentData>& segments = cells_[cell].segments;

    for (SegmentIdx j = 0; j < (SegmentIdx)protoSegments.size(); ++j)
    {
      const SegmentData segmentData = {vector<SynapseData>(),
                                       0,
                                       protoSegments[j].getDestroyed(),
                                       protoSegments[j].getLastUsedIteration(),
                                       nextFlatIdx_++};
      segments.push_back(segmentData);
      const Segment segment = {cell, j, segmentData.flatIdx};
      segmentForFlatIdx_.push_back(segment);

      auto protoSynapses = protoSegments[j].getSynapses();
      vector<SynapseData>& synapses = segments[j].synapses;

      for (SynapseIdx k = 0; k < protoSynapses.size(); ++k)
      {
        CellIdx presynapticCell = protoSynapses[k].getPresynapticCell();
        SynapseData synapseData = {presynapticCell,
                                   protoSynapses[k].getPermanence(),
                                   protoSynapses[k].getDestroyed()};
        synapses.push_back(synapseData);

        if (synapseData.destroyed)
        {
          segments[j].numDestroyedSynapses++;
        }
        else
        {
          numSynapses_++;

          Synapse synapse = Synapse(k, segment);
          synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
        }
      }

      if (segmentData.destroyed)
      {
        cells_[cell].numDestroyedSegments++;
      }
      else
      {
        numSegments_++;
      }
    }
  }

  iteration_ = proto.getIteration();
}

CellIdx Connections::numCells() const
{
  return cells_.size();
}

UInt Connections::numSegments() const
{
  return numSegments_;
}

UInt Connections::numSegments(CellIdx cell) const
{
  const CellData& cellData = cells_[cell];
  return cellData.segments.size() - cellData.numDestroyedSegments;
}

UInt Connections::numSynapses() const
{
  return numSynapses_;
}

UInt Connections::numSynapses(const Segment& segment) const
{
  const SegmentData& segmentData = cells_[segment.cell].segments[segment.idx];
  return segmentData.synapses.size() - segmentData.numDestroyedSynapses;
}

bool Connections::operator==(const Connections &other) const
{
  if (maxSegmentsPerCell_ != other.maxSegmentsPerCell_) return false;
  if (maxSynapsesPerSegment_ != other.maxSynapsesPerSegment_) return false;

  if (cells_.size() != other.cells_.size()) return false;

  for (CellIdx i = 0; i < cells_.size(); ++i)
  {
    auto segments = cells_[i].segments;
    auto otherSegments = other.cells_[i].segments;

    if (segments.size() != otherSegments.size()) return false;

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j)
    {
      auto segment = segments[j];
      auto otherSegment = otherSegments[j];
      auto synapses = segment.synapses;
      auto otherSynapses = otherSegment.synapses;

      if (segment.destroyed != otherSegment.destroyed) return false;
      if (segment.lastUsedIteration != otherSegment.lastUsedIteration) return false;
      if (synapses.size() != otherSynapses.size()) return false;

      for (SynapseIdx k = 0; k < synapses.size(); ++k)
      {
        auto synapse = synapses[k];
        auto otherSynapse = synapses[k];

        if (synapse.presynapticCell != otherSynapse.presynapticCell) return false;
        if (synapse.permanence != otherSynapse.permanence) return false;
        if (synapse.destroyed != otherSynapse.destroyed) return false;
      }
    }
  }

  if (synapsesForPresynapticCell_.size() != other.synapsesForPresynapticCell_.size()) return false;

  for (auto i = synapsesForPresynapticCell_.begin(); i != synapsesForPresynapticCell_.end(); ++i)
  {
    auto synapses = i->second;
    auto otherSynapses = other.synapsesForPresynapticCell_.at(i->first);

    if (synapses.size() != otherSynapses.size()) return false;

    for (SynapseIdx j = 0; j < synapses.size(); ++j)
    {
      auto synapse = synapses[j];
      auto otherSynapse = otherSynapses[j];
      auto segment = synapse.segment;
      auto otherSegment = otherSynapse.segment;
      auto cell = segment.cell;
      auto otherCell = otherSegment.cell;

      if (synapse.idx != otherSynapse.idx) return false;
      if (segment.idx != otherSegment.idx) return false;
      if (cell != otherCell) return false;
    }
  }

  if (numSegments_ != other.numSegments_) return false;
  if (numSynapses_ != other.numSynapses_) return false;
  if (iteration_ != other.iteration_) return false;

  return true;
}

Segment::Segment()
{}

Segment::Segment(CellIdx cell, SegmentIdx idx, UInt32 flatIdx)
  : cell(cell), idx(idx), flatIdx(flatIdx)
{}

bool Segment::operator==(const Segment &other) const
{
  return idx == other.idx && cell == other.cell;
}

bool Segment::operator<=(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell <= other.cell;
  }
  else
  {
    return idx <= other.idx;
  }
}

bool Segment::operator<(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell < other.cell;
  }
  else
  {
    return idx < other.idx;
  }
}

bool Segment::operator>=(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell >= other.cell;
  }
  else
  {
    return idx >= other.idx;
  }
}

bool Segment::operator>(const Segment &other) const
{
  if (cell != other.cell)
  {
    return cell > other.cell;
  }
  else
  {
    return idx > other.idx;
  }
}

bool Synapse::operator==(const Synapse &other) const
{
  return idx == other.idx && segment == other.segment;
}
