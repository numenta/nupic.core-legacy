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
#include <iomanip>
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

Connections::Connections(CellIdx numCells)
{
  initialize(numCells);
}

void Connections::initialize(CellIdx numCells)
{
  cells_ = vector<CellData>(numCells);

  // Every time a segment or synapse is created, we assign it an ordinal and
  // increment the nextOrdinal. Ordinals are never recycled, so they can be used
  // to order segments or synapses by age.
  nextSegmentOrdinal_ = 0;
  nextSynapseOrdinal_ = 0;

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
  Segment segment;
  if (destroyedSegments_.size() > 0)
  {
    segment = destroyedSegments_.back();
    destroyedSegments_.pop_back();
  }
  else
  {
    segment = segments_.size();
    segments_.push_back(SegmentData());
    segmentOrdinals_.push_back(0);
  }

  SegmentData& segmentData = segments_[segment];
  segmentData.cell = cell;

  CellData& cellData = cells_[cell];
  segmentOrdinals_[segment] = nextSegmentOrdinal_++;
  cellData.segments.push_back(segment);

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSegment(segment);
  }

  return segment;
}

Synapse Connections::createSynapse(Segment segment,
                                   CellIdx presynapticCell,
                                   Permanence permanence)
{
  NTA_CHECK(permanence > 0);

  Synapse synapse;
  if (destroyedSynapses_.size() > 0)
  {
    synapse = destroyedSynapses_.back();
    destroyedSynapses_.pop_back();
  }
  else
  {
    synapse.flatIdx = synapses_.size();
    synapses_.push_back(SynapseData());
    synapseOrdinals_.push_back(0);
  }

  SynapseData& synapseData = synapses_[synapse];
  synapseData.segment = segment;
  synapseData.presynapticCell = presynapticCell;
  synapseData.permanence = permanence;

  SegmentData& segmentData = segments_[segment];
  synapseOrdinals_[synapse] = nextSynapseOrdinal_++;
  segmentData.synapses.push_back(synapse);

  synapsesForPresynapticCell_[presynapticCell].push_back(synapse);

  for (auto h : eventHandlers_)
  {
    h.second->onCreateSynapse(synapse);
  }

  return synapse;
}

bool Connections::segmentExists_(Segment segment) const
{
  const SegmentData& segmentData = segments_[segment];
  const vector<Segment>& segmentsOnCell = cells_[segmentData.cell].segments;
  return (std::find(segmentsOnCell.begin(), segmentsOnCell.end(), segment)
          != segmentsOnCell.end());
}

bool Connections::synapseExists_(Synapse synapse) const
{
  const SynapseData& synapseData = synapses_[synapse];
  const vector<Synapse>& synapsesOnSegment = segments_[synapseData.segment].synapses;
  return (std::find(synapsesOnSegment.begin(), synapsesOnSegment.end(), synapse)
          != synapsesOnSegment.end());
}

void Connections::removeSynapseFromPresynapticMap_(Synapse synapse)
{
  const SynapseData& synapseData = synapses_[synapse];
  vector<Synapse>& presynapticSynapses =
    synapsesForPresynapticCell_.at(synapseData.presynapticCell);

  auto it = std::find(presynapticSynapses.begin(), presynapticSynapses.end(),
                      synapse);
  NTA_ASSERT(it != presynapticSynapses.end());
  presynapticSynapses.erase(it);

  if (presynapticSynapses.size() == 0)
  {
    synapsesForPresynapticCell_.erase(synapseData.presynapticCell);
  }
}

void Connections::destroySegment(Segment segment)
{
  NTA_ASSERT(segmentExists_(segment));
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySegment(segment);
  }

  SegmentData& segmentData = segments_[segment];
  for (Synapse synapse : segmentData.synapses)
  {
    // Don't call destroySynapse, since it's unnecessary to do index-shifting.
    removeSynapseFromPresynapticMap_(synapse);
    destroyedSynapses_.push_back(synapse);
  }
  segmentData.synapses.clear();

  CellData& cellData = cells_[segmentData.cell];

  const auto segmentOnCell =
    std::lower_bound(cellData.segments.begin(), cellData.segments.end(),
                     segment,
                     [&](Segment a, Segment b)
                     {
                       return segmentOrdinals_[a] < segmentOrdinals_[b];
                     });

  NTA_ASSERT(segmentOnCell != cellData.segments.end());
  NTA_ASSERT(*segmentOnCell == segment);

  cellData.segments.erase(segmentOnCell);

  destroyedSegments_.push_back(segment);
}

void Connections::destroySynapse(Synapse synapse)
{
  NTA_ASSERT(synapseExists_(synapse));
  for (auto h : eventHandlers_)
  {
    h.second->onDestroySynapse(synapse);
  }

  removeSynapseFromPresynapticMap_(synapse);

  SegmentData& segmentData = segments_[synapses_[synapse].segment];
  const auto synapseOnSegment =
    std::lower_bound(segmentData.synapses.begin(), segmentData.synapses.end(),
                     synapse,
                     [&](Synapse a, Synapse b)
                     {
                       return synapseOrdinals_[a] < synapseOrdinals_[b];
                     });

  NTA_ASSERT(synapseOnSegment != segmentData.synapses.end());
  NTA_ASSERT(*synapseOnSegment == synapse);

  segmentData.synapses.erase(synapseOnSegment);

  destroyedSynapses_.push_back(synapse);
}

void Connections::updateSynapsePermanence(Synapse synapse,
                                          Permanence permanence)
{
  for (auto h : eventHandlers_)
  {
    h.second->onUpdateSynapsePermanence(synapse, permanence);
  }

  synapses_[synapse].permanence = permanence;
}

const vector<Segment>& Connections::segmentsForCell(CellIdx cell) const
{
  return cells_[cell].segments;
}

Segment Connections::getSegment(CellIdx cell, SegmentIdx idx) const
{
  return cells_[cell].segments[idx];
}

const vector<Synapse>& Connections::synapsesForSegment(Segment segment) const
{
  return segments_[segment].synapses;
}

CellIdx Connections::cellForSegment(Segment segment) const
{
  return segments_[segment].cell;
}

SegmentIdx Connections::idxOnCellForSegment(Segment segment) const
{
  const vector<Segment>& segments = segmentsForCell(cellForSegment(segment));
  const auto it = std::find(segments.begin(), segments.end(), segment);
  NTA_ASSERT(it != segments.end());
  return std::distance(segments.begin(), it);
}

void Connections::mapSegmentsToCells(
  const Segment* segments_begin, const Segment* segments_end,
  CellIdx* cells_begin) const
{
  CellIdx* out = cells_begin;

  for (auto segment = segments_begin;
       segment != segments_end;
       ++segment, ++out)
  {
    NTA_ASSERT(segmentExists_(*segment));
    *out = segments_[*segment].cell;
  }
}

Segment Connections::segmentForSynapse(Synapse synapse) const
{
  return synapses_[synapse].segment;
}

const SegmentData& Connections::dataForSegment(Segment segment) const
{
  return segments_[segment];
}

const SynapseData& Connections::dataForSynapse(Synapse synapse) const
{
  return synapses_[synapse];
}

UInt32 Connections::segmentFlatListLength() const
{
  return segments_.size();
}

bool Connections::compareSegments(Segment a, Segment b) const
{
  const SegmentData& aData = segments_[a];
  const SegmentData& bData = segments_[b];
  if (aData.cell < bData.cell)
  {
    return true;
  }
  else if (bData.cell < aData.cell)
  {
    return false;
  }
  else
  {
    return segmentOrdinals_[a] < segmentOrdinals_[b];
  }
}

vector<Synapse> Connections::synapsesForPresynapticCell(
  CellIdx presynapticCell) const
{
  if (synapsesForPresynapticCell_.find(presynapticCell) ==
      synapsesForPresynapticCell_.end())
    return vector<Synapse>{};

  return synapsesForPresynapticCell_.at(presynapticCell);
}

Synapse Connections::minPermanenceSynapse_(Segment segment) const
{
  // Use special EPSILON logic to compensate for floating point differences
  // between C++ and other environments.

  bool found = false;
  Permanence minPermanence = std::numeric_limits<Permanence>::max();
  Synapse minSynapse;

  for (Synapse synapse : segments_[segment].synapses)
  {
    if (synapses_[synapse].permanence < minPermanence - EPSILON)
    {
      minSynapse = synapse;
      minPermanence = synapses_[synapse].permanence;
      found = true;
    }
  }

  NTA_CHECK(found);

  return minSynapse;
}

void Connections::computeActivity(
  vector<UInt32>& numActiveConnectedSynapsesForSegment,
  vector<UInt32>& numActivePotentialSynapsesForSegment,
  CellIdx activePresynapticCell,
  Permanence connectedPermanence) const
{
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == segments_.size());

  if (synapsesForPresynapticCell_.count(activePresynapticCell))
  {
    for (Synapse synapse :
           synapsesForPresynapticCell_.at(activePresynapticCell))
    {
      const SynapseData& synapseData = synapses_[synapse];
      ++numActivePotentialSynapsesForSegment[synapseData.segment];

      NTA_ASSERT(synapseData.permanence > 0);
      if (synapseData.permanence >= connectedPermanence - EPSILON)
      {
        ++numActiveConnectedSynapsesForSegment[synapseData.segment];
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
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == segments_.size());

  for (CellIdx cell : activePresynapticCells)
  {
    if (synapsesForPresynapticCell_.count(cell))
    {
      for (Synapse synapse : synapsesForPresynapticCell_.at(cell))
      {
        const SynapseData& synapseData = synapses_[synapse];
        ++numActivePotentialSynapsesForSegment[synapseData.segment];

        NTA_ASSERT(synapseData.permanence > 0);
        if (synapseData.permanence >= connectedPermanence - EPSILON)
        {
          ++numActiveConnectedSynapsesForSegment[synapseData.segment];
        }
      }
    }
  }
}

template<typename FloatType>
static void saveFloat_(std::ostream& outStream, FloatType v)
{
  outStream << std::setprecision(std::numeric_limits<FloatType>::max_digits10)
            << v
            << " ";
}

void Connections::save(std::ostream& outStream) const
{
  // Write a starting marker.
  outStream << "Connections" << endl;
  outStream << Connections::VERSION << endl;

  outStream << cells_.size() << " "
            << endl;

  for (CellData cellData : cells_)
  {
    const vector<Segment>& segments = cellData.segments;
    outStream << segments.size() << " ";

    for (Segment segment : segments)
    {
      const SegmentData& segmentData = segments_[segment];

      const vector<Synapse>& synapses = segmentData.synapses;
      outStream << synapses.size() << " ";

      for (Synapse synapse : synapses)
      {
        const SynapseData& synapseData = synapses_[synapse];
        outStream << synapseData.presynapticCell << " ";
        saveFloat_(outStream, synapseData.permanence);
      }
      outStream << endl;
    }
    outStream << endl;
  }
  outStream << endl;

  outStream << "~Connections" << endl;
}

void Connections::write(ConnectionsProto::Builder& proto) const
{
  proto.setVersion(Connections::VERSION);

  auto protoCells = proto.initCells(cells_.size());

  for (CellIdx i = 0; i < cells_.size(); ++i)
  {
    const vector<Segment>& segments = cells_[i].segments;
    auto protoSegments = protoCells[i].initSegments(segments.size());

    for (SegmentIdx j = 0; j < (SegmentIdx)segments.size(); ++j)
    {
      const SegmentData& segmentData = segments_[segments[j]];
      const vector<Synapse>& synapses = segmentData.synapses;

      auto protoSynapses = protoSegments[j].initSynapses(synapses.size());

      for (SynapseIdx k = 0; k < synapses.size(); ++k)
      {
        const SynapseData& synapseData = synapses_[synapses[k]];

        protoSynapses[k].setPresynapticCell(synapseData.presynapticCell);
        protoSynapses[k].setPermanence(synapseData.permanence);
      }
    }
  }
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
  inStream >> numCells;

  initialize(numCells);

  // This logic is complicated by the fact that old versions of the Connections
  // serialized "destroyed" segments and synapses, which we now ignore.
  cells_.resize(numCells);
  for (UInt cell = 0; cell < numCells; cell++)
  {
    CellData& cellData = cells_[cell];

    UInt numSegments;
    inStream >> numSegments;

    for (SegmentIdx j = 0; j < numSegments; j++)
    {
      bool destroyedSegment = false;
      if (version < 2)
      {
        inStream >> destroyedSegment;
      }

      Segment segment = {(UInt32)-1};
      {
        SegmentData segmentData = {};
        segmentData.cell = cell;

        if (!destroyedSegment)
        {
          segment = segments_.size();
          cellData.segments.push_back(segment);
          segments_.push_back(segmentData);
          segmentOrdinals_.push_back(nextSegmentOrdinal_++);
        }
      }

      UInt numSynapses;
      inStream >> numSynapses;

      for (SynapseIdx k = 0; k < numSynapses; k++)
      {
        SynapseData synapseData = {};
        inStream >> synapseData.presynapticCell;
        inStream >> synapseData.permanence;

        bool destroyedSynapse = false;
        if (version < 2)
        {
          inStream >> destroyedSynapse;
        }

        if (!destroyedSegment && !destroyedSynapse)
        {
          synapseData.segment = segment;

          SegmentData& segmentData = segments_[segment];

          Synapse synapse = {(UInt32)synapses_.size()};
          segmentData.synapses.push_back(synapse);
          synapses_.push_back(synapseData);
          synapseOrdinals_.push_back(nextSynapseOrdinal_++);

          synapsesForPresynapticCell_[synapseData.presynapticCell].push_back(
            synapse);
        }
      }
    }
  }

  inStream >> marker;
  NTA_CHECK(marker == "~Connections");
}

void Connections::read(ConnectionsProto::Reader& proto)
{
  // Check the saved version.
  UInt version = proto.getVersion();
  NTA_CHECK(version <= Connections::VERSION);

  auto protoCells = proto.getCells();

  initialize(protoCells.size());

  for (CellIdx cell = 0; cell < protoCells.size(); ++cell)
  {
    CellData& cellData = cells_[cell];

    auto protoSegments = protoCells[cell].getSegments();

    for (SegmentIdx j = 0; j < (SegmentIdx)protoSegments.size(); ++j)
    {
      Segment segment;
      {
        const SegmentData segmentData = {vector<Synapse>(),
                                         cell};
        segment = segments_.size();
        cellData.segments.push_back(segment);
        segments_.push_back(segmentData);
        segmentOrdinals_.push_back(nextSegmentOrdinal_++);
      }

      SegmentData& segmentData = segments_[segment];

      auto protoSynapses = protoSegments[j].getSynapses();

      for (SynapseIdx k = 0; k < protoSynapses.size(); ++k)
      {
        CellIdx presynapticCell = protoSynapses[k].getPresynapticCell();
        SynapseData synapseData = {presynapticCell,
                                   protoSynapses[k].getPermanence(),
                                   segment};
        Synapse synapse = {(UInt32)synapses_.size()};
        synapses_.push_back(synapseData);
        synapseOrdinals_.push_back(nextSynapseOrdinal_++);
        segmentData.synapses.push_back(synapse);

        synapsesForPresynapticCell_[presynapticCell].push_back(synapse);
      }
    }
  }
}

CellIdx Connections::numCells() const
{
  return cells_.size();
}

UInt Connections::numSegments() const
{
  return segments_.size() - destroyedSegments_.size();
}

UInt Connections::numSegments(CellIdx cell) const
{
  return cells_[cell].segments.size();
}

UInt Connections::numSynapses() const
{
  return synapses_.size() - destroyedSynapses_.size();
}

UInt Connections::numSynapses(Segment segment) const
{
  return segments_[segment].synapses.size();
}

bool Connections::operator==(const Connections &other) const
{
  if (cells_.size() != other.cells_.size()) return false;

  for (CellIdx i = 0; i < cells_.size(); ++i)
  {
    const CellData& cellData = cells_[i];
    const CellData& otherCellData = other.cells_[i];

    if (cellData.segments.size() != otherCellData.segments.size())
    {
      return false;
    }

    for (SegmentIdx j = 0; j < (SegmentIdx)cellData.segments.size(); ++j)
    {
      Segment segment = cellData.segments[j];
      const SegmentData& segmentData = segments_[segment];
      Segment otherSegment = otherCellData.segments[j];
      const SegmentData& otherSegmentData = other.segments_[otherSegment];

      if (segmentData.synapses.size() != otherSegmentData.synapses.size() ||
          segmentData.cell != otherSegmentData.cell)
      {
        return false;
      }

      for (SynapseIdx k = 0; k < (SynapseIdx)segmentData.synapses.size(); ++k)
      {
        Synapse synapse = segmentData.synapses[k];
        const SynapseData& synapseData = synapses_[synapse];
        Synapse otherSynapse = otherSegmentData.synapses[k];
        const SynapseData& otherSynapseData = other.synapses_[otherSynapse];

        if (synapseData.presynapticCell != otherSynapseData.presynapticCell ||
            synapseData.permanence != otherSynapseData.permanence)
        {
          return false;
        }

        // Two functionally identical instances may have different flatIdxs.
        NTA_ASSERT(synapseData.segment == segment);
        NTA_ASSERT(otherSynapseData.segment == otherSegment);
      }
    }
  }

  if (synapsesForPresynapticCell_.size() !=
      other.synapsesForPresynapticCell_.size()) return false;

  for (auto i = synapsesForPresynapticCell_.begin();
       i != synapsesForPresynapticCell_.end(); ++i)
  {
    const vector<Synapse>& synapses = i->second;
    const vector<Synapse>& otherSynapses =
      other.synapsesForPresynapticCell_.at(i->first);

    if (synapses.size() != otherSynapses.size()) return false;

    for (SynapseIdx j = 0; j < synapses.size(); ++j)
    {
      Synapse synapse = synapses[j];
      const SynapseData& synapseData = synapses_[synapse];
      const SegmentData& segmentData = segments_[synapseData.segment];
      Synapse otherSynapse = otherSynapses[j];
      const SynapseData& otherSynapseData = other.synapses_[otherSynapse];
      const SegmentData& otherSegmentData =
        other.segments_[otherSynapseData.segment];

      if (segmentData.cell != otherSegmentData.cell)
      {
        return false;
      }
    }
  }

  return true;
}
