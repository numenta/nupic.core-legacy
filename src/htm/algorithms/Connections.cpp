/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2014-2016, Numenta, Inc.
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
 * ---------------------------------------------------------------------- */

/** @file
 * Implementation of Connections
 */

#include <algorithm> // nth_element
#include <climits>
#include <iomanip>
#include <iostream>

#include <htm/algorithms/Connections.hpp>

using std::endl;
using std::string;
using std::vector;
using namespace htm;

Connections::Connections(const CellIdx numCells, 
		         const Permanence connectedThreshold, 
			 const bool timeseries) {
  initialize(numCells, connectedThreshold, timeseries);
}

void Connections::initialize(CellIdx numCells, Permanence connectedThreshold, bool timeseries) {
  cells_ = vector<CellData>(numCells);
  segments_.clear();
  destroyedSegments_.clear();
  synapses_.clear();
  destroyedSynapses_.clear();
  potentialSynapsesForPresynapticCell_.clear();
  connectedSynapsesForPresynapticCell_.clear();
  potentialSegmentsForPresynapticCell_.clear();
  connectedSegmentsForPresynapticCell_.clear();
  segmentOrdinals_.clear();
  synapseOrdinals_.clear();
  eventHandlers_.clear();
  NTA_CHECK(connectedThreshold >= minPermanence);
  NTA_CHECK(connectedThreshold <= maxPermanence);
  connectedThreshold_ = connectedThreshold - htm::Epsilon;

  // Every time a segment or synapse is created, we assign it an ordinal and
  // increment the nextOrdinal. Ordinals are never recycled, so they can be used
  // to order segments or synapses by age.
  nextSegmentOrdinal_ = 0;
  nextSynapseOrdinal_ = 0;

  nextEventToken_ = 0;

  timeseries_ = timeseries;
  reset();
}

UInt32 Connections::subscribe(ConnectionsEventHandler *handler) {
  UInt32 token = nextEventToken_++;
  eventHandlers_[token] = handler;
  return token;
}

void Connections::unsubscribe(UInt32 token) {
  delete eventHandlers_.at(token);
  eventHandlers_.erase(token);
}

Segment Connections::createSegment(const CellIdx cell, 
	                           const SegmentIdx maxSegmentsPerCell,
				   const UInt32 iteration) { //TODO move iteration to Connections.iteration_ ?

  //limit number of segmets per cell. If exceeded, remove the least recently used ones.
  NTA_ASSERT(maxSegmentsPerCell > 0);
  while (maxSegmentsPerCell > 1 && numSegments(cell) >= maxSegmentsPerCell) {
    const auto& destroyCandidates = segmentsForCell(cell);
    const auto compareSegmentsByLRU = [&](const Segment a, const Segment b) {
      return (dataForSegment(a).lastUsed < dataForSegment(b).lastUsed); };
    const auto leastRecentlyUsedSegment = std::min_element(destroyCandidates.cbegin(), 
        destroyCandidates.cend(), compareSegmentsByLRU);

    destroySegment(*leastRecentlyUsedSegment);
  }

  //proceed to create a new segment
  Segment segment;
  if (!destroyedSegments_.empty() ) { //reuse old, destroyed segs
    segment = destroyedSegments_.back();
    destroyedSegments_.pop_back();
  } else { //create a new segment
    NTA_CHECK(segments_.size() < std::numeric_limits<Segment>::max()) << "Add segment failed: Range of Segment (data-type) insufficinet size."
	    << (size_t)segments_.size() << " < " << (size_t)std::numeric_limits<Segment>::max();
    segment = static_cast<Segment>(segments_.size());
    const SegmentData& segmentData = SegmentData(cell, iteration);
    segments_.push_back(segmentData);
    segmentOrdinals_.push_back(0);
  }

  CellData &cellData = cells_[cell];
  segmentOrdinals_[segment] = nextSegmentOrdinal_++;
  cellData.segments.push_back(segment); //assign the new segment to its mother-cell

  for (auto h : eventHandlers_) {
    h.second->onCreateSegment(segment);
  }

  return segment;
}


Synapse Connections::createSynapse(Segment segment,
                                   CellIdx presynapticCell,
                                   Permanence permanence) {
  // Get an index into the synapses_ list, for the new synapse to reside at.
  Synapse synapse;
  if (!destroyedSynapses_.empty() ) {
    synapse = destroyedSynapses_.back();
    destroyedSynapses_.pop_back();
  } else {
    NTA_CHECK(synapses_.size() < std::numeric_limits<Synapse>::max()) << "Add synapse failed: Range of Synapse (data-type) insufficient size."
	    << synapses_.size() << " < " << (size_t)std::numeric_limits<Synapse>::max();
    synapse = static_cast<Synapse>(synapses_.size());
    synapses_.push_back(SynapseData());
    synapseOrdinals_.push_back(0);
  }

  // Fill in the new synapse's data
  SynapseData &synapseData    = synapses_[synapse];
  synapseData.presynapticCell = presynapticCell;
  synapseData.segment         = segment;
  synapseOrdinals_[synapse]   = nextSynapseOrdinal_++;
  // Start in disconnected state.
  synapseData.permanence           = connectedThreshold_ - 1.0f;
  synapseData.presynapticMapIndex_ = 
    (Synapse)potentialSynapsesForPresynapticCell_[presynapticCell].size();
  potentialSynapsesForPresynapticCell_[presynapticCell].push_back(synapse);
  potentialSegmentsForPresynapticCell_[presynapticCell].push_back(segment);

  SegmentData &segmentData = segments_[segment];
  segmentData.synapses.push_back(synapse);


  for (auto h : eventHandlers_) {
    h.second->onCreateSynapse(synapse);
  }

  updateSynapsePermanence(synapse, permanence);

  return synapse;
}

bool Connections::segmentExists_(const Segment segment) const {
  const SegmentData &segmentData = segments_[segment];
  const vector<Segment> &segmentsOnCell = cells_[segmentData.cell].segments;
  return (std::find(segmentsOnCell.begin(), segmentsOnCell.end(), segment) !=
          segmentsOnCell.end());
}

bool Connections::synapseExists_(const Synapse synapse) const {
  const SynapseData &synapseData = synapses_[synapse];
  const vector<Synapse> &synapsesOnSegment =
      segments_[synapseData.segment].synapses;
  return (std::find(synapsesOnSegment.begin(), synapsesOnSegment.end(),
                    synapse) != synapsesOnSegment.end());
}

/**
 * Helper method to remove a synapse from a presynaptic map, by moving the
 * last synapse in the list over this synapse.
 */
void Connections::removeSynapseFromPresynapticMap_(
    const Synapse index,
    vector<Synapse> &preSynapses,
    vector<Segment> &preSegments)
{
  NTA_ASSERT( !preSynapses.empty() );
  NTA_ASSERT( index < preSynapses.size() );
  NTA_ASSERT( preSynapses.size() == preSegments.size() );

  const auto move = preSynapses.back();
  synapses_[move].presynapticMapIndex_ = index;
  preSynapses[index] = move;
  preSynapses.pop_back();

  preSegments[index] = preSegments.back();
  preSegments.pop_back();
}


void Connections::destroySegment(const Segment segment) {
  NTA_ASSERT(segmentExists_(segment));
  for (auto h : eventHandlers_) {
    h.second->onDestroySegment(segment);
  }

  SegmentData &segmentData = segments_[segment];

  // Destroy synapses from the end of the list, so that the index-shifting is
  // easier to do.
  while( !segmentData.synapses.empty() )
    destroySynapse(segmentData.synapses.back());

  CellData &cellData = cells_[segmentData.cell];

  const auto segmentOnCell =
      std::lower_bound(cellData.segments.begin(), cellData.segments.end(),
                       segment, [&](Segment a, Segment b) {
                         return segmentOrdinals_[a] < segmentOrdinals_[b];
                       });

  NTA_ASSERT(segmentOnCell != cellData.segments.end());
  NTA_ASSERT(*segmentOnCell == segment);

  cellData.segments.erase(segmentOnCell);

  destroyedSegments_.push_back(segment);
}


void Connections::destroySynapse(const Synapse synapse) {
  NTA_ASSERT(synapseExists_(synapse));
  for (auto h : eventHandlers_) {
    h.second->onDestroySynapse(synapse);
  }

  const SynapseData &synapseData = synapses_[synapse];
        SegmentData &segmentData = segments_[synapseData.segment];
  const auto         presynCell  = synapseData.presynapticCell;

  if( synapseData.permanence >= connectedThreshold_ ) {
    segmentData.numConnected--;

    removeSynapseFromPresynapticMap_(
      synapseData.presynapticMapIndex_,
      connectedSynapsesForPresynapticCell_.at( presynCell ),
      connectedSegmentsForPresynapticCell_.at( presynCell ));

    if( connectedSynapsesForPresynapticCell_.at( presynCell ).empty() ){
      connectedSynapsesForPresynapticCell_.erase( presynCell );
      connectedSegmentsForPresynapticCell_.erase( presynCell );
    }
  }
  else {
    removeSynapseFromPresynapticMap_(
      synapseData.presynapticMapIndex_,
      potentialSynapsesForPresynapticCell_.at( presynCell ),
      potentialSegmentsForPresynapticCell_.at( presynCell ));

    if( potentialSynapsesForPresynapticCell_.at( presynCell ).empty() ){
      potentialSynapsesForPresynapticCell_.erase( presynCell );
      potentialSegmentsForPresynapticCell_.erase( presynCell );
    }
  }

  const auto synapseOnSegment =
      std::lower_bound(segmentData.synapses.begin(), segmentData.synapses.end(),
                       synapse, [&](Synapse a, Synapse b) {
                         return synapseOrdinals_[a] < synapseOrdinals_[b];
                       });

  NTA_ASSERT(synapseOnSegment != segmentData.synapses.end());
  NTA_ASSERT(*synapseOnSegment == synapse);

  segmentData.synapses.erase(synapseOnSegment);

  destroyedSynapses_.push_back(synapse);
}


void Connections::updateSynapsePermanence(const Synapse synapse,
                                          Permanence permanence) {
  permanence = std::min(permanence, maxPermanence );
  permanence = std::max(permanence, minPermanence );

  auto &synData = synapses_[synapse];
  
  const bool before = synData.permanence >= connectedThreshold_;
  const bool after  = permanence         >= connectedThreshold_;
  synData.permanence = permanence;

  if( before == after ) { //no change
      return;
  }
    const auto &presyn    = synData.presynapticCell;
    auto &potentialPresyn = potentialSynapsesForPresynapticCell_[presyn];
    auto &potentialPreseg = potentialSegmentsForPresynapticCell_[presyn];
    auto &connectedPresyn = connectedSynapsesForPresynapticCell_[presyn];
    auto &connectedPreseg = connectedSegmentsForPresynapticCell_[presyn];
    const auto &segment   = synData.segment;
    auto &segmentData     = segments_[segment];
    
    if( after ) { //connect
      segmentData.numConnected++;

      // Remove this synapse from presynaptic potential synapses.
      removeSynapseFromPresynapticMap_( synData.presynapticMapIndex_,
                                        potentialPresyn, potentialPreseg );

      // Add this synapse to the presynaptic connected synapses.
      synData.presynapticMapIndex_ = (Synapse)connectedPresyn.size();
      connectedPresyn.push_back( synapse );
      connectedPreseg.push_back( segment );
    }
    else { //disconnected
      segmentData.numConnected--;

      // Remove this synapse from presynaptic connected synapses.
      removeSynapseFromPresynapticMap_( synData.presynapticMapIndex_,
                                        connectedPresyn, connectedPreseg );

      // Add this synapse to the presynaptic connected synapses.
      synData.presynapticMapIndex_ = (Synapse)potentialPresyn.size();
      potentialPresyn.push_back( synapse );
      potentialPreseg.push_back( segment );
    }

    for (auto h : eventHandlers_) { //TODO handle callbacks in performance-critical method only in Debug?
      h.second->onUpdateSynapsePermanence(synapse, permanence);
    }
}


SegmentIdx Connections::idxOnCellForSegment(const Segment segment) const {
  const vector<Segment> &segments = segmentsForCell(cellForSegment(segment));
  const auto it = std::find(segments.begin(), segments.end(), segment);
  NTA_ASSERT(it != segments.end());
  return (SegmentIdx)std::distance(segments.begin(), it);
}


void Connections::mapSegmentsToCells(const Segment *segments_begin,
                                     const Segment *segments_end,
                                     CellIdx *cells_begin) const {
  CellIdx *out = cells_begin;

  for (auto segment = segments_begin; segment != segments_end;
       ++segment, ++out) {
    NTA_ASSERT(segmentExists_(*segment));
    *out = segments_[*segment].cell;
  }
}


bool Connections::compareSegments(const Segment a, const Segment b) const {
  const SegmentData &aData = segments_[a];
  const SegmentData &bData = segments_[b];
  if (aData.cell < bData.cell) {
    return true;
  } else if (bData.cell < aData.cell) {
    return false;
  } else {
    return segmentOrdinals_[a] < segmentOrdinals_[b];
  }
}

vector<Synapse>
Connections::synapsesForPresynapticCell(CellIdx presynapticCell) const {
  vector<Synapse> all(
      potentialSynapsesForPresynapticCell_.at(presynapticCell).begin(),
      potentialSynapsesForPresynapticCell_.at(presynapticCell).end());
  all.insert( all.end(),
      connectedSynapsesForPresynapticCell_.at(presynapticCell).begin(),
      connectedSynapsesForPresynapticCell_.at(presynapticCell).end());
  return all;
}

void Connections::reset()
{
  if( not timeseries_ ) {
    NTA_WARN << "Connections::reset() called with timeseries=false.";
  }
  previousUpdates_.clear();
  currentUpdates_.clear();
}


void Connections::computeActivity(
    vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
    const vector<CellIdx> &activePresynapticCells)
{
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());

  if( timeseries_ ) {
    // Before each cycle of computation move the currentUpdates to the previous
    // updates, and zero the currentUpdates in preparation for learning.
    previousUpdates_.swap( currentUpdates_ );
    currentUpdates_.clear();
  }

  // Iterate through all connected synapses.
  for (const auto& cell : activePresynapticCells) {
    if (connectedSegmentsForPresynapticCell_.count(cell)) {
      for(const auto& segment : connectedSegmentsForPresynapticCell_.at(cell)) {
        ++numActiveConnectedSynapsesForSegment[segment];
      }
    }
  }
}

void Connections::computeActivity(
    vector<SynapseIdx> &numActiveConnectedSynapsesForSegment,
    vector<SynapseIdx> &numActivePotentialSynapsesForSegment,
    const vector<CellIdx> &activePresynapticCells) {
  NTA_ASSERT(numActiveConnectedSynapsesForSegment.size() == segments_.size());
  NTA_ASSERT(numActivePotentialSynapsesForSegment.size() == segments_.size());

  // Iterate through all connected synapses.
  computeActivity(
      numActiveConnectedSynapsesForSegment,
      activePresynapticCells );

  // Iterate through all potential synapses.
  std::copy( numActiveConnectedSynapsesForSegment.begin(),
             numActiveConnectedSynapsesForSegment.end(),
             numActivePotentialSynapsesForSegment.begin());
  for (const auto& cell : activePresynapticCells) {
    if (potentialSegmentsForPresynapticCell_.count(cell)) {
      for(const auto& segment : potentialSegmentsForPresynapticCell_.at(cell)) {
        ++numActivePotentialSynapsesForSegment[segment];
      }
    }
  }
}


void Connections::adaptSegment(const Segment segment, 
                               const SDR &inputs,
                               const Permanence increment,
                               const Permanence decrement, 
			       const bool pruneZeroSynapses)
{
  const auto &inputArray = inputs.getDense();

  if( timeseries_ ) {
    previousUpdates_.resize( synapses_.size(), minPermanence );
    currentUpdates_.resize(  synapses_.size(), minPermanence );
  }

  const auto& synapses = synapsesForSegment(segment);
  for( size_t i = 0; i <  synapses.size(); i++) {
      const auto synapse = synapses[i];
      const SynapseData &synapseData = dataForSynapse(synapse);

      Permanence update;
      if( inputArray[synapseData.presynapticCell] ) {
        update = increment;
      } else {
        update = -decrement;
      }

    //prune permanences that reached zero
    if (pruneZeroSynapses && synapseData.permanence + update < htm::minPermanence + htm::Epsilon) {
      destroySynapse(synapse);
      i--; // do not advance `i`, as `destroySynapse` just modified inplace the synapses_, so now a `synapses_[i]`
      // is the "next" synapse. 
      continue;
    }

    //update synapse, but for TS only if changed
    if(timeseries_) {
      if( update != previousUpdates_[synapse] ) {
        updateSynapsePermanence(synapse, synapseData.permanence + update);
      }
      currentUpdates_[ synapse ] = update;
    } else {
      updateSynapsePermanence(synapse, synapseData.permanence + update);
    }
  }

  //destroy segment if it is empty
  if(synapses.empty()) {
    destroySegment(segment);
  }
}


/**
 * Called for under-performing Segments (can have synapses pruned, etc.). After
 * the call, Segment will have at least segmentThreshold synapses connected, so
 * the Segment could be active next time.
 */
void Connections::raisePermanencesToThreshold(
                  const Segment    segment,
                  const UInt       segmentThreshold)
{
  if( segmentThreshold == 0 ) // No synapses requested to be connected, done.
    return;

  NTA_ASSERT(segment < segments_.size()) << "Accessing segment out of bounds.";
  auto &segData = segments_[segment];
  if( segData.numConnected >= segmentThreshold )
    return;   // The segment already satisfies the requirement, done.

  vector<Synapse> &synapses = segData.synapses;
  if( synapses.empty())
    return;   // No synapses to raise permanences to, no work to do.

  // Prune empty segment? No. 
  // The SP calls this method, but the SP does not do any pruning. 
  // The TM already has code to do pruning, but it doesn't ever call this method.

  // There can be situations when synapses are pruned so the segment has too few
  // synapses to ever activate, so we cannot satisfy the >= segmentThreshold
  // connected.  In this case the method should do the next best thing and
  // connect as many synapses as it can.

  // Keep segmentThreshold within synapses range.
  const auto threshold = std::min((size_t)segmentThreshold, synapses.size());

  // Sort the potential pool by permanence values, and look for the synapse with
  // the N'th greatest permanence, where N is the desired minimum number of
  // connected synapses.  Then calculate how much to increase the N'th synapses
  // permance by such that it becomes a connected synapse.  After that there
  // will be at least N synapses connected.

  // Threshold is ensured to be >=1 by condition at very beginning if(thresh == 0)... 
  auto minPermSynPtr = synapses.begin() + threshold - 1;

  const auto permanencesGreater = [&](const Synapse &A, const Synapse &B)
    { return synapses_[A].permanence > synapses_[B].permanence; };
  // Do a partial sort, it's faster than a full sort.
  std::nth_element(synapses.begin(), minPermSynPtr, synapses.end(), permanencesGreater);

  const Real increment = connectedThreshold_ - synapses_[ *minPermSynPtr ].permanence;
  if( increment <= 0 ) // If minPermSynPtr is already connected then ...
    return;            // Enough synapses are already connected.

  // Raise the permance of all synapses in the potential pool uniformly.
  for( const auto &syn : synapses ) //TODO vectorize: vector + const to all members
    updateSynapsePermanence(syn, synapses_[syn].permanence + increment); //this is performance HOTSPOT
}


void Connections::bumpSegment(const Segment segment, const Permanence delta) {
  const vector<Synapse> &synapses = synapsesForSegment(segment);
  for( const auto &syn : synapses ) {
    updateSynapsePermanence(syn, synapses_[syn].permanence + delta);
  }
}


void Connections::destroyMinPermanenceSynapses(
                              const Segment segment, Int nDestroy,
                              const vector<CellIdx> &excludeCells)
{
  NTA_ASSERT( nDestroy >= 0 );
  if( nDestroy <= 0 ) return; // Nothing to do.

  // Don't destroy any cells that are in excludeCells.
  vector<Synapse> destroyCandidates;
  for( Synapse synapse : synapsesForSegment(segment)) {
    const CellIdx presynapticCell = dataForSynapse(synapse).presynapticCell;

    if( not std::binary_search(excludeCells.cbegin(), excludeCells.cend(), presynapticCell)) {
      destroyCandidates.push_back(synapse);
    }
  }

  const auto comparePermanences = [&](const Synapse A, const Synapse B) {
    const Permanence A_perm = dataForSynapse(A).permanence;
    const Permanence B_perm = dataForSynapse(B).permanence;
    if( A_perm == B_perm ) {
      return A < B;
    }
    else {
      return A_perm < B_perm;
    }
  };
  std::sort(destroyCandidates.begin(), destroyCandidates.end(), comparePermanences);

  nDestroy = std::min( nDestroy, (Int) destroyCandidates.size() );
  for(Int i = 0; i < nDestroy; i++) {
    destroySynapse( destroyCandidates[i] );
  }
}


namespace htm {
std::ostream& operator<< (std::ostream& stream, const Connections& self)
{
  stream << "Connections:" << std::endl;
  const auto numPresyns = self.potentialSynapsesForPresynapticCell_.size();
  stream << "    Inputs (" << numPresyns
         << ") ~> Outputs (" << self.cells_.size()
         << ") via Segments (" << self.numSegments() << ")" << std::endl;

  UInt        segmentsMin   = -1;
  Real        segmentsMean  = 0.0f;
  UInt        segmentsMax   = 0u;
  UInt        potentialMin  = -1;
  Real        potentialMean = 0.0f;
  UInt        potentialMax  = 0;
  SynapseIdx  connectedMin  = -1;
  Real        connectedMean = 0.0f;
  SynapseIdx  connectedMax  = 0;
  UInt        synapsesDead      = 0;
  UInt        synapsesSaturated = 0;
  for( const auto cellData : self.cells_ )
  {
    const UInt numSegments = (UInt) cellData.segments.size();
    segmentsMin   = std::min( segmentsMin, numSegments );
    segmentsMax   = std::max( segmentsMax, numSegments );
    segmentsMean += numSegments;

    for( const auto seg : cellData.segments ) {
      const auto &segData = self.dataForSegment( seg );

      const UInt numPotential = (UInt) segData.synapses.size();
      potentialMin   = std::min( potentialMin, numPotential );
      potentialMax   = std::max( potentialMax, numPotential );
      potentialMean += numPotential;

      connectedMin   = std::min( connectedMin, segData.numConnected );
      connectedMax   = std::max( connectedMax, segData.numConnected );
      connectedMean += segData.numConnected;

      for( const auto syn : segData.synapses ) {
        const auto &synData = self.dataForSynapse( syn );
        if( synData.permanence == minPermanence )
          { synapsesDead++; }
        else if( synData.permanence == maxPermanence )
          { synapsesSaturated++; }
      }
    }
  }
  segmentsMean  = segmentsMean  / self.numCells();
  potentialMean = potentialMean / self.numSegments();
  connectedMean = connectedMean / self.numSegments();

  stream << "    Segments on Cell Min/Mean/Max "
         << segmentsMin << " / " << segmentsMean << " / " << segmentsMax << std::endl;
  stream << "    Potential Synapses on Segment Min/Mean/Max "
         << potentialMin << " / " << potentialMean << " / " << potentialMax << std::endl;
  stream << "    Connected Synapses on Segment Min/Mean/Max "
         << connectedMin << " / " << connectedMean << " / " << connectedMax << std::endl;

  stream << "    Synapses Dead (" << (Real) synapsesDead / self.numSynapses()
         << "%) Saturated (" <<   (Real) synapsesSaturated / self.numSynapses() << "%)" << std::endl;

  return stream;
}
}



bool Connections::operator==(const Connections &other) const {
  if (cells_.size() != other.cells_.size())
    return false;

  for (CellIdx i = 0; i < static_cast<CellIdx>(cells_.size()); i++) {
    const CellData &cellData = cells_[i];
    const CellData &otherCellData = other.cells_[i];

    if (cellData.segments.size() != otherCellData.segments.size()) {
      return false;
    }

    for (SegmentIdx j = 0; j < static_cast<SegmentIdx>(cellData.segments.size()); j++) {
      const Segment segment = cellData.segments[j];
      const SegmentData &segmentData = segments_[segment];
      const Segment otherSegment = otherCellData.segments[j];
      const SegmentData &otherSegmentData = other.segments_[otherSegment];

      if (segmentData.synapses.size() != otherSegmentData.synapses.size() ||
          segmentData.cell != otherSegmentData.cell) {
        return false;
      }

      for (SynapseIdx k = 0; k < static_cast<SynapseIdx>(segmentData.synapses.size()); k++) {
        const Synapse synapse = segmentData.synapses[k];
        const SynapseData &synapseData = synapses_[synapse];
        const Synapse otherSynapse = otherSegmentData.synapses[k];
        const SynapseData &otherSynapseData = other.synapses_[otherSynapse];

        if (synapseData.presynapticCell != otherSynapseData.presynapticCell ||
            synapseData.permanence != otherSynapseData.permanence) {
          return false;
        }

        // Two functionally identical instances may have different flatIdxs.
        NTA_ASSERT(synapseData.segment == segment);
        NTA_ASSERT(otherSynapseData.segment == otherSegment);
      }
    }
  }

  return true;
}

