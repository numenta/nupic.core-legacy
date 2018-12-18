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
 * Definitions for the Connections class in C++
 */

#ifndef NTA_CONNECTIONS_HPP
#define NTA_CONNECTIONS_HPP

#include <climits>
#include <utility>
#include <vector>

#include <nupic/math/Math.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>
#include <nupic/ntypes/Sdr.hpp>

namespace nupic {

namespace algorithms {

namespace connections {
typedef UInt32 CellIdx;
typedef UInt16 SegmentIdx; /** Index of segment in cell. */
typedef UInt16 SynapseIdx; /** Index of synapse in segment. */
typedef UInt32 Segment;    /** Index of segment's data. */
typedef UInt32 Synapse;    /** Index of synapse's data. */
typedef Real32 Permanence;
const Permanence minPermanence = 0.0f;
const Permanence maxPermanence = 1.0f;

static const Permanence EPSILON = 0.00001f;

/**
 * SynapseData class used in Connections.
 *
 * @b Description
 * The SynapseData contains the underlying data for a synapse.
 *
 * @param presynapticCellIdx
 * Cell that this synapse gets input from.
 *
 * @param permanence
 * Permanence of synapse.
 */
struct SynapseData {
  CellIdx presynapticCell;
  Permanence permanence;
  Segment segment;
};

/**
 * SegmentData class used in Connections.
 *
 * @b Description
 * The SegmentData contains the underlying data for a Segment.
 *
 * @param synapses
 * Synapses on this segment.
 *
 * @param cell
 * The cell that this segment is on.
 */
struct SegmentData {
  std::vector<Synapse> synapses;
  CellIdx cell;
  SynapseIdx numConnected;
};

/**
 * CellData class used in Connections.
 *
 * @b Description
 * The CellData contains the underlying data for a Cell.
 *
 * @param segments
 * Segments on this cell.
 *
 */
struct CellData {
  std::vector<Segment> segments;
};

/**
 * A base class for Connections event handlers.
 *
 * @b Description
 * This acts as a plug-in point for logging / visualizations.
 */
class ConnectionsEventHandler {
public:
  virtual ~ConnectionsEventHandler() {}

  /**
   * Called after a segment is created.
   */
  virtual void onCreateSegment(Segment segment) {}

  /**
   * Called before a segment is destroyed.
   */
  virtual void onDestroySegment(Segment segment) {}

  /**
   * Called after a synapse is created.
   */
  virtual void onCreateSynapse(Synapse synapse) {}

  /**
   * Called before a synapse is destroyed.
   */
  virtual void onDestroySynapse(Synapse synapse) {}

  /**
   * Called after a synapse's permanence crosses the connected threshold.
   */
  virtual void onUpdateSynapsePermanence(Synapse synapse,
                                         Permanence permanence) {}
};

/**
 * Connections implementation in C++.
 *
 * @b Description
 * The Connections class is a data structure that represents the
 * connections of a collection of cells. It is used in the HTM
 * learning algorithms to store and access data related to the
 * connectivity of cells.
 *
 * Its main utility is to provide a common, optimized data structure
 * that all HTM learning algorithms can use. It is flexible enough to
 * support any learning algorithm that operates on a collection of cells.
 *
 * Each type of connection (proximal, distal basal, apical) should be
 * represented by a different instantiation of this class. This class
 * will help compute the activity along those connections due to active
 * input cells. The responsibility for what effect that activity has on
 * the cells and connections lies in the user of this class.
 *
 * This class is optimized to store connections between cells, and
 * compute the activity of cells due to input over the connections.
 *
 * This class assigns each segment a unique "flatIdx" so that it's
 * possible to use a simple vector to associate segments with values.
 * Create a vector of length `connections.segmentFlatListLength()`,
 * iterate over segments and update the vector at index `segment`.
 *
 */
class Connections : public Serializable
 {
public:
  static const UInt16 VERSION = 2;

  /**
   * Connections empty constructor.
   * (Does not call `initialize`.)
   */
  Connections(){};

  /**
   * Connections constructor.
   *
   * @param numCells           Number of cells.
   * @param connectedThreshold Permanence threshold for synapses connecting or
   *                           disconnecting.
   */
  Connections(CellIdx numCells, Permanence connectedThreshold = 0.5f);

  virtual ~Connections() {}

  /**
   * Initialize connections.
   *
   * @param numCells           Number of cells.
   * @param connectedThreshold Permanence threshold for synapses connecting or
   *                           disconnecting.
   */
  void initialize(CellIdx numCells, Permanence connectedThreshold = 0.5f);

  /**
   * Creates a segment on the specified cell.
   *
   * @param cell Cell to create segment on.
   *
   * @retval Created segment.
   */
  Segment createSegment(CellIdx cell);

  /**
   * Creates a synapse on the specified segment.
   *
   * @param segment         Segment to create synapse on.
   * @param presynapticCell Cell to synapse on.
   * @param permanence      Initial permanence of new synapse.
   *
   * @reval Created synapse.
   */
  Synapse createSynapse(Segment segment,
                        CellIdx presynapticCell,
                        Permanence permanence);

  /**
   * Destroys segment.
   *
   * @param segment Segment to destroy.
   */
  void destroySegment(Segment segment);

  /**
   * Destroys synapse.
   *
   * @param synapse Synapse to destroy.
   */
  void destroySynapse(Synapse synapse);

  /**
   * Updates a synapse's permanence.
   *
   * @param synapse    Synapse to update.
   * @param permanence New permanence.
   */
  void updateSynapsePermanence(Synapse synapse, Permanence permanence);

  /**
   * Gets the segments for a cell.
   *
   * @param cell Cell to get segments for.
   *
   * @retval Segments on cell.
   */
  const std::vector<Segment> &segmentsForCell(CellIdx cell) const;

  /**
   * Gets the synapses for a segment.
   *
   * @param segment Segment to get synapses for.
   *
   * @retval Synapses on segment.
   */
  const std::vector<Synapse> &synapsesForSegment(Segment segment) const;

  /**
   * Gets the cell that this segment is on.
   *
   * @param segment Segment to get the cell for.
   *
   * @retval Cell that this segment is on.
   */
  CellIdx cellForSegment(Segment segment) const;

  /**
   * Gets the index of this segment on its respective cell.
   *
   * @param segment Segment to get the idx for.
   *
   * @retval Index of the segment.
   */
  SegmentIdx idxOnCellForSegment(Segment segment) const;

  /**
   * Get the cell for each provided segment.
   *
   * @param segments
   * The segments to query
   *
   * @param cells
   * Output array with the same length as 'segments'
   */
  void mapSegmentsToCells(const Segment *segments_begin,
                          const Segment *segments_end,
                          CellIdx *cells_begin) const;

  /**
   * Gets the segment that this synapse is on.
   *
   * @param synapse Synapse to get Segment for.
   *
   * @retval Segment that this synapse is on.
   */
  Segment segmentForSynapse(Synapse synapse) const;

  /**
   * Gets the data for a segment.
   *
   * @param segment Segment to get data for.
   *
   * @retval Segment data.
   */
  const SegmentData &dataForSegment(Segment segment) const;

  /**
   * Gets the data for a synapse.
   *
   * @param synapse Synapse to get data for.
   *
   * @retval Synapse data.
   */
  const SynapseData &dataForSynapse(Synapse synapse) const;

  /**
   * Get the segment at the specified cell and offset.
   *
   * @param cell The cell that the segment is on.
   * @param idx The index of the segment on the cell.
   *
   * @retval Segment
   */
  Segment getSegment(CellIdx cell, SegmentIdx idx) const;

  /**
   * Get the vector length needed to use segments as indices.
   *
   * @retval A vector length
   */
  UInt32 segmentFlatListLength() const;

  /**
   * Compare two segments. Returns true if a < b.
   *
   * Segments are ordered first by cell, then by their order on the cell.
   *
   * @param a Left segment to compare
   * @param b Right segment to compare
   *
   * @retval true if a < b, false otherwise.
   */
  bool compareSegments(Segment a, Segment b) const;

  /**
   * Returns the synapses for the source cell that they synapse on.
   *
   * @param presynapticCell(int) Source cell index
   *
   * @return Synapse indices
   */
  std::vector<Synapse>
  synapsesForPresynapticCell(CellIdx presynapticCell) const;

  /**
   * Compute the segment excitations for a vector of active presynaptic
   * cells.
   *
   * The output vectors aren't grown or cleared. They must be
   * preinitialized with the length returned by
   * getSegmentFlatVectorLength().
   *
   * @param numActiveConnectedSynapsesForSegment
   * An output vector for active connected synapse counts per segment.
   *
   * @param numActivePotentialSynapsesForSegment
   * An output vector for active potential synapse counts per segment.
   *
   * @param activePresynapticCells
   * Active cells in the input.
   *
   * @param connectedPermanence
   * Minimum permanence for a synapse to be "connected".
   */
  void
  computeActivity(std::vector<UInt32> &numActiveConnectedSynapsesForSegment,
                  std::vector<UInt32> &numActivePotentialSynapsesForSegment,
                  const std::vector<CellIdx> &activePresynapticCells,
                  Permanence connectedPermanence) const;

  /**
   * Compute the segment excitations for a single active presynaptic cell.
   *
   * The output vectors aren't grown or cleared. They must be
   * preinitialized with the length returned by
   * getSegmentFlatVectorLength().
   *
   * @param numActiveConnectedSynapsesForSegment
   * An output vector for active connected synapse counts per segment.
   *
   * @param numActivePotentialSynapsesForSegment
   * An output vector for active potential synapse counts per segment.
   *
   * @param activePresynapticCells
   * Active cells in the input.
   *
   * @param connectedPermanence
   * Minimum permanence for a synapse to be "connected".
   */
  void
  computeActivity(std::vector<UInt32> &numActiveConnectedSynapsesForSegment,
                  std::vector<UInt32> &numActivePotentialSynapsesForSegment,
                  CellIdx activePresynapticCell,
                  Permanence connectedPermanence) const;

  /**
   * The primary method in charge of learning.   Adapts the permanence values of
   * the synapses based on the input SDR.  Learning is applied to a single
   * segment.  Permanence values are increased for synapses connected to input
   * bits that are turned on, and decreased for synapses connected to inputs
   * bits that are turned off.
   *
   * @param segment  Index of segment to apply learning to.  Is returned by 
   *        method getSegment.
   * @param inputVector  An SDR
   * @param increment  Change in permanence for synapses with active presynapses.
   * @param decrement  Change in permanence for synapses with inactive presynapses.
   */
  void adaptSegment(const Segment segment, SDR &inputs,
                    const Permanence increment,
                    const Permanence decrement);

  /**
   * Ensures a minimum number of connected synapses.  This raises permance
   * values until the desired number of synapses have permanences above the
   * permanenceThreshold.  This is applied to a single segment.
   *
   * @param segment  Index of segment on cell.   Is returned by method getSegment.
   * @param permanenceThreshold  Connected threshold of synapses
   * @param segmentThreshold  Desired number of connected synapses
   */
  void raisePermanencesToThreshold(const Segment    segment,
                                   const Permanence permanenceThreshold,
                                   const UInt       segmentThreshold);

  /**
   * Modify all permanence on the given segment, uniformly.
   *
   * @param segment  Index of segment on cell. Is returned by method getSegment.
   * @param delta  Change in permanence value
   */
  void bumpSegment(const Segment segment, const Permanence delta);

  // Serialization

  /**
   * Saves serialized data to output stream.
   */
  virtual void save(std::ostream &outStream) const override;


  /**
   * Loads serialized data from input stream.
   */
  virtual void load(std::istream &inStream) override;


  // Debugging

  /**
   * Gets the number of cells.
   *
   * @retval Number of cells.
   */
  CellIdx numCells() const;

  /**
   * Gets the number of segments.
   *
   * @retval Number of segments.
   */
  UInt numSegments() const;

  /**
   * Gets the number of segments on a cell.
   *
   * @retval Number of segments.
   */
  UInt numSegments(CellIdx cell) const;

  /**
   * Gets the number of synapses.
   *
   * @retval Number of synapses.
   */
  UInt numSynapses() const;

  /**
   * Gets the number of synapses on a segment.
   *
   * @retval Number of synapses.
   */
  UInt numSynapses(Segment segment) const;

  /**
   * Comparison operator.
   */
  bool operator==(const Connections &other) const;
  bool operator!=(const Connections &other) const;

  /**
   * Add a connections events handler.
   *
   * The Connections instance takes ownership of the eventHandlers
   * object. Don't delete it. When calling from Python, call
   * eventHandlers.__disown__() to avoid garbage-collecting the object
   * while this instance is still using it. It will be deleted on
   * `unsubscribe`.
   *
   * @param handler
   * An object implementing the ConnectionsEventHandler interface
   *
   * @retval Unsubscribe token
   */
  UInt32 subscribe(ConnectionsEventHandler *handler);

  /**
   * Remove an event handler.
   *
   * @param token
   * The return value of `subscribe`.
   */
  void unsubscribe(UInt32 token);

protected:
  /**
   * Gets the synapse with the lowest permanence on the segment.
   *
   * @param segment Segment whose synapses to consider.
   *
   * @retval Synapse with the lowest permanence.
   */
  Synapse minPermanenceSynapse_(Segment segment) const;

  /**
   * Check whether this segment still exists on its cell.
   *
   * @param Segment
   *
   * @retval True if it's still in its cell's segment list.
   */
  bool segmentExists_(Segment segment) const;

  /**
   * Check whether this synapse still exists on its segment.
   *
   * @param Synapse
   *
   * @retval True if it's still in its segment's synapse list.
   */
  bool synapseExists_(Synapse synapse) const;

  /**
   * Remove a synapse from synapsesForPresynapticCell_.
   *
   * @param Synapse
   */
  void removeSynapseFromPresynapticMap_(Synapse synapse);

private:
  std::vector<CellData>    cells_;
  std::vector<SegmentData> segments_;
  std::vector<Segment>     destroyedSegments_;
  std::vector<SynapseData> synapses_;
  std::vector<Synapse>     destroyedSynapses_;
  Permanence               connectedThreshold_;

  // Extra bookkeeping for faster computing of segment activity.
  std::map<CellIdx, std::vector<Synapse>> synapsesForPresynapticCell_;

  std::vector<UInt64> segmentOrdinals_;
  std::vector<UInt64> synapseOrdinals_;
  UInt64 nextSegmentOrdinal_;
  UInt64 nextSynapseOrdinal_;

  UInt32 nextEventToken_;
  std::map<UInt32, ConnectionsEventHandler *> eventHandlers_;
}; // end class Connections

} // end namespace connections

} // end namespace algorithms

} // end namespace nupic

#endif // NTA_CONNECTIONS_HPP
