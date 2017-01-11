/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
 * Declaration of SparseMatrixConnections class
 */

#ifndef NTA_SPARSE_MATRIX_CONNECTIONS_HPP
#define NTA_SPARSE_MATRIX_CONNECTIONS_HPP

#include <nupic/math/SegmentMatrixAdapter.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/utils/Random.hpp>

namespace nupic {

  /**
   * Wraps the SparseMatrix with an easy-to-read API that stores dendrite
   * segments as rows in the matrix.
   *
   * The internal SparseMatrix is part of the public API. It is exposed via the
   * "matrix" member variable.
   */
  class SparseMatrixConnections :
    public SegmentMatrixAdapter<SparseMatrix<UInt32, Real32, Int32, Real64>> {

  public:
    /**
     * SparseMatrixConnections constructor
     *
     * @param numCells
     * The number of cells in this Connections
     *
     * @param numInputs
     * The number of input bits, i.e. the number of columns in the internal
     * SparseMatrix
     */
    SparseMatrixConnections(UInt32 numCells, UInt32 numInputs);

    /**
     * Compute the number of active synapses on each segment.
     *
     * @param activeInputs
     * The active input bits
     *
     * @param overlaps
     * Output buffer that will be filled with a number of active synapses for
     * each segment. This number is the "overlap" between the input SDR and the
     * SDR formed by each segment's synapses.
     */
    void computeActivity(
      const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
      Int32* overlaps_begin) const;

    /**
     * Compute the number of active connected synapses on each segment.
     *
     * @param activeInputs
     * The active input bits
     *
     * @param permanenceThreshold
     * The minimum permanence required for a synapse to be "connected"
     *
     * @param overlaps
     * Output buffer that will be filled with a number of active connected
     * synapses for each segment. This number is the "overlap" between the input
     * SDR and the SDR formed by each segment's connected synapses.
     */
    void computeActivity(
      const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
      Real32 permanenceThreshold, Int32* overlaps_begin) const;

    /**
     * For each specified segment, update the permanence of each synapse
     * according to whether the synapse would be active given the specified
     * active inputs.
     *
     * @param segments
     * The segments to modify
     *
     * @param activeInputs
     * The active inputs. Used to compute the active synapses.
     *
     * @param activePermanenceDelta
     * Additive constant for each active synapse's permanence
     *
     * @param inactivePermanenceDelta
     * Additive constant for each inactive synapse's permanence
     */
    void adjustSynapses(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
      Real32 activePermanenceDelta, Real32 inactivePermanenceDelta);

    /**
     * For each specified segment, add a delta to the permanences of the
     * synapses that would be active given the specified active inputs.
     *
     * @param segments
     * The segments to modify
     *
     * @param activeInputs
     * The active inputs. Used to compute the active synapses.
     *
     * @param permanenceDelta
     * Additive constant for each active synapse's permanence
     */
    void adjustActiveSynapses(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
      Real32 permanenceDelta);

    /**
     * For each specified segment, add a delta to the permanences of the
     * synapses that would be inactive given the specified active inputs.
     *
     * @param segments
     * The segments to modify
     *
     * @param activeInputs
     * The active inputs. Used to compute the active synapses.
     *
     * @param permanenceDelta
     * Additive constant for each inactive synapse's permanence
     */
    void adjustInactiveSynapses(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* activeInputs_begin, const UInt32* activeInputs_end,
      Real32 permanenceDelta);

    /**
     * For each specified segments, grow synapses to all specified inputs that
     * aren't already connected to the segment.
     *
     * @param segments
     * The segments to modify
     *
     * @param inputs
     * The inputs to connect to
     *
     * @param initialPermanence
     * The permanence for each added synapse
     */
    void growSynapses(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* inputs_begin, const UInt32* inputs_end,
      Real32 initialPermanence);

    /**
     * For each specified segments, grow synapses to a random subset of the
     * inputs that aren't already connected to the segment.
     *
     * @param segments
     * The segments to modify
     *
     * @param inputs
     * The inputs to sample
     *
     * @param sampleSize
     * The number of synapses to attempt to grow per segment
     *
     * @param initialPermanence
     * The permanence for each added synapse
     *
     * @param rng
     * Random number generator
     */
    void growSynapsesToSample(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* inputs_begin, const UInt32* inputs_end,
      Int32 sampleSize, Real32 initialPermanence, nupic::Random& rng);

    /**
     * For each specified segments, grow synapses to a random subset of the
     * inputs that aren't already connected to the segment.
     *
     * @param segments
     * The segments to modify
     *
     * @param inputs
     * The inputs to sample
     *
     * @param sampleSizes
     * The number of synapses to attempt to grow for each segment.
     * This list must be the same length as 'segments'.
     *
     * @param initialPermanence
     * The permanence for each added synapse
     *
     * @param rng
     * Random number generator
     */
    void growSynapsesToSample(
      const UInt32* segments_begin, const UInt32* segments_end,
      const UInt32* inputs_begin, const UInt32* inputs_end,
      const Int32* sampleSizes_begin, const Int32* sampleSizes_end,
      Real32 initialPermanence, nupic::Random& rng);

    /**
     * Clip all permanences to a minimum of 0.0 and a maximum of 1.0.
     * For any synapse with <= 0.0 permanence, destroy the synapse.
     * For any synapse with > 1.0 permanence, set the permanence to 1.0.
     *
     * @param segments
     * The segments to modify
     */
    void clipPermanences(
      const UInt32* segments_begin, const UInt32* segments_end);

    /**
     * Get the number of synapses for each specified segment.
     *
     * @param segments
     * The segments to query
     *
     * @param out
     * An output buffer that will be filled with the counts
     */
    void mapSegmentsToSynapseCounts(
      const UInt32* segments_begin, const UInt32* segments_end,
      Int32* out_begin) const;
  };

};

#endif // NTA_SPARSE_MATRIX_CONNECTIONS_HPP
