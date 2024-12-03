/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef NTA_OUTSYNAPSE_HPP
#define NTA_OUTSYNAPSE_HPP

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp> // NTA_ASSERT

using namespace nupic;

namespace nupic {
namespace algorithms {
namespace Cells4 {

class Cells4;
//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
/**
 * The type of synapse we use to propagate activation forward. It contains
 * indices for the *destination* cell, and the *destination* segment on that
 * cell. The cell index is between 0 and nCols * nCellsPerCol.
 */
class OutSynapse {
public:
private:
  UInt _dstCellIdx;
  UInt _dstSegIdx; // index in _segActivity

public:
  OutSynapse(UInt dstCellIdx = (UInt)-1, UInt dstSegIdx = (UInt)-1
             // Cells4* cells =NULL
             )
      : _dstCellIdx(dstCellIdx), _dstSegIdx(dstSegIdx) {
    // TODO: FIX this
    // NTA_ASSERT(invariants(cells));
  }

  OutSynapse(const OutSynapse &o)
      : _dstCellIdx(o._dstCellIdx), _dstSegIdx(o._dstSegIdx) {}

  OutSynapse &operator=(const OutSynapse &o) {
    _dstCellIdx = o._dstCellIdx;
    _dstSegIdx = o._dstSegIdx;
    return *this;
  }

  UInt dstCellIdx() const { return _dstCellIdx; }
  UInt dstSegIdx() const { return _dstSegIdx; }

  /**
   * Checks whether this outgoing synapses is going to given destination
   * or not.
   */
  bool goesTo(UInt dstCellIdx, UInt dstSegIdx) const {
    return _dstCellIdx == dstCellIdx && _dstSegIdx == dstSegIdx;
  }

  /**
   * Need for is_in/not_in tests.
   */
  bool equals(const OutSynapse &o) const {
    return _dstCellIdx == o._dstCellIdx && _dstSegIdx == o._dstSegIdx;
  }

  /**
   * Checks that the destination cell index and destination segment index
   * are in range.
   */
  bool invariants(Cells4 *cells = nullptr) const;
};

//--------------------------------------------------------------------------------
bool operator==(const OutSynapse &a, const OutSynapse &b);

// End namespace
} // namespace Cells4
} // namespace algorithms
} // namespace nupic

#endif // NTA_OUTSYNAPSE_HPP
