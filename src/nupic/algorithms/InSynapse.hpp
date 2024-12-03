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

#ifndef NTA_INSYNAPSE_HPP
#define NTA_INSYNAPSE_HPP

#include <nupic/types/Types.hpp>

#include <fstream>
#include <ostream>

using namespace nupic;

//--------------------------------------------------------------------------------

namespace nupic {
namespace algorithms {
namespace Cells4 {

//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
/**
 * The type of synapse contained in a Segment. It has the source cell index
 * of the synapse, and a permanence value. The source cell index is between
 * 0 and nCols * nCellsPerCol.
 */
class InSynapse {
private:
  UInt _srcCellIdx;
  Real _permanence;

public:
  inline InSynapse() : _srcCellIdx((UInt)-1), _permanence(0) {}

  inline InSynapse(UInt srcCellIdx, Real permanence)
      : _srcCellIdx(srcCellIdx), _permanence(permanence) {}

  inline InSynapse(const InSynapse &o)
      : _srcCellIdx(o._srcCellIdx), _permanence(o._permanence) {}

  inline InSynapse &operator=(const InSynapse &o) {
    _srcCellIdx = o._srcCellIdx;
    _permanence = o._permanence;
    return *this;
  }

  inline bool operator==(const InSynapse &other) const {
    return _srcCellIdx == other._srcCellIdx && _permanence == other._permanence;
  }
  inline bool operator!=(const InSynapse &other) const {
    return !operator==(other);
  }

  inline UInt srcCellIdx() const { return _srcCellIdx; }
  const inline Real &permanence() const { return _permanence; }
  inline Real &permanence() { return _permanence; }

  inline void print(std::ostream &outStream) const;
};

//--------------------------------------------------------------------------------
#ifndef SWIG
std::ostream &operator<<(std::ostream &outStream, const InSynapse &s);
#endif

// end namespace
} // namespace Cells4
} // namespace algorithms
} // namespace nupic

#endif // NTA_INSYNAPSE_HPP
