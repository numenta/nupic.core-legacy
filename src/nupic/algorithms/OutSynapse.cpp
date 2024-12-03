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

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/OutSynapse.hpp>
using namespace nupic::algorithms::Cells4;
using namespace nupic;

bool OutSynapse::invariants(Cells4 *cells) const {
  bool ok = true;
  if (cells) {
    ok &= _dstCellIdx < cells->nCells();
    ok &= _dstSegIdx < cells->__nSegmentsOnCell(_dstCellIdx);
  }
  return ok;
}

namespace nupic {
namespace algorithms {
namespace Cells4 {
bool operator==(const OutSynapse &a, const OutSynapse &b) {
  return a.equals(b);
}
} // namespace Cells4
} // namespace algorithms
} // namespace nupic
