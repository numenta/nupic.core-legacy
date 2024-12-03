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

#include <iomanip>
#include <nupic/algorithms/InSynapse.hpp>

using namespace nupic::algorithms::Cells4;

inline void InSynapse::print(std::ostream &outStream) const {
  outStream << _srcCellIdx << ',' << std::setprecision(4) << _permanence;
}

//--------------------------------------------------------------------------------

namespace nupic {
namespace algorithms {
namespace Cells4 {

std::ostream &operator<<(std::ostream &outStream, const InSynapse &s) {
  s.print(outStream);
  return outStream;
}
} // namespace Cells4
} // namespace algorithms
} // namespace nupic
