/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
 * ---------------------------------------------------------------------
 */

#include <algorithm>
#include <iterator>
#include <numeric>
#include <set>

#include "nupic/algorithms/Anomaly.hpp"
#include "nupic/utils/Log.hpp"

using namespace std;
using namespace nupic;
using namespace nupic::algorithms::anomaly;
using namespace nupic::sdr;

namespace nupic {
namespace algorithms {
namespace anomaly {

Real computeRawAnomalyScore(const SDR& active,
                            const SDR& predicted) {

  // Return 0 if no active columns are present
  if (active.getSum() == 0) {
    return 0.0f;
  }

  NTA_CHECK(active.dimensions == predicted.dimensions);

  // Calculate and return percent of active columns that were not predicted.
  SDR both(active.dimensions);
  both.intersection(active, predicted);

  return (active.getSum() - both.getSum()) / Real(active.getSum());
}

Real computeRawAnomalyScore(vector<UInt>& active,
                            vector<UInt>& predicted)
{
  // Don't divide by zero.  Return 0 if no active columns are present.
  if (active.size() == 0) {
    return 0.0f;
  }

  vector<UInt> correctPredictions;
  sort( active.begin(),    active.end());
  sort( predicted.begin(), predicted.end());
  set_intersection(active.begin(), active.end(),
                   predicted.begin(), predicted.end(),
                   back_inserter( correctPredictions ));

  return (Real) (active.size() - correctPredictions.size()) / active.size();
}

}}} // End namespace
