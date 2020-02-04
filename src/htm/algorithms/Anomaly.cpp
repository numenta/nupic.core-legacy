/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

#include "htm/algorithms/Anomaly.hpp"
#include "htm/utils/Log.hpp"

using namespace std;
using namespace htm;

namespace htm {

Real computeRawAnomalyScore(const SDR& active,
                            const SDR& predicted) {

  NTA_ASSERT(active.dimensions == predicted.dimensions);

  // Return 0 if no active columns are present
  if (active.getSum() == 0) {
    return static_cast<Real>(0);
  }

  // Calculate and return percent of active columns that were not predicted.
  SDR both(active.dimensions);
  both.intersection(active, predicted);

  const Real score = (active.getSum() - both.getSum()) / static_cast<Real>(active.getSum());
  NTA_ASSERT(score >= 0.0f and score <= 1.0f) << "Anomaly score out of bounds!";
  return score;
}

} // End namespace
