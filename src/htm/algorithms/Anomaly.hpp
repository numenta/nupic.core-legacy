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

#ifndef HTM_ALGORITHMS_ANOMALY_HPP
#define HTM_ALGORITHMS_ANOMALY_HPP

#include <htm/types/Types.hpp>
#include <htm/types/Sdr.hpp> // sdr::SDR

namespace htm {


/**
 * Computes the raw anomaly score.
 *
 * The raw anomaly score is the fraction of active columns not predicted.
 * Do not use these methods directly, these are for testing and internal implementation. 
 * Use `TM.anomaly` (+AnomalyLikelihood, MovingAverage for more specific needs). 
 *
 * @param activeColumns: SDR with active columns (not cells) from current step (T)
 * @param prevPredictedColumns: SDR of predictive columns indices from prev step (T-1)
 * @return anomaly score 0..1 (Real32)
 */
Real32 computeRawAnomalyScore(const SDR& active, 
                              const SDR& predicted);

} //end-ns

#endif // HTM_ALGORITHMS_ANOMALY_HPP
