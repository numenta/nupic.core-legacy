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
#ifndef NUPIC_ALGORITHMS_ANOMALY_LIKELIHOOD_HPP_
#define NUPIC_ALGORITHMS_ANOMALY_LIKELIHOOD_HPP_

#include <nupic/types/Types.hpp>

#include <string>
/**
Note: this is an implementation from python in nupic repository.

This module analyzes and estimates the distribution of averaged anomaly scores
from a given model. Given a new anomaly score `s`, estimates `P(score >= s)`.

The number `P(score >= s)` represents the likelihood of the current state of
predictability. For example, a likelihood of 0.01 or 1% means we see this much
predictability about one out of every 100 records. The number is not as unusual
as it seems. For records that arrive every minute, this means once every hour
and 40 minutes. A likelihood of 0.0001 or 0.01% means we see it once out of
10,000 records, or about once every 7 days.

USAGE
-----

There are two ways to use the code: using the AnomalyLikelihood helper class or
using the raw individual functions.


Helper Class
------------
The helper class AnomalyLikelihood is the easiest to use.  To use it simply
create an instance and then feed it successive anomaly scores:

anomalyLikelihood = AnomalyLikelihood()
while still_have_data:
  # Get anomaly score from model

  # Compute probability that an anomaly has ocurred
  anomalyProbability = anomalyLikelihood.anomalyProbability(
      value, anomalyScore, timestamp)

**/

namespace nupic {
  namespace algorithms {
    namespace anomaly {

class AnomalyLikelihood {
  public:
    AnomalyLikelihood(UInt learningPeriod=288, UInt estimationSamples=100, UInt historicWindowSize=8640, UInt reestimationPeriod=100, UInt aggregationWindow=10);

    Real anomalyProbability(Real rawValue, Real anomalyScore, int timestamp=-1);
};

struct DistributionParams {
  DistributionParams(std::string name, Real32 mean, Real32 variance, Real32 stdev) :
    name(name),mean(mean), variance(variance), stdev(stdev) {}
  std::string name;
  Real32 mean;
  Real32 variance;
  Real32 stdev;
};

}}} //end-ns
#endif
