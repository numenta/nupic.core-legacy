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
#include <nupic/utils/MovingAverage.hpp>

#include <string>
#include <boost/circular_buffer.hpp>
#include <vector>
#include <cmath>

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

using namespace std;

namespace nupic {
  namespace algorithms {
    namespace anomaly {

struct DistributionParams {
  DistributionParams(std::string name, Real32 mean, Real32 variance, Real32 stdev) :
    name(name),mean(mean), variance(variance), stdev(stdev) {}
  std::string name;
  Real32 mean;
  Real32 variance;
  Real32 stdev;
};

class AnomalyLikelihood {

  public:
    AnomalyLikelihood(UInt learningPeriod=288, UInt estimationSamples=100, UInt historicWindowSize=8640, UInt reestimationPeriod=100, UInt aggregationWindow=10);

    Real anomalyProbability(Real anomalyScore, int timestamp=-1);

  private:
    //methods:
    vector<Real> estimateAnomalyLikelihoods(vector<Real> anomalyScores, UInt skipRecords=0, UInt verbosity=0);
    vector<Real>  updateAnomalyLikelihoods(vector<Real> anomalyScores, UInt verbosity=0);
    Real32 tailProbability(Real32 x) const;
    DistributionParams estimateNormal(vector<Real> sampleData, bool performLowerBoundCheck=true);
    DistributionParams nullDistribution() const;

    //private static methods
static Real  computeLogLikelihood(Real likelihood)  {
  /**
    Compute a log scale representation of the likelihood value. Since the
    likelihood computations return low probabilities that often go into four 9's
    or five 9's, a log value is more useful for visualization, thresholding,
    etc.
   **/
    // The log formula is:
    //     Math.log(1.0000000001 - likelihood) / Math.log(1.0 - 0.9999999999)
    return log(1.0000000001f - likelihood) / -23.02585084720009f;
}

static UInt calcSkipRecords_(UInt numIngested, UInt windowSize, UInt learningPeriod)  {
    /** Return the value of skipRecords for passing to estimateAnomalyLikelihoods

    If `windowSize` is very large (bigger than the amount of data) then this
    could just return `learningPeriod`. But when some values have fallen out of
    the historical sliding window of anomaly records, then we have to take those
    into account as well so we return the `learningPeriod` minus the number
    shifted out.

    @param numIngested - (int) number of data points that have been added to the
      sliding window of historical data points.
    @param windowSize - (int) size of sliding window of historical data points.
    @param learningPeriod - (int) the number of iterations required for the
      algorithm to learn the basic patterns in the dataset and for the anomaly
      score to 'settle down'.
    **/
    int diff = numIngested - (int)windowSize;
    UInt numShiftedOut = max(0, diff);
    return min(numIngested, max((UInt)0, learningPeriod - numShiftedOut));
}


static std::vector<Real> circularBufferToVector(boost::circular_buffer<Real> cb) {
  cb.linearize();
  auto d1 = cb.array_one();
  vector<Real> data(d1.first, d1.first+d1.second);
  auto d2 = cb.array_two();
  data.insert(end(data), d2.first, d2.first+d2.second);

  assert(data.size() == cb.size() && data.front() == cb.front() && data.back() == cb.back() );
  return data;
}


    // variables
    UInt learningPeriod = 288;
    UInt reestimationPeriod = 100;

    DistributionParams distribution ={ "unknown", 0.0, 0.0, 0.0};
    UInt probationaryPeriod;
    UInt iteration;
    nupic::util::MovingAverage averagedAnomaly; // running average of anomaly scores
    boost::circular_buffer<Real> runningLikelihoods; // sliding window of the likelihoods
    boost::circular_buffer<Real> runningRawAnomalyScores;
    boost::circular_buffer<Real> runningAverageAnomalies; //sliding window of running averages of anomaly scores

};

}}} //end-ns
#endif
