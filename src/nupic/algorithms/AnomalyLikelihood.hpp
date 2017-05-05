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
#include <nupic/utils/Log.hpp>

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

There are 3 ways to use the code: 
- using the convenience Anomaly class with `mode=LIKELIHOOD` (method `compute()`),
- using the AnomalyLikelihood helper class (method `anomalyProbability()`), or
- using the raw individual functions.
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

  /**
      NOTE: Anomaly likelihood scores are reported at a flat 0.5 for
    learningPeriod + estimationSamples iterations.

    claLearningPeriod and learningPeriod are specifying the same variable,
    although claLearningPeriod is a deprecated name for it.

    @param learningPeriod (claLearningPeriod: deprecated) - (int) the number of
      iterations required for the algorithm to learn the basic patterns in the
      dataset and for the anomaly score to 'settle down'. The default is based
      on empirical observations but in reality this could be larger for more
      complex domains. The downside if this is too large is that real anomalies
      might get ignored and not flagged.

    @param estimationSamples - (int) the number of reasonable anomaly scores
      required for the initial estimate of the Gaussian. The default of 100
      records is reasonable - we just need sufficient samples to get a decent
      estimate for the Gaussian. It's unlikely you will need to tune this since
      the Gaussian is re-estimated every 10 iterations by default.

    @param historicWindowSize - (int) size of sliding window of historical
      data points to maintain for periodic reestimation of the Gaussian. Note:
      the default of 8640 is based on a month's worth of history at 5-minute
      intervals.

    @param reestimationPeriod - (int) how often we re-estimate the Gaussian
      distribution. The ideal is to re-estimate every iteration but this is a
      performance hit. In general the system is not very sensitive to this
      number as long as it is small relative to the total number of records
      processed.

  **/
    AnomalyLikelihood(UInt learningPeriod=288, UInt estimationSamples=100, UInt historicWindowSize=8640, UInt reestimationPeriod=100, UInt aggregationWindow=10);


    /**
    This is the main "compute" method. 

    Compute the probability that the current anomaly score represents
    an anomaly given the historical distribution of anomaly scores. The closer
    the number is to 1, the higher the chance it is an anomaly.

    @param anomalyScore - the current anomaly score
    @param timestamp - (optional) timestamp of the ocurrence,
                       default (-1) results in using iteration step.
    @return the anomalyLikelihood for this record.
    **/
    Real anomalyProbability(Real anomalyScore, int timestamp=-1);

  //public constants:
  /** "neutral" anomalous value; 
    * returned at the beginning until the system is burned-in; 
    * 0.5 (from <0..1>) means "neither anomalous, neither expected"
    */
    const Real DEFAULT_ANOMALY = 0.5; 
    /**
     * minimal thresholds of standard distribution, if values get lower (rounding err, constant values)
     * we round to these minimal defaults
     */
    const Real THRESHOLD_MEAN = 0.03;
    const Real THRESHOLD_VARIANCE = 0.0003; 

  private:
    //methods:

  /**
  Given a series of anomaly scores, compute the likelihood for each score. This
  function should be called once on a bunch of historical anomaly scores for an
  initial estimate of the distribution. It should be called again every so often
  (say every 50 records) to update the estimate.

  :param anomalyScores: a list of anomaly scores

                        For best results, the list should be between 1000
                        and 10,000 records
  :param averagingWindow: integer number of records to average over
  :param skipRecords: integer specifying number of records to skip when
                      estimating distributions. If skip records are >=
                      len(anomalyScores), a very broad distribution is returned
                      that makes everything pretty likely.
  :param verbosity: integer controlling extent of printouts for debugging

                      0 = none
                      1 = occasional information
                      2 = print every record

  :returns: vector of lilelihoods: , one for each aggregated point
  **/
    vector<Real> estimateAnomalyLikelihoods(vector<Real> anomalyScores, UInt skipRecords=0, UInt verbosity=0);

  /**
  Compute updated probabilities for anomalyScores using the given params.

  :param anomalyScores: a list of records. Each record is a list with the

  :param verbosity: integer controlling extent of printouts for debugging
  :type verbosity: UInt

  :returns: a vector of likelihoods, one for each aggregated point
  **/
    vector<Real>  updateAnomalyLikelihoods(vector<Real> anomalyScores, UInt verbosity=0);
 /**
  Given the normal distribution specified by the mean and standard deviation
  in distributionParams (the distribution is an instance member of the class), 
  return the probability of getting samples further from the mean. 
  For values above the mean, this is the probability of getting
  samples > x and for values below the mean, the probability of getting
  samples < x. This is the Q-function: the tail probability of the normal distribution.
  **/
    Real32 tailProbability(Real32 x) const;
  /**
  :param sampleData:
  :type sampleData: vector array //TODO of what? likelihoods? rawScores? ....?
  :param performLowerBoundCheck:
  :type performLowerBoundCheck: bool
  :returns: A DistributionParams (struct) containing the parameters of a normal distribution based on
      the ``sampleData``.
  **/
    DistributionParams estimateNormal(vector<Real> sampleData, bool performLowerBoundCheck=true);
  /** returns parameters of a Null distribution **/
    DistributionParams nullDistribution() const {
    return DistributionParams("normal", 0.5, 1e6, 1e3);
}

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

  NTA_ASSERT(data.size() == cb.size() && data.front() == cb.front() && data.back() == cb.back() );
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
