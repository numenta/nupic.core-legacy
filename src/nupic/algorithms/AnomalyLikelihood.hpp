#ifndef NUPIC_ALGORITHMS_ANOMALY_LIKELIHOOD_HPP_
#define NUPIC_ALGORITHMS_ANOMALY_LIKELIHOOD_HPP_

#include <nupic/types/Types.hpp>
#include <nupic/utils/MovingAverage.hpp>
#include <nupic/utils/SlidingWindow.hpp>
#include <nupic/utils/Log.hpp>

#include <string>
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
  DistributionParams(std::string name, Real mean, Real variance, Real stdev) :
    name(name),mean(mean), variance(variance), stdev(stdev) {}
  std::string name;
  Real mean;
  Real variance;
  Real stdev;
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


  /**
    Compute a log scale representation of the likelihood value. Since the
    likelihood computations return low probabilities that often go into four 9's
    or five 9's, a log value is more useful for visualization, thresholding,
    etc.
   **/
  static Real  computeLogLikelihood(Real likelihood)  { //public - for visualizations, debug,...
    // The log formula is:
    //     Math.log(1.0000000001 - likelihood) / Math.log(1.0 - 0.9999999999)
    return log(1.0000000001f - likelihood) / -23.02585084720009f;
  }


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

    const UInt learningPeriod; //these 3 are from constructor
    const UInt reestimationPeriod;
    const UInt probationaryPeriod;

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
    vector<Real> estimateAnomalyLikelihoods_(const vector<Real>& anomalyScores, UInt skipRecords=0, UInt verbosity=0);


  /**
  Compute updated probabilities for anomalyScores using the given params.
  :param anomalyScores: a list of records. Each record is a list with the
  :param verbosity: integer controlling extent of printouts for debugging
  :type verbosity: UInt
  :returns: a vector of likelihoods, one for each aggregated point
  **/
    vector<Real>  updateAnomalyLikelihoods_(const vector<Real>& anomalyScores, UInt verbosity=0);


 /**
  Given the normal distribution specified by the mean and standard deviation
  in distributionParams (the distribution is an instance member of the class), 
  return the probability of getting samples further from the mean. 
  For values above the mean, this is the probability of getting
  samples > x and for values below the mean, the probability of getting
  samples < x. This is the Q-function: the tail probability of the normal distribution.
  **/
    Real tailProbability_(Real x) const;


  /**
  :param anomalyScores: vector of (raw) anomaly scores
  :param performLowerBoundCheck (bool)
  :returns: A DistributionParams (struct) containing the parameters of a normal distribution based on
      the ``anomalyScores``.
  **/
    DistributionParams estimateNormal_(const vector<Real>& anomalyScores, bool performLowerBoundCheck=true);


    //private variables
    DistributionParams distribution_ ={ "unknown", 0.0, 0.0, 0.0}; //distribution passed around the class

    UInt iteration_; 
    int lastTimestamp_ = -1;  //helper for time measurements
    int initialTimestamp_ = -1;

    nupic::util::MovingAverage averagedAnomaly_; // running average of anomaly scores
    nupic::util::SlidingWindow<Real> runningLikelihoods_; // sliding window of the likelihoods
    nupic::util::SlidingWindow<Real> runningRawAnomalyScores_;
    nupic::util::SlidingWindow<Real> runningAverageAnomalies_; //sliding window of running averages of anomaly scores

};

}}} //end-ns
#endif
