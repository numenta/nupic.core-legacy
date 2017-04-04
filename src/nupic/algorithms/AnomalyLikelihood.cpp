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


#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#include <boost/circular_buffer.hpp>

#include <nupic/types/Types.hpp>
#include <nupic/utils/MovingAverage.hpp>

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


Raw functions
-------------
There are two lower level functions, estimateAnomalyLikelihoods and
updateAnomalyLikelihoods. The details of these are described below.

**/

namespace nupic {
  namespace algorithms { 
    namespace anomaly {

using namespace std;
using namespace nupic;
using namespace nupic::util;

struct DistributionParams {
  DistributionParams(string name, Real32 mean, Real32 variance, Real32 stdev) :
    name(name),mean(mean), variance(variance), stdev(stdev) {}
  string name;
  Real32 mean;
  Real32 variance;
  Real32 stdev;
};

class AnomalyLikelihood {
public:
    //Constructors

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
    AnomalyLikelihood(UInt learningPeriod=288, UInt estimationSamples=100, UInt historicWindowSize=8640, UInt reestimationPeriod=100) :
    learningPeriod(learningPeriod),
    reestimationPeriod(reestimationPeriod),
    historicalScores(historicWindowSize)  {
        iteration = 0;
        probationaryPeriod = learningPeriod+estimationSamples;
        assert(historicWindowSize >= estimationSamples); // cerr << "estimationSamples exceeds historicWindowSize";
    }

    
    Real anomalyProbability(Real rawValue, Real anomalyScore, int timestamp=-1) { //TODO "rawValue" could be typed to T rawValue
    /**
    Compute the probability that the current value plus anomaly score represents
    an anomaly given the historical distribution of anomaly scores. The closer
    the number is to 1, the higher the chance it is an anomaly.

    @param value - the current metric ("raw") input value, eg. "orange", or
                   '21.2' (deg. Celsius), ...
    @param anomalyScore - the current anomaly score
    @param timestamp - (optional) timestamp of the ocurrence,
                       default (-1) results in using iteration step.
    @return the anomalyLikelihood for this record.
    **/
    Real likelihood = 0.5f;

    if (timestamp == -1) {
      timestamp = this->iteration;
    }

    // We ignore the first probationaryPeriod data points - as we cannot reliably compute distribution statistics for estimating likelihood
    if (this->iteration < this->probationaryPeriod) {
      return 0.5f;
    } //else {

    vector<Real> likes;
      // On a rolling basis we re-estimate the distribution
      if ( this->iteration == 0 || (this->iteration % this->reestimationPeriod) == 0  || this->distribution.name == "unknown" ) {

        auto numSkipRecords = this->calcSkipRecords_(this->iteration, this->historicalScores.getMaxSize(), this->learningPeriod);
        likes = estimateAnomalyLikelihoods(this->historicalScores.getSlidingWindow(), numSkipRecords);  // updates this->distribution; 
      }

      auto likelihoods = updateAnomalyLikelihoods(likes, this->distribution); //TODO use likes or this->historicalScores.getSlidingWindow() here?

      assert(likelihoods.size() > 0); 
      likelihood = 1.0 - likelihoods[0]; 
      assert(likelihood >= 0.0 && likelihood <= 1.0);

    // Before we exit update historical scores and iteration
    // this->historicalScores.compute(anomalyScore);  - is already computed in updateAnomalyLikelihoods() above
    // this->runningAveragedAnomalies.append() - also already computed above
    this->iteration++;
    this->runningRawValues.push_back(rawValue); 
    this->historicalLikelihoods.push_back(likelihood);

    return likelihood;
    }
    
private:
    UInt learningPeriod = 288;
    UInt reestimationPeriod = 100;
  
    DistributionParams distribution ={ "unknown", 0.0, 0.0, 0.0};   
    UInt probationaryPeriod;
    UInt iteration;
    MovingAverage historicalScores; // running average of anomaly scores
    vector<Real> historicalLikelihoods; // sliding window of the likelihoods
    vector<Real> runningAverageAnomalies; //sliding window of running averages of anomaly scores
    vector<Real> runningRawValues; // sliding window of the raw values


/**
#
# USAGE FOR LOW-LEVEL FUNCTIONS
# -----------------------------
#
# There are two primary interface routines:
#
# estimateAnomalyLikelihoods: batch routine, called initially and once in a
#                                while
# updateAnomalyLikelihoods: online routine, called for every new data point
#
# 1. Initially::
#
#    likelihoods, avgRecordList, estimatorParams = \
# estimateAnomalyLikelihoods(metric_data)
#
# 2. Whenever you get new data::
#
#    likelihoods, avgRecordList, estimatorParams = \
# updateAnomalyLikelihoods(data2, estimatorParams)
#
# 3. And again (make sure you use the new estimatorParams returned in the above
#   call to updateAnomalyLikelihoods!)::
#
#    likelihoods, avgRecordList, estimatorParams = \
# updateAnomalyLikelihoods(data3, estimatorParams)
#
# 4. Every once in a while update estimator with a lot of recent data::
#
#    likelihoods, avgRecordList, estimatorParams = \
# estimateAnomalyLikelihoods(lots_of_metric_data)
#
#
# PARAMS
# ~~~~~~
#
# The parameters dict returned by the above functions has the following
# structure. Note: the client does not need to know the details of this.
#
# ::
#
#  {
#    "distribution":               # describes the distribution
#      {
#        "name": STRING,           # name of the distribution, such as 'normal'
#        "mean": SCALAR,           # mean of the distribution
#        "variance": SCALAR,       # variance of the distribution
#
#        # There may also be some keys that are specific to the distribution
#      },
#
#    "historicalLikelihoods": []   # Contains the last windowSize likelihood
#                                  # values returned
#
#    "movingAverage":              # stuff needed to compute a rolling average
#                                  # of the anomaly scores
#      {
#        "windowSize": SCALAR,     # the size of the averaging window
#        "historicalValues": [],   # list with the last windowSize anomaly
#                                  # scores
#        "total": SCALAR,          # the total of the values in historicalValues
#      },
#
#  }

**/
  /** returns parameters of a Null distribution **/
  DistributionParams nullDistribution() const {
    return DistributionParams("normal", 0.5, 1e6, 1e3);
  }

 Real32 tailProbability(Real32 x, DistributionParams params) {
 /**
  Given the normal distribution specified by the mean and standard deviation
  in distributionParams, return the probability of getting samples further
  from the mean. For values above the mean, this is the probability of getting
  samples > x and for values below the mean, the probability of getting
  samples < x. This is the Q-function: the tail probability of the normal distribution.

  :param distributionParams: dict with 'mean' and 'stdev' of the distribution
  **/
  if (x < params.mean) {
    // Gaussian is symmetrical around mean, so flip to get the tail probability
    Real32 xp = 2 * params.mean - x;
    return tailProbability(xp, params);
  }

  // Calculate the Q function with the complementary error function, explained
  // here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
  Real32 z = (x - params.mean) / params.stdev;
  return 0.5 * erfc(z/1.4142);
  }


inline Real  computeLogLikelihood(Real likelihood) const {
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


DistributionParams estimateNormal(vector<Real> sampleData, bool performLowerBoundCheck=true) {
  /** 
  :param sampleData:
  :type sampleData: Numpy array.
  :param performLowerBoundCheck:
  :type performLowerBoundCheck: bool
  :returns: A dict containing the parameters of a normal distribution based on
      the ``sampleData``.
  **/
    auto mean = compute_mean(sampleData);
    auto var = compute_var(sampleData, mean);
  DistributionParams params = DistributionParams("normal", mean, var, 0.0); 

  if (performLowerBoundCheck) {
    /* Handle edge case of almost no deviations and super low anomaly scores. We
     find that such low anomaly means can happen, but then the slightest blip
     of anomaly score can cause the likelihood to jump up to red.
     */
    if (params.mean < 0.03) {
      params.mean= 0.03;
    }

    // Catch all for super low variance to handle numerical precision issues
    if (params.variance < 0.0003) {
      params.variance = 0.0003;
    }
  }

  // Compute standard deviation
  if (params.variance > 0) {
    params.stdev = sqrt(params.variance);
  } else{
    params.stdev = 0;
  }

  return params;
}

Real compute_mean(vector<Real> v) const {
    Real sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

Real compute_var(vector<Real> v, Real mean) const {
    Real sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return sq_sum / v.size() - mean * mean;
}

vector<Real> filterLikelihoods_(vector<Real> likelihoods, Real redThreshold=0.99999, Real yellowThreshold=0.999) {
  /**
  Filter the list of raw (pre-filtered) likelihoods so that we only preserve
  sharp increases in likelihood. 'likelihoods' can be a numpy array of floats or
  a list of floats.

  :returns: A new list of floats likelihoods containing the filtered values.
  **/
  redThreshold    = 1.0 - redThreshold;
  yellowThreshold = 1.0 - yellowThreshold;

  // The first value is untouched
  vector<Real> filteredLikelihoods; 
  filteredLikelihoods.push_back(likelihoods.front()); 

  for (Real v: likelihoods) { 
   
    if (v <= redThreshold) { //TODO review the IFs
      // Value is in the redzone
        
      if (filteredLikelihoods.back() > redThreshold) {
        // Previous value is not in redzone, so leave as-is
          filteredLikelihoods.push_back(v);
      } else { 
        filteredLikelihoods.push_back(yellowThreshold);
      }
    } else {
      // Value is below the redzone, so leave as-is
      filteredLikelihoods.push_back(v);
    }
  }
  return filteredLikelihoods;
}


inline UInt calcSkipRecords_(UInt numIngested, UInt windowSize, UInt learningPeriod) const {
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


vector<Real>  updateAnomalyLikelihoods(vector<Real> anomalyScores, DistributionParams params, UInt verbosity=0) {
  /**
  Compute updated probabilities for anomalyScores using the given params.

  :param anomalyScores: a list of records. Each record is a list with the
                        following three elements: [timestamp, value, score]

                        Example::

                            [datetime.datetime(2013, 8, 10, 23, 0), 6.0, 1.0]

  :param params: the JSON dict returned by estimateAnomalyLikelihoods
  :param verbosity: integer controlling extent of printouts for debugging
  :type verbosity: int

  :returns: 3-tuple consisting of:

            - likelihoods

              numpy array of likelihoods, one for each aggregated point

            - avgRecordList 

              list of averaged input records - stored in runningAverageAnomalies instance member

            - params

              an updated JSON object containing the state of this metric.-- available from historicalScores.getXXX()

  **/
  if (verbosity > 3) {
    cout << "In updateAnomalyLikelihoods."<< endl;
    cout << "Number of anomaly scores: "<<  anomalyScores.size() << endl;
//    cout << "First 20:", anomalyScores[0:min(20, len(anomalyScores))])
    cout << "Params: name=" <<  params.name << " mean="<<params.mean <<" var="<<params.variance <<" stdev="<<params.stdev <<endl;
  }

 assert(anomalyScores.size() > 0); // "Must have at least one anomalyScore"

  // Compute moving averages of these new scores using the previous values
  // as well as likelihood for these scores using the old estimator 
  vector<Real> likelihoods;
  for (auto v : anomalyScores) {
    auto newAverage = historicalScores.compute(v); 
    this->runningAverageAnomalies.push_back(newAverage);
    likelihoods.push_back(tailProbability(newAverage, params));
  }

  // Filter the likelihood values. First we prepend the historical likelihoods
  // to the current set. Then we filter the values.  We peel off the likelihoods
  // to return and the last windowSize values to store for later.
  this->historicalLikelihoods.insert(historicalLikelihoods.end(), likelihoods.begin(),likelihoods.end()); //append
  UInt toCrop = max((unsigned long)0, min((unsigned long)this->historicalScores.getMaxSize(), historicalLikelihoods.size())); //TODO review, the max(0, does not make sense here
  this->historicalLikelihoods.erase(historicalLikelihoods.begin(), historicalLikelihoods.begin() + toCrop); 
  assert(this->historicalLikelihoods.size() <= this->historicalScores.getMaxSize());

  auto filteredLikelihoods = filterLikelihoods_(likelihoods);

  if (verbosity > 3) {
    cout << "Number of likelihoods:"<< likelihoods.size() << endl;
//    print("First 20 likelihoods:", likelihoods[0:min(20, len(likelihoods))])
    cout << "Leaving updateAnomalyLikelihoods."<< endl;
  }

  return filteredLikelihoods;
}



vector<Real> estimateAnomalyLikelihoods(vector<Real> anomalyScores, UInt averagingWindow=10, UInt skipRecords=0, UInt verbosity=0) {
  /**
  Given a series of anomaly scores, compute the likelihood for each score. This
  function should be called once on a bunch of historical anomaly scores for an
  initial estimate of the distribution. It should be called again every so often
  (say every 50 records) to update the estimate.

  :param anomalyScores: a list of records. Each record is a list with the
                        following three elements: [timestamp, value, score]

                        Example::

                            [datetime.datetime(2013, 8, 10, 23, 0), 6.0, 1.0]

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

  :returns: 3-tuple consisting of:

            - likelihoods

              numpy array of likelihoods, one for each aggregated point

            - avgRecordList

              list of averaged input records

            - params
              a small JSON dict that contains the state of the estimator

  **/
  if (verbosity > 1) {
    cout << "In estimateAnomalyLikelihoods."<<endl;
    cout << "Number of anomaly scores:" <<  anomalyScores.size() << endl;
    cout << "Skip records="<<  skipRecords << endl;
//    print("First 20:", anomalyScores[0:min(20, len(anomalyScores))])
  }

  assert(anomalyScores.size() >= 0); // "Must have at least one anomalyScore"
  auto dataValues = historicalScores.getSlidingWindow();  //FIXME the "data" for distrib estimation - should it be raw values, or anomaly scores?

  // Estimate the distribution of anomaly scores based on aggregated records
  if (dataValues.size()  <= skipRecords) {
    this->distribution = nullDistribution();
  } else {
    dataValues.erase(dataValues.begin(), dataValues.begin() + skipRecords);// remove first skipRecords 
    this->distribution = estimateNormal(dataValues);
  
    /* HACK ALERT! The CLA model currently does not handle constant metric values //FIXME - this is unrelated to anomalies and should be removed from this code!!
     very well (time of day encoder changes sometimes lead to unstable SDR's
     even though the metric is constant). Until this is resolved, we explicitly
     detect and handle completely flat metric values by reporting them as not
     anomalous.
    */
      auto metricDistribution = estimateNormal(dataValues, false);
      if (metricDistribution.variance < 1.5e-5) {
        this->distribution = nullDistribution();
      } //end of hack
  }

  // Estimate likelihoods based on this distribution
  vector<Real> likelihoods;
  for (auto s : dataValues) {
    likelihoods.push_back(tailProbability(s, distribution));
  }
  // Filter likelihood values
  auto filteredLikelihoods = filterLikelihoods_(likelihoods);


/*  if verbosity > 1:
    print("Discovered params=")
    print(params)
    print("Number of likelihoods:", len(likelihoods))
    print("First 20 likelihoods:", (
      filteredLikelihoods[0:min(20, len(filteredLikelihoods))] ))
    print("leaving estimateAnomalyLikelihoods")
*/

  return filteredLikelihoods;
}

};
}}} //ns