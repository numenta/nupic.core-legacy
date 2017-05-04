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

#include <nupic/algorithms/AnomalyLikelihood.hpp>

#include <iostream>
#include <numeric> //accumulate, inner_product

#include <nupic/utils/Log.hpp> // NTA_CHECK

using namespace std;
using namespace nupic;
using namespace nupic::util;
using namespace nupic::algorithms::anomaly;

namespace nupic {
  namespace algorithms {
    namespace anomaly {

      Real compute_mean(vector<Real> v); //forward declaration
      Real compute_var(vector<Real> v, Real mean); 

AnomalyLikelihood::AnomalyLikelihood(UInt learningPeriod, UInt estimationSamples, UInt historicWindowSize, UInt reestimationPeriod, UInt aggregationWindow) :
    learningPeriod(learningPeriod),
    reestimationPeriod(reestimationPeriod),
    averagedAnomaly(aggregationWindow) {
        iteration = 0;
        probationaryPeriod = learningPeriod+estimationSamples;
        NTA_CHECK(historicWindowSize >= estimationSamples); // cerr << "estimationSamples exceeds historicWindowSize";
        NTA_CHECK(aggregationWindow < reestimationPeriod && reestimationPeriod < historicWindowSize);
        
        runningAverageAnomalies.set_capacity(historicWindowSize); 
        runningLikelihoods.set_capacity(historicWindowSize);
        runningRawAnomalyScores.set_capacity(historicWindowSize);
        NTA_CHECK(runningLikelihoods.capacity() == historicWindowSize);
    }

    
Real AnomalyLikelihood::anomalyProbability(Real anomalyScore, int timestamp) {  //FIXME even timestamp is not really used, remove too? 
    Real likelihood = 0.5f;

    if (timestamp == -1) {
      timestamp = this->iteration;
    }

    // store into relevant variables
    this->runningRawAnomalyScores.push_back(anomalyScore); 
    auto newAvg = this->averagedAnomaly.compute(anomalyScore); 
    this->runningAverageAnomalies.push_back(newAvg);
    this->iteration++;
    this->runningLikelihoods.push_back(likelihood);
    
    // We ignore the first probationaryPeriod data points - as we cannot reliably compute distribution statistics for estimating likelihood
    if (this->iteration < this->probationaryPeriod) {
      return 0.5;
    } //else {

    auto anomalies = circularBufferToVector(this->runningAverageAnomalies); 
    
      // On a rolling basis we re-estimate the distribution
      if ( this->iteration == 0 || (this->iteration % this->reestimationPeriod) == 0  || this->distribution.name == "unknown" ) {

        auto numSkipRecords = calcSkipRecords_(this->iteration, this->runningAverageAnomalies.capacity(), this->learningPeriod); //FIXME this erase (numSkipRecords) is a problem when we use sliding window (as opposed to vector)! - should we skip only once on beginning, or on each call of this fn?
        estimateAnomalyLikelihoods(anomalies, numSkipRecords);  // called to update this->distribution;  
      }
    
    auto likelihoods = updateAnomalyLikelihoods(anomalies);

      NTA_CHECK(likelihoods.size() > 0); 
      likelihood = 1.0 - likelihoods[0]; 
      NTA_CHECK(likelihood >= 0.0 && likelihood <= 1.0);

    return likelihood;
    }
    

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
**/

Real32 AnomalyLikelihood::tailProbability(Real32 x) const {
     NTA_CHECK(distribution.name != "unknown" && distribution.stdev > 0);
     
  if (x < distribution.mean) {
    // Gaussian is symmetrical around mean, so flip to get the tail probability
    Real32 xp = 2 * distribution.mean - x;
    NTA_CHECK(xp != x);
    return tailProbability(xp);
  }

  // Calculate the Q function with the complementary error function, explained
  // here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
  Real32 z = (x - distribution.mean) / distribution.stdev;
  return 0.5 * erfc(z/1.4142);
  }


DistributionParams AnomalyLikelihood::estimateNormal(vector<Real> sampleData, bool performLowerBoundCheck) {
    auto mean = compute_mean(sampleData);
    auto var = compute_var(sampleData, mean);
  DistributionParams params = DistributionParams("normal", mean, var, 0.0); 

  if (performLowerBoundCheck) {
    /* Handle edge case of almost no deviations and super low anomaly scores. We
     find that such low anomaly means can happen, but then the slightest blip
     of anomaly score can cause the likelihood to jump up to red.
     */
    if (params.mean < 0.03) { //TODO make these magic numbers a constructor parameter, or at least a const variable
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

  /**
  Filter the list of raw (pre-filtered) likelihoods so that we only preserve
  sharp increases in likelihood. 'likelihoods' can be a numpy array of floats or
  a list of floats.

  :returns: A new list of floats likelihoods containing the filtered values.
  **/
static vector<Real> filterLikelihoods_(vector<Real> likelihoods, Real redThreshold=0.99999, Real yellowThreshold=0.999){ //TODO make the redThreshold params of AnomalyLikelihood constructor() 
  redThreshold    = 1.0 - redThreshold;  //TODO maybe we could use the true meaning already in the parameters
  yellowThreshold = 1.0 - yellowThreshold;

  NTA_CHECK(redThreshold > 0.0 && redThreshold < 1.0);
  NTA_CHECK(yellowThreshold > 0.0 && yellowThreshold < 1.0);
  NTA_CHECK(yellowThreshold >= redThreshold); 

  NTA_ASSERT(likelihoods.size() >= 1);
  // The first value is untouched
  vector<Real> filteredLikelihoods(likelihoods); 

  for (size_t i=1; i< likelihoods.size(); i++) { 
    // value is in the redzone & so was previous 
    if (likelihoods[i] <= redThreshold && filteredLikelihoods[i-1] <= redThreshold) { //TODO review the (original) IFs
      filteredLikelihoods[i] = yellowThreshold;
    }
  }

  NTA_ASSERT(filteredLikelihoods.size()==likelihoods.size());
  return filteredLikelihoods;
}


vector<Real>  AnomalyLikelihood::updateAnomalyLikelihoods(vector<Real> anomalyScores, UInt verbosity) { 
  if (verbosity > 3) {
    cout << "In updateAnomalyLikelihoods."<< endl;
    cout << "Number of anomaly scores: "<<  anomalyScores.size() << endl;
//    cout << "First 20:", anomalyScores[0:min(20, len(anomalyScores))])
    cout << "Params: name=" <<  distribution.name << " mean="<<distribution.mean <<" var="<<distribution.variance <<" stdev="<<distribution.stdev <<endl;
  }

 NTA_CHECK(anomalyScores.size() > 0); // "Must have at least one anomalyScore"

  // Compute moving averages of these new scores using the previous values
  // as well as likelihood for these scores using the old estimator 
  vector<Real> likelihoods;
  likelihoods.resize(runningAverageAnomalies.size()); 
  for (auto newAverage : runningAverageAnomalies) { //TODO we could use transform() here (? or would it be less clear?) 
    likelihoods.push_back(tailProbability(newAverage)); 
  }

  // Filter the likelihood values. First we prepend the historical likelihoods
  // to the current set. Then we filter the values.  We peel off the likelihoods
  // to return and the last windowSize values to store for later.
  UInt toCrop = min((UInt)this->averagedAnomaly.getMaxWindowSize(), (UInt)runningLikelihoods.size());
  this->runningLikelihoods.insert(runningLikelihoods.end() - toCrop, likelihoods.begin(),likelihoods.end()); //append & crop
  cerr << "FUU " << this->runningLikelihoods.size() << "  <= " << this->averagedAnomaly.getMaxWindowSize() << endl; 
//!  NTA_CHECK(this->runningLikelihoods.size() <= this->averagedAnomaly.getMaxWindowSize()); //FIXME fix this check, returns ~"14xx <= 10"

  auto filteredLikelihoods = filterLikelihoods_(likelihoods);

  if (verbosity > 3) {
    cout << "Number of likelihoods:"<< likelihoods.size() << endl;
//    print("First 20 likelihoods:", likelihoods[0:min(20, len(likelihoods))])
    cout << "Leaving updateAnomalyLikelihoods."<< endl;
  }

  return filteredLikelihoods;
}


vector<Real> AnomalyLikelihood::estimateAnomalyLikelihoods(vector<Real> anomalyScores, UInt skipRecords, UInt verbosity) { //FIXME averagingWindow not used, I guess it's not a sliding window, but aggregating window (discrete steps)!
  if (verbosity > 1) {
    cout << "In estimateAnomalyLikelihoods."<<endl;
    cout << "Number of anomaly scores:" <<  anomalyScores.size() << endl;
    cout << "Skip records="<<  skipRecords << endl;
//    print("First 20:", anomalyScores[0:min(20, len(anomalyScores))])
  }

  NTA_CHECK(anomalyScores.size() > 0); // "Must have at least one anomalyScore"
  auto dataValues = anomalyScores; //FIXME the "data" should be anomaly scores, or raw values? 

  // Estimate the distribution of anomaly scores based on aggregated records
  if (dataValues.size()  <= skipRecords) {
    this->distribution = nullDistribution();
  } else {
    dataValues.erase(dataValues.begin(), dataValues.begin() + skipRecords);// remove first skipRecords
    this->distribution = estimateNormal(dataValues);
  }

  // Estimate likelihoods based on this distribution
  vector<Real> likelihoods;
  likelihoods.resize(dataValues.size());
  for (auto s : dataValues) {
    likelihoods.push_back(tailProbability(s));
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


/// HELPER methods (only used internaly in this cpp file)
Real compute_mean(vector<Real> v)  { //TODO do we have a (more comp. stable) implementation of mean/variance?
  NTA_CHECK(v.size() > 0); //avoid division by zero! 
    Real sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

Real compute_var(vector<Real> v, Real mean)  {
  NTA_CHECK(v.size() > 0); //avoid division by zero!
    Real sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return (sq_sum / v.size()) - (mean * mean);
}
}}} //ns
