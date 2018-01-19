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

      Real compute_mean(const vector<Real>& v); //forward declaration
      Real compute_var(const vector<Real>& v, Real mean); 
   static std::vector<Real> circularBufferToVector(boost::circular_buffer<Real> cb); //TODO replace with SlidingWindow
   static UInt calcSkipRecords_(UInt numIngested, UInt windowSize, UInt learningPeriod);


AnomalyLikelihood::AnomalyLikelihood(UInt learningPeriod, UInt estimationSamples, UInt historicWindowSize, UInt reestimationPeriod, UInt aggregationWindow) :
    learningPeriod(learningPeriod),
    reestimationPeriod(reestimationPeriod),
    probationaryPeriod(learningPeriod+estimationSamples),
    averagedAnomaly_(aggregationWindow) { //FIXME discuss aggregationWindow (python aggregates 20(window5)->4, here 20(w5)->20smooth
        iteration_ = 0;
        NTA_CHECK(historicWindowSize >= estimationSamples); // cerr << "estimationSamples exceeds historicWindowSize";
        NTA_CHECK(aggregationWindow < reestimationPeriod && reestimationPeriod < historicWindowSize);
        
        runningAverageAnomalies_.set_capacity(historicWindowSize); 
        runningLikelihoods_.set_capacity(historicWindowSize);
        runningRawAnomalyScores_.set_capacity(historicWindowSize);
        NTA_CHECK(runningLikelihoods_.capacity() == historicWindowSize);
    }


Real AnomalyLikelihood::anomalyProbability(Real anomalyScore, int timestamp) {  //FIXME even timestamp is not really used, remove too? 
    Real likelihood = DEFAULT_ANOMALY;

    //time handling:
    if (timestamp <0) { //use iterations
      timestamp = this->iteration_;
    } else { //use time
      NTA_ASSERT(timestamp > lastTimestamp_); //monotonic time! 
      lastTimestamp_ = timestamp;//lastTimestamp_ is used just for this check
    }
    if(initialTimestamp_ == -1) { // (re)set first,initial timestamp
      initialTimestamp_ = timestamp;
    }
    //if timestamp is not used (-1), this is iteration_
    UInt timeElapsed = (UInt)(timestamp - initialTimestamp_);  //this will be used, relative time since first timestamp (the "first" can be reseted)

    // store into relevant variables
    this->runningRawAnomalyScores_.push_back(anomalyScore);
    auto newAvg = this->averagedAnomaly_.compute(anomalyScore);
    this->runningAverageAnomalies_.push_back(newAvg);
    this->iteration_++;
 
    // We ignore the first probationaryPeriod data points - as we cannot reliably compute distribution statistics for estimating likelihood
    if (timeElapsed < this->probationaryPeriod) {
      this->runningLikelihoods_.push_back(likelihood); //after that, pushed below with real likelihood; here just 0.5
      return DEFAULT_ANOMALY;
    } //else {

    auto anomalies = circularBufferToVector(this->runningAverageAnomalies_); 
    
      // On a rolling basis we re-estimate the distribution
      if ((timeElapsed >= initialTimestamp_ + reestimationPeriod)   || distribution_.name == "unknown" ) {

        auto numSkipRecords = calcSkipRecords_(this->iteration_, this->runningAverageAnomalies_.capacity(), this->learningPeriod); //FIXME this erase (numSkipRecords) is a problem when we use sliding window (as opposed to vector)! - should we skip only once on beginning, or on each call of this fn?
        estimateAnomalyLikelihoods_(anomalies, numSkipRecords);  // called to update this->distribution_; 
        if  (timeElapsed >= initialTimestamp_ + reestimationPeriod)  { initialTimestamp_ = -1; } //reset init T
      }
    
    auto likelihoods = updateAnomalyLikelihoods_(anomalies);

      NTA_CHECK(likelihoods.size() > 0); 
      likelihood = 1.0 - likelihoods[0]; 
      NTA_CHECK(likelihood >= 0.0 && likelihood <= 1.0);

    this->runningLikelihoods_.push_back(likelihood);
    NTA_ASSERT(runningLikelihoods_.size()==runningRawAnomalyScores_.size() &&
             runningLikelihoods_.size()==runningAverageAnomalies_.size());

    return likelihood;
    }
    

/**
#
# USAGE FOR LOW-LEVEL FUNCTIONS
# -----------------------------
#
# There are two primary interface routines:
#
# estimateAnomalyLikelihoods_: batch routine, called initially and once in a
#                                while
# updateAnomalyLikelihoods: online routine, called for every new data point
#
# 1. Initially::
#
#    likelihoods, avgRecordList, estimatorParams = \
# estimateAnomalyLikelihoods_(metric_data)
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
# estimateAnomalyLikelihoods_(lots_of_metric_data)
#
**/

Real AnomalyLikelihood::tailProbability_(Real x) const {
     NTA_CHECK(distribution_.name != "unknown" && distribution_.stdev > 0);
     
  if (x < distribution_.mean) {
    // Gaussian is symmetrical around mean, so flip to get the tail probability
    Real xp = 2 * distribution_.mean - x;
    NTA_ASSERT(xp != x);
    return tailProbability_(xp);
  }

  // Calculate the Q function with the complementary error function, explained
  // here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
  Real z = (x - distribution_.mean) / distribution_.stdev;
  return 0.5 * erfc(z/1.4142);
  }


DistributionParams AnomalyLikelihood::estimateNormal_(const vector<Real>& anomalyScores, bool performLowerBoundCheck) {
    auto mean = compute_mean(anomalyScores);
    auto var = compute_var(anomalyScores, mean);
  DistributionParams params = DistributionParams("normal", mean, var, 0.0); 

  if (performLowerBoundCheck) {
    /* Handle edge case of almost no deviations and super low anomaly scores. We
     find that such low anomaly means can happen, but then the slightest blip
     of anomaly score can cause the likelihood to jump up to red.
     */
    if (params.mean < THRESHOLD_MEAN) { //TODO make these magic numbers a constructor parameter, or at least a const variable
      params.mean= THRESHOLD_MEAN;
    }

    // Catch all for super low variance to handle numerical precision issues
    if (params.variance < THRESHOLD_VARIANCE) {
      params.variance = THRESHOLD_VARIANCE;
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
static vector<Real> filterLikelihoods_(const vector<Real>& likelihoods, Real redThreshold=0.99999, Real yellowThreshold=0.999){ //TODO make the redThreshold params of AnomalyLikelihood constructor() 
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


vector<Real>  AnomalyLikelihood::updateAnomalyLikelihoods_(const vector<Real>& anomalyScores, UInt verbosity) { 
  if (verbosity > 3) {
    cout << "In updateAnomalyLikelihoods."<< endl;
    cout << "Number of anomaly scores: "<<  anomalyScores.size() << endl;
//    cout << "First 20:", anomalyScores[0:min(20, len(anomalyScores))])
    cout << "Params: name=" <<  distribution_.name << " mean="<<distribution_.mean <<" var="<<distribution_.variance <<" stdev="<<distribution_.stdev <<endl;
  }

 NTA_CHECK(anomalyScores.size() > 0); // "Must have at least one anomalyScore"

  // Compute moving averages of these new scores using the previous values
  // as well as likelihood for these scores using the old estimator 
  vector<Real> likelihoods;
  likelihoods.reserve(runningAverageAnomalies_.size()); 
  for (auto newAverage : runningAverageAnomalies_) { //TODO we could use transform() here (? or would it be less clear?) 
    likelihoods.push_back(tailProbability_(newAverage)); 
  }

/*
  // Filter the likelihood values. First we prepend the historical likelihoods
  // to the current set. Then we filter the values.  We peel off the likelihoods
  // to return and the last windowSize values to store for later.
  UInt toCrop = min((UInt)this->averagedAnomaly_.getMaxWindowSize(), (UInt)runningLikelihoods_.size());
  this->runningLikelihoods_.insert(runningLikelihoods_.end() - toCrop, likelihoods.begin(),likelihoods.end()); //append & crop //We only update runningLikelihoods in the main loop, not here
*/
  auto filteredLikelihoods = filterLikelihoods_(likelihoods);

  if (verbosity > 3) {
    cout << "Number of likelihoods:"<< likelihoods.size() << endl;
//    print("First 20 likelihoods:", likelihoods[0:min(20, len(likelihoods))])
    cout << "Leaving updateAnomalyLikelihoods."<< endl;
  }

  return filteredLikelihoods;
}


vector<Real> AnomalyLikelihood::estimateAnomalyLikelihoods_(const vector<Real>& anomalyScores, UInt skipRecords, UInt verbosity) { //FIXME averagingWindow not used, I guess it's not a sliding window, but aggregating window (discrete steps)!
  if (verbosity > 1) {
    cout << "In estimateAnomalyLikelihoods_."<<endl;
    cout << "Number of anomaly scores:" <<  anomalyScores.size() << endl;
    cout << "Skip records="<<  skipRecords << endl;
//    print("First 20:", anomalyScores[0:min(20, len(anomalyScores))])
  }

  NTA_CHECK(anomalyScores.size() > 0); // "Must have at least one anomalyScore"
  auto dataValues = anomalyScores; //FIXME the "data" should be anomaly scores, or raw values? 

  // Estimate the distribution of anomaly scores based on aggregated records
  if (dataValues.size()  <= skipRecords) {
    this->distribution_ =  DistributionParams("normal", 0.5, 1e6, 1e3); //null distribution
  } else {
    dataValues.erase(dataValues.begin(), dataValues.begin() + skipRecords);// remove first skipRecords
    this->distribution_ = estimateNormal_(dataValues);
  }

  // Estimate likelihoods based on this distribution
  vector<Real> likelihoods;
  likelihoods.reserve(dataValues.size());
  for (auto s : dataValues) {
    likelihoods.push_back(tailProbability_(s));
  }
  // Filter likelihood values
  auto filteredLikelihoods = filterLikelihoods_(likelihoods);


/*  if verbosity > 1:
    print("Discovered params=")
    print(params)
    print("Number of likelihoods:", len(likelihoods))
    print("First 20 likelihoods:", (
      filteredLikelihoods[0:min(20, len(filteredLikelihoods))] ))
    print("leaving estimateAnomalyLikelihoods_")
*/

  return filteredLikelihoods;
}


/// HELPER methods (only used internaly in this cpp file)
Real compute_mean(const vector<Real>& v)  { //TODO do we have a (more comp. stable) implementation of mean/variance?
  NTA_ASSERT(v.size() > 0); //avoid division by zero! 
    Real sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}


Real compute_var(const vector<Real>& v, Real mean)  {
  NTA_ASSERT(v.size() > 0); //avoid division by zero!
    Real sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return (sq_sum / v.size()) - (mean * mean);
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


}}} //ns
