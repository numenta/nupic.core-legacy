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

#ifndef NUPIC_ALGORITHMS_ANOMALY_HPP
#define NUPIC_ALGORITHMS_ANOMALY_HPP

#include <memory> // Needed for smart pointer templates
#include <vector>
#include <nupic/types/Serializable.hpp>
#include <nupic/algorithms/AnomalyLikelihood.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/MovingAverage.hpp> // Needed for for smart pointer templates
#include <nupic/types/Sdr.hpp> // sdr::SDR

namespace nupic {
namespace algorithms {
namespace anomaly {


/**
 * Computes the raw anomaly score.
 *
 * The raw anomaly score is the fraction of active columns not predicted.
 *
 * @param activeColumns: array of active column indices
 * @param prevPredictedColumns: array of columns indices predicted in
 *     prev step
 * @return anomaly score 0..1 (Real32)
 */
Real32 computeRawAnomalyScore(std::vector<UInt> &active,
                              std::vector<UInt> &predicted);

Real32 computeRawAnomalyScore(const sdr::SDR& active, const sdr::SDR& predicted);

enum class AnomalyMode { PURE, LIKELIHOOD, WEIGHTED };

class Anomaly : public Serializable {
public:
  /**
   * Utility class for generating anomaly scores in different ways.
   *
   * Supported modes:
   *    PURE - the raw anomaly score as computed by computeRawAnomalyScore
   *    LIKELIHOOD - uses the AnomalyLikelihood class on top of the raw
   *        anomaly scores (not implemented in C++)
   *    WEIGHTED - multiplies the likelihood result with the raw anomaly
   *        score that was used to generate the likelihood (not
   *        implemented in C++)
   *
   *    @param slidingWindowSize (optional) - how many elements are
   *        summed up; enables moving average on final anomaly score;
   *        int >= 0
   *    @param mode (optional) - (enum) how to compute anomaly;
   *        possible values are AnomalyMode::
   *          - PURE - the default, how much anomal the value is;
   *              Real32 0..1 where 1=totally unexpected
   *          - LIKELIHOOD - uses the anomaly_likelihood code;
   *              models probability of receiving this value and
   *              anomalyScore
   *          - WEIGHTED - "pure" anomaly weighted by "likelihood"
   *              (anomaly * likelihood)
   *    @param binaryAnomalyThreshold (optional) - if set [0,1] anomaly
   *        score will be discretized to 1/0
   *        (1 iff >= binaryAnomalyThreshold). The transformation is
   *        applied after moving average is computed.
   */
  Anomaly(UInt slidingWindowSize = 0, AnomalyMode mode = AnomalyMode::PURE,
          Real32 binaryAnomalyThreshold = 0);

  /**
   * Compute the anomaly score as the percent of active columns not
   * predicted.
   *
   * @param active: array of active column indices
   * @param predicted: array of columns indices predicted in prev step (T-1)
   * @param timestamp: (optional) date timestamp when the sample occured
   *                   (used in anomaly-likelihood), -1 defaults to using iteration step
   * @return the computed anomaly score; Real32 0..1
   */
  Real compute(std::vector<UInt> &active,
               std::vector<UInt> &predicted,
               int timestamp = -1);

  Real compute(const sdr::SDR &active,
               const sdr::SDR &predicted,
               int timestamp = -1);

  
  CerealAdapter;
  template<class Archive>
  void save_ar(Archive & ar) const {
    std::string name("Anomaly");
    ar(CEREAL_NVP(name), 
       CEREAL_NVP(mode_), 
       CEREAL_NVP(binaryThreshold_), 
       CEREAL_NVP(movingAverage_), 
       CEREAL_NVP(likelihood_) );
  }
  template<class Archive>
  void load_ar(Archive & ar) {
    std::string name;
    UInt slidingWindowSize;
    ar(CEREAL_NVP(name),
       CEREAL_NVP(mode_),
       CEREAL_NVP(binaryThreshold_),
       CEREAL_NVP(likelihood_),
       CEREAL_NVP(slidingWindowSize));
    if (slidingWindowSize > 0u) {
      movingAverage_.reset(new nupic::util::MovingAverage(slidingWindowSize));
    }
    ar(CEREAL_NVP(movingAverage_));
  }

private:
  AnomalyMode mode_;
  Real32 binaryThreshold_;
  std::unique_ptr<nupic::util::MovingAverage> movingAverage_;
  AnomalyLikelihood likelihood_; //TODO which params/how pass them to constructor?
};
}}} //end-ns

#endif // NUPIC_ALGORITHMS_ANOMALY_HPP
