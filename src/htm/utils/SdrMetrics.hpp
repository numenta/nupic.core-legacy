/* ---------------------------------------------------------------------
 * Copyright (C) 2019, David McDougall.
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
 * ---------------------------------------------------------------------- */

/** @file
 * Definitions for SDR Metrics classes
 */

#ifndef SDR_METRICS_HPP
#define SDR_METRICS_HPP

#include <vector>
#include <htm/types/Sdr.hpp>
#include <htm/types/Types.hpp>

namespace htm {

/**
 * Helper for SDR metrics trackers, including: Sparsity,
 * ActivationFrequency, and Overlap classes.
 *
 * Subclasses must override method "callback".
 */
class MetricsHelper_ {
public:
    const UInt              &period     = period_;
    const UInt              &samples    = samples_;
    const std::vector<UInt> &dimensions = dimensions_;

    /**
     * Add an SDR datum to this Metric.  This method can only be called if the
     * Metric was constructed with dimensions and NOT an SDR.
     *
     * The given SDR's dimensions must be the same as this Metric's dimensions.
     */
    void addData(const SDR &data);

    virtual ~MetricsHelper_();

private:
    std::vector<UInt> dimensions_;
    const SDR* dataSource_;
    UInt callback_handle_;
    UInt destroyCallback_handle_;

protected:
    UInt period_;
    UInt samples_;

    /**
     * @param dimensions of SDR.  Add data to this metric by calling method
     * addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time constant for exponential moving average.
     */
    MetricsHelper_( const std::vector<UInt> &dimensions, UInt period );

    /**
     * @param dataSource SDR to track.  Add data to the metric by assigning to
     * this SDR.  This class deals with adding a callback to this SDR so that
     * your SDR-MetricsTracker is notified after every update to the SDR.
     *
     * @param period Time constant for exponential moving average.
     */
    MetricsHelper_( const SDR &dataSource, UInt period );

    void deconstruct();

    /**
     * Add another datum to the metric.
     *      Subclasses MUST override this method!
     *
     * @param dataSource SDR to add to the metric.
     *
     * @param alpha Weight for the new datum.  Metrics trackers should use an
     * exponential weighting scheme, so that they can efficiently process
     * streaming data.  This class deals with finding a suitable weight for each
     * sample.
     */
    virtual void callback( const SDR &dataSource, Real alpha ) = 0;
};


/**
 * Sparsity class
 *
 * ### Description
 * Measures the sparsity of an SDR.  This accumulates measurements using an
 * exponential moving average, and outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      Sparsity B( A, 1000 )
 *      A.randomize( 0.01 )
 *      A.randomize( 0.15 )
 *      A.randomize( 0.05 )
 *      B.sparsity ->  0.05
 *      B.min()    ->  0.01
 *      B.max()    ->  0.15
 *      B.mean()   -> ~0.07
 *      B.std()    -> ~0.06
 *      cout << B  -> Sparsity Min/Mean/Std/Max 0.01 / 0.0700033 / 0.0588751 / 0.15
 */
class Sparsity : public MetricsHelper_ {
public:
    /**
     * @param dataSource SDR to track.  Add data to this sparsity metric by
     * assigning to this SDR.
     *
     * @param period Time constant for exponential moving average.
     */
    Sparsity( const SDR &dataSource, UInt period );

    /**
     * @param dimensions of SDR.  Add data to this sparsity metric by calling
     * method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time constant for exponential moving average.
     */
    Sparsity( const std::vector<UInt> &dimensions, UInt period );

    const Real &sparsity = sparsity_;

    Real min() const;
    Real max() const;
    Real mean() const;
    Real std() const;

    friend std::ostream& operator<< ( std::ostream &, const Sparsity & );

private:
    Real sparsity_;
    Real min_;
    Real max_;
    Real mean_;
    Real variance_;

    void initialize();

    void callback(const SDR &dataSource, Real alpha) override;
};


/**
 * ActivationFrequency class
 *
 * ### Description
 * Measures the activation frequency of each value in an SDR.  This accumulates
 * measurements using an exponential moving average, and outputs a summary of
 * results.
 *
 * Activation frequencies are Real numbers in the range [0, 1], where zero
 * indicates never active, and one indicates always active.
 *
 * Example Usage:
 *      SDR A( 2 )
 *      ActivationFrequency B( A, 1000 )
 *      A.setDense({ 0, 0 })
 *      A.setDense({ 1, 1 })
 *      A.setDense({ 0, 1 })
 *      B.activationFrequency -> { 0.33, 0.66 }
 *      B.min()     -> ~0.33
 *      B.max()     -> ~0.66
 *      B.mean()    ->  0.50
 *      B.std()     -> ~0.16
 *      B.entropy() -> ~0.92
 *      cout << B   -> Activation Frequency Min/Mean/Std/Max 0.333333 / 0.5 / 0.166667 / 0.666667
 *                     Entropy 0.918296
 */
class ActivationFrequency : public MetricsHelper_ {
public:
    /**
     * @param dataSource SDR to track.  Add data to this ActivationFrequency
     * instance by assigning to this SDR.
     *
     * @param period Time constant for exponential moving average.
     *
     * @param initialValue - Optional, Makes this ActivationFrequency instance
     * think that it is the result of a long running process (even though it was
     * just created).  This assigns an initial activation frequency to all bits
     * in the SDR, and causes it to always use the exponential moving average
     * instead of the regular average which is usually applied to the first
     * "period" many samples.
     * Note: This argument is useful for using this metric as part of boosting
     *       algorithms which seek to push the activation frequencies to a
     *       target value. These algorithms will overreact to the default early
     *       behavior of this class during the first "period" many samples.
     */
    ActivationFrequency( const SDR &dataSource, UInt period, Real initialValue = -1 );

    /**
     * @param dimensions of SDR.  Add data to this ActivationFrequency
     * instance by calling method addData(SDR&) with an SDR which has
     * these dimensions.
     *
     * @param period       - Same as other constructor overload.
     * @param initialValue - Same as other constructor overload.
     */
    ActivationFrequency( const std::vector<UInt> &dimensions, UInt period,
                         Real initialValue = -1 );

    const std::vector<Real> &activationFrequency = activationFrequency_;

    Real min() const;
    Real max() const;
    Real mean() const;
    Real std() const;

    /**
     * Binary entropy is a measurement of information.  It measures how well the
     * SDR utilizes its resources (bits).  A low entropy indicates that many
     * bits in the SDR are under-utilized and do not transmit as much
     * information as they could.  A high entropy indicates that the SDR
     * optimally utilizes its resources.  The most optimal use of SDR resources
     * is when all bits have an equal activation frequency.  For convenience,
     * the entropy is scaled by the theoretical maximum into the range [0, 1].
     *
     * @returns Binary entropy of SDR, scaled to range [0, 1].
     */
    Real entropy() const;

    friend std::ostream& operator<< (std::ostream &, const ActivationFrequency &);

private:
    std::vector<Real> activationFrequency_;
    bool alwaysExponential_;

    void initialize(UInt size, Real initialValue);

    static Real binary_entropy_(const std::vector<Real> &frequencies);

    void callback(const SDR &dataSource, Real alpha) override;
};


/**
 * Overlap class
 *
 * ### Description
 * Measures the overlap between successive assignments to an SDR.  This class
 * accumulates measurements using an exponential moving average, and outputs a
 * summary of results.
 *
 * This class normalizes the overlap into the range [0, 1] by dividing by the
 * number of active values.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      Overlap B( A, 1000 )
 *      A.randomize( 0.05 )
 *      A.addNoise( 0.95 )  ->  5% overlap
 *      A.addNoise( 0.55 )  -> 45% overlap
 *      A.addNoise( 0.72 )  -> 28% overlap
 *      B.overlap   ->  0.28
 *      B.min()     ->  0.05
 *      B.max()     ->  0.45
 *      B.mean()    ->  0.26
 *      B.std()     -> ~0.16
 *      cout << B   -> Overlap Min/Mean/Std/Max 0.05 / 0.260016 / 0.16389 / 0.45
 */
class Overlap : public MetricsHelper_ {
public:
    /**
     * @param dataSource SDR to track.  Add data to this Overlap instance
     * by assigning to this SDR.
     *
     * @param period Time constant for exponential moving average.
     */
    Overlap( const SDR &dataSource, UInt period );

    /**
     * @param dimensions of SDR.  Add data to this Overlap instance
     * by calling method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time constant for exponential moving average.
     */
    Overlap( const std::vector<UInt> &dimensions, UInt period );

    /* For use with time-series data sets. */
    void reset();

    const Real &overlap = overlap_;

    Real min() const;
    Real max() const;
    Real mean() const;
    Real std() const;

    friend std::ostream& operator<< ( std::ostream &, const Overlap & );

private:
    SDR  previous_;
    bool previousValid_;
    Real overlap_;
    Real min_;
    Real max_;
    Real mean_;
    Real variance_;

    void initialize();

    void callback(const SDR &dataSource, Real alpha) override;
};

/**
 * Metrics class
 *
 * ### Description
 * Measures an SDR.  This applies the following three metrics:
 *      Sparsity
 *      ActivationFrequency
 *      Overlap
 *
 * This accumulates measurements using an exponential moving average, and
 * outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      Metrics M( A, 1000 )
 *
 *      A.randomize( 0.10 )
 *      for(int i = 0; i < 20; i++)
 *          A.addNoise( 0.55 )
 *
 *      M.sparsity            -> Sparsity
 *      M.activationFrequency -> ActivationFrequency
 *      M.overlap             -> Overlap
 *      cout << M; ->
 *         SDR( 2000 )
 *            Sparsity Min/Mean/Std/Max 0.1 / 0.1 / 0 / 0.1
 *            Activation Frequency Min/Mean/Std/Max 0 / 0.1 / 0.100464 / 0.666667
 *            Entropy 0.822222
 *            Overlap Min/Mean/Std/Max 0.45 / 0.45 / 0 / 0.45
 */
// TODO: Add flags to enable/disable which metrics this uses?
class Metrics {
public:
    /**
     * @param dataSource SDR to track.  Add data to this Metrics instance
     * by assigning to this SDR.
     *
     * @param period Time constant for exponential moving average.
     */
    Metrics( const SDR &dataSource, UInt period );

    /**
     * @param dimensions of SDR.  Add data to this Metrics instance
     * by calling method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time constant for exponential moving average.
     */
    Metrics( const std::vector<UInt> &dimensions, UInt period );

    /* For use with time-series data sets. */
    void reset();

    const std::vector<UInt>   &dimensions          = dimensions_;
    const Sparsity            &sparsity            = sparsity_;
    const ActivationFrequency &activationFrequency = activationFrequency_;
    const Overlap             &overlap             = overlap_;

    /**
     * Add an SDR datum to these Metrics.  This method can only be called if
     * Metrics was constructed with dimensions and NOT an SDR.
     *
     * The given SDR's dimensions must be the same as this Metric's dimensions.
     */
    void addData(const SDR &data);

    friend std::ostream& operator<<(std::ostream& stream, const Metrics &M);

private:
    std::vector<UInt>   dimensions_;
    Sparsity            sparsity_;
    ActivationFrequency activationFrequency_;
    Overlap             overlap_;
};

} // end namespace htm
#endif // end ifndef SDR_METRICS_HPP
