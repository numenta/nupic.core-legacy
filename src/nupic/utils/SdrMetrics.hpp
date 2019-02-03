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
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for SDR Metrics classes
 */

#ifndef SDR_METRICS_HPP
#define SDR_METRICS_HPP

#include <vector>
#include <numeric>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>
#include <nupic/utils/Random.hpp>
#include <cmath> // std::log2 isnan
#include <regex>

using namespace std;

namespace nupic {


/**
 * Helper for SDR metrics trackers, including: SDR_Sparsity,
 * SDR_ActivationFrequency, and SDR_Overlap classes.
 *
 * Subclasses must override method "callback".
 */
class SDR_MetricsHelper_ {
protected:
    UInt period_;
    int  samples_;
    vector<UInt> dimensions_;
    SDR*  dataSource_;
    UInt  callback_handle_;
    UInt  destroyCallback_handle_;

    /**
     * @param dimensions of SDR.  Add data to this metric by calling method
     * addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_MetricsHelper_( const vector<UInt> dimensions, UInt period ) {
        NTA_CHECK( period > 0u );
        NTA_CHECK( dimensions.size() > 0 );
        dimensions_ = dimensions,
        period_     = period;
        samples_    = 0;
        dataSource_ = nullptr;
        callback_handle_        = -1;
        destroyCallback_handle_ = -1;
    }

    /**
     * @param dataSource SDR to track.  Add data to the metric by assigning to
     * this SDR.  This class deals with adding a callback to this SDR so that
     * your SDR-MetricsTracker is notified after every update to the SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_MetricsHelper_( SDR &dataSource, UInt period )
        : SDR_MetricsHelper_(dataSource.dimensions, period)
    {
        dataSource_ = &dataSource;
        callback_handle_ = dataSource_->addCallback( [&](){
            callback( *dataSource_, 1.0f / std::min( period_, (UInt) ++samples_ ));
        });
        destroyCallback_handle_ = dataSource_->addDestroyCallback( [&](){
            deconstruct();
        });
    }

    void deconstruct() {
        if( dataSource_ != nullptr ) {
            dataSource_->removeCallback( callback_handle_ );
            dataSource_->removeDestroyCallback( destroyCallback_handle_ );
            dataSource_ = nullptr;
        }
    }

    /**
     * Add another datum to the metric.
     *
     * @param dataSource SDR to add to the metric.
     *
     * @param alpha Weight for the new datum.  Metrics trackers should use an
     * exponential weighting scheme, so that they can efficiently process
     * streaming data.  This class deals with finding a suitable weight for each
     * sample.
     */
    virtual void callback( SDR &dataSource, Real alpha ) = 0;

public:
    const int          &samples    = samples_;
    const UInt         &period     = period_;
    const vector<UInt> &dimensions = dimensions_;

    /**
     * Add an SDR datum to this Metric.  This method can only be called if the
     * Metric was constructed with dimensions and NOT an SDR.
     *
     * The given SDR's dimensions must be the same as this Metric's dimensions.
     */
    void addData(SDR &data) {
        NTA_CHECK( dataSource_ == nullptr );
        NTA_CHECK( dimensions_ == data.dimensions );
        callback( data, 1.0f / std::min( period_, (UInt) ++samples_ ));
    }

    ~SDR_MetricsHelper_() {
        deconstruct();
    }
};

/**
 * SDR_Sparsity class
 *
 * ### Description
 * Measures the sparsity of an SDR.  This accumulates measurements using an
 * exponential moving average, and outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      SDR_Sparsity B( A, 1000 )
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
class SDR_Sparsity : public SDR_MetricsHelper_ {
private:
    Real min_;
    Real max_;
    Real mean_;
    Real variance_;
    Real sparsity_;

    void initialize() {
        sparsity_   =  1234.56789f;
        min_        =  1234.56789f;
        max_        = -1234.56789f;
        mean_       =  1234.56789f;
        variance_   =  1234.56789f;
    }

    void callback(SDR &dataSource, Real alpha) override {
        sparsity_ = dataSource.getSparsity();
        min_ = std::min( min_, sparsity_ );
        max_ = std::max( max_, sparsity_ );
        // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        // See section 9.
        const Real diff      = sparsity_ - mean_;
        const Real incr      = alpha * diff;
                   mean_    += incr;
                   variance_ = (1.0f - alpha) * (variance_ + diff * incr);
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this sparsity metric by
     * assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Sparsity( SDR &dataSource, UInt period )
        : SDR_MetricsHelper_( dataSource, period )
        { initialize(); }

    /**
     * @param dimensions of SDR.  Add data to this sparsity metric by calling
     * method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Sparsity( const vector<UInt> dimensions, UInt period )
        : SDR_MetricsHelper_( dimensions, period )
        { initialize(); }

    const Real &sparsity = sparsity_;
    Real min() const { return min_; }
    Real max() const { return max_; }
    Real mean() const { return mean_; }
    Real std() const { return std::sqrt( variance_ ); }

    friend std::ostream& operator<<(std::ostream& stream, const SDR_Sparsity &S)
    {
        return stream << "Sparsity Min/Mean/Std/Max "
            << S.min() << " / " << S.mean() << " / "
            << S.std() << " / " << S.max() << endl;
    }
};

/**
 * SDR_ActivationFrequency class
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
 *      SDR_ActivationFrequency B( A, 1000 )
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
class SDR_ActivationFrequency : public SDR_MetricsHelper_ {
private:
    vector<Real> activationFrequency_;

    void initialize(UInt size) {
        activationFrequency_.assign( size, 1234.56789f );
    }

    void callback(SDR &dataSource, Real alpha) override
    {
        const auto decay = 1.0f - alpha;
        for(auto &value : activationFrequency_)
            value *= decay;

        const auto &sparse = dataSource.getFlatSparse();
        for(const auto &idx : sparse)
            activationFrequency_[idx] += alpha;
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_ActivationFrequency
     * instance by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_ActivationFrequency( SDR &dataSource, UInt period )
        : SDR_MetricsHelper_( dataSource, period )
        { initialize( dataSource.size ); }

    /**
     * @param dimensions of SDR.  Add data to this SDR_ActivationFrequency
     * instance by calling method addData(SDR&) with an SDR which has
     * these dimensions.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_ActivationFrequency( const vector<UInt> dimensions, UInt period )
        : SDR_MetricsHelper_( dimensions, period )
        {
            UInt size = 1;
            for(const auto &dim : dimensions)
                size *= dim;
            initialize( size );
        }

    const vector<Real> &activationFrequency = activationFrequency_;

    Real min() const {
        return *std::min_element(activationFrequency_.begin(),
                                 activationFrequency_.end());
    }

    Real max() const {
        return *std::max_element(activationFrequency_.begin(),
                                 activationFrequency_.end());
    }

    Real mean() const  {
        const auto sum = std::accumulate( activationFrequency_.begin(),
                                          activationFrequency_.end(),
                                          0.0f);
        return (Real) sum / activationFrequency_.size();
    }

    Real std() const {
        const auto mean_ = mean();
        auto sum_squares = 0.0f;
        for(const auto &frequency : activationFrequency) {
            const auto displacement = frequency - mean_;
            sum_squares += displacement * displacement;
        }
        const auto variance = sum_squares / activationFrequency.size();

        return std::sqrt( variance );
    }

    static Real binary_entropy_(const vector<Real> &frequencies) {
        Real accumulator = 0.0f;
        for(const auto &p  : frequencies) {
            const auto  p_ = 1.0f - p;
            const auto  e  = -p * std::log2( p ) - p_ * std::log2( p_ );
            accumulator   += isnan(e) ? 0.0f : e;
        }
        return accumulator / frequencies.size();
    }

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
    Real entropy() const {
        const auto max_extropy = binary_entropy_({ mean() });
        if( max_extropy == 0.0f )
            return 0.0f;
        return binary_entropy_( activationFrequency ) / max_extropy;
    }

    friend std::ostream& operator<< (std::ostream& stream,
                                     const SDR_ActivationFrequency &F)
    {
        stream << "Activation Frequency Min/Mean/Std/Max "
            << F.min() << " / " << F.mean() << " / "
            << F.std() << " / " << F.max() << endl;
        return stream << "Entropy " << F.entropy() << endl;
    }
};


/**
 * SDR_Overlap class
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
 *      SDR_Overlap B( A, 1000 )
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
class SDR_Overlap : public SDR_MetricsHelper_ {
private:
    SDR  previous_;
    Real overlap_;
    Real min_;
    Real max_;
    Real mean_;
    Real variance_;

    void initialize() {
        // This class needs two samples before its data is valid, instead of one
        // sample like SDR_MetricsHelper_ class  expects, so start the samples
        // counter one behind.
        samples_   -=  1;
        overlap_    =  1234.56789f;
        min_        =  1234.56789f;
        max_        = -1234.56789f;
        mean_       =  1234.56789f;
        variance_   =  1234.56789f;
        previous_.getFlatSparse();
    }

    void callback(SDR &dataSource, Real alpha) override {
        const auto nbits = std::max( previous_.getSum(), dataSource.getSum() );
        const auto overlap = (nbits == 0u) ? 0.0f
                               : (Real) previous_.getOverlap( dataSource ) / nbits;
        previous_.setSDR( dataSource );
        // Ignore first data point, need two to compute.  Account for the
        // initial decrement to samples counter.
        if( samples + 1 < 2 ) return;
        overlap_ = overlap; // Don't overwrite initial value until have valid data.
        min_     = std::min( min_, overlap );
        max_     = std::max( max_, overlap );
        // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        // See section 9.
        const Real diff      = overlap - mean_;
        const Real incr      = alpha * diff;
                   mean_    += incr;
                   variance_ = (1.0f - alpha) * (variance_ + diff * incr);
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_Overlap instance
     * by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Overlap( SDR &dataSource, UInt period )
        : SDR_MetricsHelper_( dataSource, period ),
          previous_( dataSource.dimensions )
        { initialize(); }

    /**
     * @param dimensions of SDR.  Add data to this SDR_Overlap instance
     * by calling method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Overlap( const vector<UInt> dimensions, UInt period )
        : SDR_MetricsHelper_( dimensions, period ),
          previous_( dimensions )
        { initialize(); }

    const Real &overlap = overlap_;
    Real min() const { return min_; }
    Real max() const { return max_; }
    Real mean() const { return mean_; }
    Real std() const { return std::sqrt( variance_ ); }

    friend std::ostream& operator<<(std::ostream& stream, const SDR_Overlap &V)
    {
        return stream << "Overlap Min/Mean/Std/Max "
            << V.min() << " / " << V.mean() << " / "
            << V.std() << " / " << V.max() << endl;
    }
};

/**
 * SDR_Metrics class
 *
 * ### Description
 * Measures an SDR.  This applies the following three metrics:
 *      SDR_Sparsity
 *      SDR_ActivationFrequency
 *      SDR_Overlap
 *
 * This accumulates measurements using an exponential moving average, and
 * outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      SDR_Metrics M( A, 1000 )
 *
 *      A.randomize( 0.10 )
 *      for(int i = 0; i < 20; i++)
 *          A.addNoise( 0.55 )
 *
 *      M.sparsity            -> SDR_Sparsity
 *      M.activationFrequency -> SDR_ActivationFrequency
 *      M.overlap             -> SDR_Overlap
 *      cout << M; ->
 *         SDR( 2000 )
 *            Sparsity Min/Mean/Std/Max 0.1 / 0.1 / 0 / 0.1
 *            Activation Frequency Min/Mean/Std/Max 0 / 0.1 / 0.100464 / 0.666667
 *            Entropy 0.822222
 *            Overlap Min/Mean/Std/Max 0.45 / 0.45 / 0 / 0.45
 */
// TODO: Add flags to enable/disable which metrics this uses?
class SDR_Metrics {
private:
    vector<UInt>            dimensions_;
    SDR_Sparsity            sparsity_;
    SDR_ActivationFrequency activationFrequency_;
    SDR_Overlap             overlap_;

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_Metrics instance
     * by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Metrics( SDR &dataSource, UInt period )
        : dimensions_( dataSource.dimensions ),
          sparsity_(            dataSource, period ),
          activationFrequency_( dataSource, period ),
          overlap_(             dataSource, period )
          {};

    /**
     * @param dimensions of SDR.  Add data to this SDR_Metrics instance
     * by calling method addData(SDR&) with an SDR which has these dimensions.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Metrics( const vector<UInt> dimensions, UInt period )
        : dimensions_( dimensions ),
          sparsity_(            dimensions, period ),
          activationFrequency_( dimensions, period ),
          overlap_(             dimensions, period )
          {};

    const vector<UInt>            &dimensions          = dimensions_;
    const SDR_Sparsity            &sparsity            = sparsity_;
    const SDR_ActivationFrequency &activationFrequency = activationFrequency_;
    const SDR_Overlap             &overlap             = overlap_;

    /**
     * Add an SDR datum to these Metrics.  This method can only be called if
     * SDR_Metrics was constructed with dimensions and NOT an SDR.
     *
     * The given SDR's dimensions must be the same as this Metric's dimensions.
     */
    void addData(SDR &data) {
        sparsity_.addData( data );
        activationFrequency_.addData( data );
        overlap_.addData( data );
    }

    friend std::ostream& operator<<(std::ostream& stream, const SDR_Metrics &M)
    {
        // Introduction line:  "SDR ( dimensions )"
        stream << "SDR( ";
        for(const auto &dim : M.dimensions_)
            stream << dim << " ";
        stream << ")" << endl;

        // Print data to temporary area for formatting.
        stringstream data_stream;
        data_stream << M.sparsity;
        data_stream << M.activationFrequency;
        data_stream << M.overlap;
        string data = data_stream.str();

        // Indent all of the data text (4 spaces).  Append indent to newlines.
        data = regex_replace( data, regex("\n"), "\n\r    " );
        // Strip trailing whitespace
        data = regex_replace( data, regex("\\s+$"), "" );
        // Insert first indent, append trailing newline.
        return stream << "    " << data << endl;
    }
};

} // end namespace nupic
#endif // end ifndef SDR_METRICS_HPP
