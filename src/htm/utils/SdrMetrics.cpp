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
 * Implementation for SDR Metrics classes
 */

#include <cmath> // log2, isnan, NAN, INFINITY
#include <numeric> // accumulate
#include <regex>
#include <htm/utils/SdrMetrics.hpp>

using namespace std;

namespace htm {


MetricsHelper_::MetricsHelper_( const vector<UInt> &dimensions, UInt period ) {
    NTA_CHECK( period > 0u );
    NTA_CHECK( dimensions.size() > 0 );
    dimensions_ = dimensions,
    period_     = period;
    samples_    = 0u;
    dataSource_ = nullptr;
    callback_handle_        = -1;
    destroyCallback_handle_ = -1;
}

MetricsHelper_::MetricsHelper_( const SDR &dataSource, UInt period )
    : MetricsHelper_(dataSource.dimensions, period)
{
    dataSource_ = &dataSource;
    callback_handle_ = dataSource_->addCallback( [&](){
        callback( *dataSource_, 1.0f / std::min( period_, (UInt) ++samples_ ));
    });
    destroyCallback_handle_ = dataSource_->addDestroyCallback( [&](){
        deconstruct();
    });
}

void MetricsHelper_::deconstruct() {
    if( dataSource_ != nullptr ) {
        dataSource_->removeCallback( callback_handle_ );
        dataSource_->removeDestroyCallback( destroyCallback_handle_ );
        dataSource_ = nullptr;
    }
}

MetricsHelper_::~MetricsHelper_() {
    deconstruct();
}

void MetricsHelper_::addData(const SDR &data) {
    NTA_CHECK( dataSource_ == nullptr )
        << "Method addData can only be called if this metric was NOT initialize with an SDR!";
    NTA_CHECK( dimensions_ == data.dimensions );
    callback( data, 1.0f / std::min( period_, (UInt) ++samples_ ));
}


/******************************************************************************/

Sparsity::Sparsity( const vector<UInt> &dimensions, UInt period )
    : MetricsHelper_( dimensions, period )
        { initialize(); }

Sparsity::Sparsity( const SDR &dataSource, UInt period )
    : MetricsHelper_( dataSource, period )
    { initialize(); }

void Sparsity::initialize() {
    sparsity_   =  NAN;
    min_        =  INFINITY;
    max_        = -INFINITY;
    mean_       =  1234.567f;
    variance_   =  1234.567f;
}

void Sparsity::callback(const SDR &dataSource, Real alpha) {
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

Real Sparsity::min() const { return min_; }
Real Sparsity::max() const { return max_; }
Real Sparsity::mean() const { return mean_; }
Real Sparsity::std() const { return std::sqrt( variance_ ); }

std::ostream& operator<<(std::ostream& stream, const Sparsity &S)
{
    return stream << "Sparsity Min/Mean/Std/Max "
        << S.min() << " / " << S.mean() << " / "
        << S.std() << " / " << S.max() << endl;
}


/******************************************************************************/

ActivationFrequency::ActivationFrequency( const vector<UInt> &dimensions,
                                          const UInt          period,
                                          const Real          initialValue )
    : MetricsHelper_( dimensions, period )
{
    UInt size = 1;
    for(const auto &dim : dimensions)
        size *= dim;
    initialize( size, initialValue );
}

ActivationFrequency::ActivationFrequency( const SDR &dataSource,
                                          const UInt period,
                                          const Real initialValue )
    : MetricsHelper_( dataSource, period )
    { initialize( dataSource.size, initialValue ); }

void ActivationFrequency::initialize( UInt size, Real initialValue ) {
    if( initialValue == -1 ) {
        activationFrequency_.assign( size, 1234.567f );
        alwaysExponential_ = false;
    }
    else {
        NTA_CHECK( initialValue >= 0.0f );
        NTA_CHECK( initialValue <= 1.0f );
        activationFrequency_.assign( size, initialValue );
        alwaysExponential_ = true;
    }
}

void ActivationFrequency::callback(const SDR &dataSource, Real alpha)
{
    if( alwaysExponential_ ) {
        alpha = 1.0f / period;
    }

    const auto decay = 1.0f - alpha;
    for(auto &value : activationFrequency_)
        value *= decay;

    const auto &sparse = dataSource.getSparse();
    for(const auto &idx : sparse)
        activationFrequency_[idx] += alpha;
}

Real ActivationFrequency::min() const {
    return *std::min_element(activationFrequency_.begin(),
                             activationFrequency_.end());
}

Real ActivationFrequency::max() const {
    return *std::max_element(activationFrequency_.begin(),
                             activationFrequency_.end());
}

Real ActivationFrequency::mean() const  {
    const auto sum = std::accumulate( activationFrequency_.begin(),
                                      activationFrequency_.end(),
                                      0.0f);
    return (Real) sum / activationFrequency_.size();
}

Real ActivationFrequency::std() const {
    const auto mean_ = mean();
    auto sum_squares = 0.0f;
    for(const auto &frequency : activationFrequency) {
        const auto displacement = frequency - mean_;
        sum_squares += displacement * displacement;
    }
    const auto variance = sum_squares / activationFrequency.size();

    return std::sqrt( variance );
}

Real ActivationFrequency::binary_entropy_(const vector<Real> &frequencies) {
    Real accumulator = 0.0f;
    for(const auto &p  : frequencies) {
        const auto  p_ = 1.0f - p;
        const auto  e  = -p * std::log2( p ) - p_ * std::log2( p_ );
        accumulator   += isnan(e) ? 0.0f : e;
    }
    return accumulator / frequencies.size();
}

Real ActivationFrequency::entropy() const {
    const auto max_extropy = binary_entropy_({ mean() });
    if( max_extropy == 0.0f )
        return 0.0f;
    return binary_entropy_( activationFrequency ) / max_extropy;
}

std::ostream& operator<< (std::ostream& stream,
                                 const ActivationFrequency &F)
{
    stream << "Activation Frequency Min/Mean/Std/Max "
        << F.min() << " / " << F.mean() << " / "
        << F.std() << " / " << F.max() << endl;
    return stream << "Entropy " << F.entropy() << endl;
}


/******************************************************************************/

Overlap::Overlap( const vector<UInt> &dimensions, UInt period )
    : MetricsHelper_( dimensions, period ),
      previous_( dimensions )
    { initialize(); }

Overlap::Overlap( const SDR &dataSource, UInt period )
    : MetricsHelper_( dataSource, period ),
      previous_( dataSource.dimensions )
    { initialize(); }

void Overlap::initialize() {
    overlap_    =  NAN;
    min_        =  INFINITY;
    max_        = -INFINITY;
    mean_       =  1234.567f;
    variance_   =  1234.567f;
    reset();
}

void Overlap::reset()
    { previousValid_ = false; }

void Overlap::callback(const SDR &dataSource, Real alpha) {
    if( not previousValid_ ) {
        previous_.setSDR( dataSource );
        previousValid_ = true;
        // It takes two data samples to compute overlap so decrement the
        // samples counter & return & wait for the next sample.
        samples_ -= 1;
        overlap_ = NAN;
        return;
    }
    const auto nbits = std::max( previous_.getSum(), dataSource.getSum() );
    const auto rawOverlap = previous_.getOverlap( dataSource );
    overlap_ = (nbits == 0u) ? 1.0f : (Real) rawOverlap / nbits;
    min_     = std::min( min_, overlap_ );
    max_     = std::max( max_, overlap_ );
    // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
    // See section 9.
    const Real diff      = overlap_ - mean_;
    const Real incr      = alpha * diff;
               mean_    += incr;
               variance_ = (1.0f - alpha) * (variance_ + diff * incr);
    previous_.setSDR( dataSource );
}

Real Overlap::min() const { return min_; }
Real Overlap::max() const { return max_; }
Real Overlap::mean() const { return mean_; }
Real Overlap::std() const { return std::sqrt( variance_ ); }

std::ostream& operator<<(std::ostream& stream, const Overlap &V)
{
    return stream << "Overlap Min/Mean/Std/Max "
        << V.min() << " / " << V.mean() << " / "
        << V.std() << " / " << V.max() << endl;
}


/******************************************************************************/

Metrics::Metrics( const vector<UInt> &dimensions, UInt period )
    : dimensions_( dimensions ),
      sparsity_(            dimensions, period ),
      activationFrequency_( dimensions, period ),
      overlap_(             dimensions, period )
      {};

Metrics::Metrics( const SDR &dataSource, UInt period )
    : dimensions_( dataSource.dimensions ),
      sparsity_(            dataSource, period ),
      activationFrequency_( dataSource, period ),
      overlap_(             dataSource, period )
      {};

void Metrics::reset()
    { overlap_.reset(); }

void Metrics::addData(const SDR &data) {
    sparsity_.addData( data );
    activationFrequency_.addData( data );
    overlap_.addData( data );
}

std::ostream& operator<<(std::ostream& stream, const Metrics &M)
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

} // end namespace htm
