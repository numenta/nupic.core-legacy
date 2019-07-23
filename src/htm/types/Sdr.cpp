/* ---------------------------------------------------------------------
 * Copyright (C) 2018-2019, David McDougall.
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
 * Implementation of the SparseDistributedRepresentation class
 * Also Known As "SDR" class
 *
 * SDR implementation in C++, refer to Sdr.hpp for detailed comments.
 */

#include "htm/types/Sdr.hpp"

#include <numeric>
#include <algorithm> // std::sort, std::accumulate

using namespace std;

namespace htm {

    void SparseDistributedRepresentation::clear() const {
        dense_valid       = false;
        sparse_valid      = false;
        coordinates_valid = false;
    }

    void SparseDistributedRepresentation::do_callbacks() const {
        for(const auto &func_ptr : callbacks) {
            if( func_ptr != nullptr )
                func_ptr();
        }
    }

    void SparseDistributedRepresentation::setDenseInplace() const {
        // Check data is valid.
        NTA_ASSERT( dense_.size() == size );
        // Set the valid flags.
        clear();
        dense_valid = true;
        do_callbacks();
    }

    void SparseDistributedRepresentation::setSparseInplace() const {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT( is_sorted(sparse_.begin(), sparse_.end()) )
                << "Sparse & Coordinate data must be sorted!";
            if( not sparse_.empty() ) {
                NTA_ASSERT( sparse_.back() < size )
                    << "Index out of bounds of the SDR!";
            }
            UInt previous = -1;
            for( const UInt idx : sparse_ ) {
                NTA_ASSERT( idx != previous )
                    << "Sparse & Coordinate data must not contain duplicates!";
                previous = idx;
            }
        #endif
        // Set the valid flags.
        clear();
        sparse_valid = true;
        do_callbacks();
    }

    void SparseDistributedRepresentation::setCoordinatesInplace() const {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT(coordinates_.size() == dimensions.size());
            for(UInt dim = 0; dim < dimensions.size(); dim++) {
                const auto &coord_vec = coordinates_[dim];
                NTA_ASSERT(coord_vec.size() == coordinates_[0].size())
                    << "All coordinate arrays must have the same size!";
                for(const auto &idx : coord_vec) {
                    NTA_ASSERT(idx < dimensions[dim])
                        << "Index out of bounds of this dimensions of the SDR!";
                }
            }
        #endif
        // Set the valid flags.
        clear();
        coordinates_valid = true;
        #ifdef NTA_ASSERTIONS_ON
            getSparse(); // Asserts that the data is sorted & unique.
        #endif
        do_callbacks();
    }

    void SparseDistributedRepresentation::deconstruct() {
        clear();
        size_ = 0;
        dimensions_.clear();
        for( auto &func : destroyCallbacks ) {
            if( func != nullptr )
                func();
        }
        callbacks.clear();
        destroyCallbacks.clear();
    }

    // Constructors
    SparseDistributedRepresentation::SparseDistributedRepresentation() {}

    SparseDistributedRepresentation::SparseDistributedRepresentation( const vector<UInt> &dimensions )
        { initialize( dimensions ); }

    void SparseDistributedRepresentation::initialize( const vector<UInt> &dimensions ) {
        dimensions_ = dimensions;
        NTA_CHECK( dimensions.size() > 0 ) << "SDR has no dimensions!";

        // Calculate the SDR's size.
        size_ = std::accumulate(dimensions.begin(), dimensions.end(), 1u, std::multiplies<int>());

        // Special case: placeholder for SDR type used in NetworkAPI
        if(dimensions != vector<UInt>{0}) {
            NTA_CHECK(size_ > 0) << "SDR: all dimensions must be > 0";
        }

        // Initialize the dense array storage, when it's needed.
        dense_valid = false;
        // Initialize the flatSparse array, nothing to do.
        sparse_valid = true;
        // Initialize the index tuple.
        coordinates_.assign( dimensions.size(), {} );
        coordinates_valid = true;
    }

    SparseDistributedRepresentation::SparseDistributedRepresentation(
                                const SparseDistributedRepresentation &value )
        : SparseDistributedRepresentation( value.dimensions )
        { setSDR( value ); }

    SparseDistributedRepresentation::~SparseDistributedRepresentation()
        { deconstruct(); }


    void SparseDistributedRepresentation::reshape(const vector<UInt> &dimensions) const {
        // Make sure we have the data in a format which does not care about the
        // dimensions, IE: dense or sparse but not coordinates
        if( not dense_valid and not sparse_valid )
            getSparse();
        coordinates_valid = false;
        coordinates_.assign( dimensions.size(), {} );
        dimensions_ = dimensions;
        // Re-Calculate the SDRs size and check that it did not change.
        UInt newSize = std::accumulate(dimensions.begin(), dimensions.end(), 1u, std::multiplies<int>());
        NTA_CHECK( newSize == size ) << "SDR.reshape changed the size of the SDR!";
    }


    void SparseDistributedRepresentation::zero() {
        sparse_.clear();
        setSparseInplace();
    }


    void SparseDistributedRepresentation::setDense( SDR_dense_t &value ) {
        NTA_ASSERT(value.size() == size);
        dense_.swap( value );
        setDenseInplace();
    }

    SDR_dense_t& SparseDistributedRepresentation::getDense() const {
        if( !dense_valid ) {
            // Convert from flatSparse to dense.
            dense_.assign( size, 0 );
            for(const auto &idx : getSparse()) {
                dense_[idx] = 1;
            }
            dense_valid = true;
        }
        return dense_;
    }

    Byte SparseDistributedRepresentation::at(const vector<UInt> &coordinates) const {
        UInt flat = 0;
        NTA_ASSERT(coordinates.size() == dimensions.size())
                    << "SDR::at() coordinates must have same dimensions as SDR!";
        for(UInt i = 0; i < dimensions.size(); i++) {
            NTA_ASSERT( coordinates[i] < dimensions[i] )
                                    << "SDR::at() coordinates out of bounds!";
            flat *= dimensions[i];
            flat += coordinates[i];
        }
        return getDense()[flat];
    }


    void SparseDistributedRepresentation::setSparse( SDR_sparse_t &value ) {
        sparse_.swap( value );
        setSparseInplace();
    }

    SDR_sparse_t& SparseDistributedRepresentation::getSparse() const {
        if( !sparse_valid ) {
            sparse_.clear(); // Clear out any old data.
            if( coordinates_valid ) {
                // Convert from coordinates to flat-sparse.
                const auto &coords = getCoordinates();
                const auto num_nz = size ? coords[0].size() : 0u;
                sparse_.reserve( num_nz );
                for(UInt nz = 0; nz < num_nz; ++nz) {
                    UInt flat = 0;
                    for(UInt dim = 0; dim < dimensions.size(); ++dim) {
                        flat *= dimensions[dim];
                        flat += coords[dim][nz];
                    }
                    sparse_.push_back(flat);
                }
            }
            else if( dense_valid ) {
                // Convert from dense to flatSparse.
                const auto &dense = getDense();
                for(UInt idx = 0; idx < size; idx++)
                    if( dense[idx] != 0 )
                        sparse_.push_back( idx );
            }
            else
                NTA_THROW << "SDR has no data!";
            sparse_valid = true;
        }
        return sparse_;
    }


    void SparseDistributedRepresentation::setCoordinates( SDR_coordinate_t &value ) {
        coordinates_.swap( value );
        setCoordinatesInplace();
    }

    SDR_coordinate_t& SparseDistributedRepresentation::getCoordinates() const {
      if( !coordinates_valid ) {
        // Clear out any old data.
        for( auto& vec : coordinates_ ) {
          vec.clear();
        }
        // Convert from sparse to coordinates.
        for( auto idx : getSparse() ) {
          for(UInt dim = (UInt)(dimensions.size() - 1); dim > 0; --dim) {
            const auto dim_sz = dimensions[dim];
            coordinates_[dim].push_back( idx % dim_sz );
            idx /= dim_sz;
          }
          coordinates_[0].push_back(idx);
        }
        coordinates_valid = true;
      }
      return coordinates_;
    }


    void SparseDistributedRepresentation::setSDR( const SparseDistributedRepresentation &value ) {
        reshape( value.dimensions );
        // Cast the data to CONST, which forces the SDR to copy the vector
        // instead of swapping it with its current data vector.  This protects
        // the input SDR from being changed.
        const SDR_sparse_t &copyDontSwap = value.getSparse();
        setSparse( copyDontSwap );
    }


    SparseDistributedRepresentation& SparseDistributedRepresentation::operator=(const SparseDistributedRepresentation& value) {
        if( dimensions.empty() ) {
            initialize( value.dimensions );
        }
        setSDR( value );
        return *this;
    }


    UInt SparseDistributedRepresentation::getOverlap(const SparseDistributedRepresentation &sdr) const {
        NTA_ASSERT( dimensions == sdr.dimensions );

        UInt ovlp = 0u;
        const auto a = this->getDense();
        const auto b = sdr.getDense();
        for( UInt i = 0u; i < size; i++ )
            ovlp += a[i] && b[i];
        return ovlp;
    }


    void SparseDistributedRepresentation::randomize(Real sparsity) {
        Random rng( 0 );
        randomize( sparsity, rng );
    }

    void SparseDistributedRepresentation::randomize(Real sparsity, Random &rng) {
        NTA_ASSERT( sparsity >= 0.0f and sparsity <= 1.0f );
        UInt nbits = (UInt) std::round( size * sparsity );

        SDR_sparse_t range( size );
        iota( range.begin(), range.end(), 0u );
        sparse_ = rng.sample( range, nbits);
        sort( sparse_.begin(), sparse_.end() );
        setSparseInplace();
    }


    void SparseDistributedRepresentation::addNoise(Real fractionNoise) {
        Random rng( 0 );
        addNoise( fractionNoise, rng );
    }

    void SparseDistributedRepresentation::addNoise(Real fractionNoise, Random &rng) {
        NTA_ASSERT( fractionNoise >= 0. and fractionNoise <= 1. );
        NTA_CHECK( ( 1 + fractionNoise) * getSparsity() <= 1. );

        const UInt num_move_bits = (UInt) std::round( fractionNoise * getSum() );
        const auto& turn_off = rng.sample(getSparse(), num_move_bits);

        auto& dns = getDense();

        vector<UInt> off_pop;
        for(UInt idx = 0; idx < size; idx++) {
            if( dns[idx] == 0 )
                off_pop.push_back( idx );
        }
        const vector<UInt> turn_on = rng.sample(off_pop, num_move_bits);

        for( auto idx : turn_on )
            dns[ idx ] = 1;
        for( auto idx : turn_off )
            dns[ idx ] = 0;

        setDenseInplace();
    }


    void SparseDistributedRepresentation::killCells(const Real fraction, const UInt seed) {
        NTA_CHECK( fraction >= 0.0 );
        NTA_CHECK( fraction <= 1.0 );
        const UInt nkill = static_cast<UInt>(round( size * fraction ));
        Random rng(seed);
        auto &data = getDense();
	      std::vector<ElemSparse> indices(size);
	      std::iota(indices.begin(), indices.end(), 0); //fills with 0,..,size-1
	      const auto toKill = rng.sample(indices, nkill); // select nkill indices to be "killed", set to OFF/0
        for(const auto dis: toKill) {
          data[dis] = 0;
        }
        setDense( data );
    }


    void SparseDistributedRepresentation::intersection(
            const SDR &input1,
            const SDR &input2) {
        intersection( { &input1, &input2 } );
    }

    void SparseDistributedRepresentation::intersection(vector<const SDR*> inputs) {
        NTA_CHECK( inputs.size() >= 2u );
        bool inplace = false;
        for( size_t i = 0; i < inputs.size(); i++ ) {
            NTA_CHECK( inputs[i] != nullptr );
            NTA_CHECK( inputs[i]->dimensions == dimensions );
            // Check for modifying this SDR inplace.
            if( inputs[i] == this ) {
                inplace = true;
                inputs[i--] = inputs.back();
                inputs.pop_back();
            }
        }
        if( inplace ) {
            getDense(); // Make sure that the dense data is valid.
        }
        if( not inplace ) {
            // Copy one of the SDRs over to the output SDR.
            const auto &denseIn = inputs.back()->getDense();
            dense_.assign( denseIn.begin(), denseIn.end() );
            inputs.pop_back();
            // inplace = true; // Now it's an inplace operation.
        }
        for(const auto &sdr_ptr : inputs) {
            const auto &data = sdr_ptr->getDense();
            for(auto z = 0u; z < data.size(); ++z) {
                dense_[z] = dense_[z] && data[z];
            }
        }
        SDR::setDenseInplace();
    }


    void SparseDistributedRepresentation::set_union(
            const SDR &input1, const SDR &input2) {
        set_union( { &input1, &input2 } );
    }

    void SparseDistributedRepresentation::set_union(vector<const SDR*> inputs) {
        NTA_CHECK( inputs.size() >= 2u );
        bool inplace = false;
        for( size_t i = 0; i < inputs.size(); i++ ) {
            NTA_CHECK( inputs[i] != nullptr );
            NTA_CHECK( inputs[i]->dimensions == dimensions );
            // Check for modifying this SDR inplace.
            if( inputs[i] == this ) {
                inplace = true;
                inputs[i--] = inputs.back();
                inputs.pop_back();
            }
        }
        if( inplace ) {
            getDense(); // Make sure that the dense data is valid.
        }
        if( not inplace ) {
            // Copy one of the SDRs over to the output SDR.
            const auto &denseIn = inputs.back()->getDense();
            dense_.assign( denseIn.begin(), denseIn.end() );
            inputs.pop_back();
            // inplace = true; // Now it's an inplace operation.
        }
        for(const auto &sdr_ptr : inputs) {
            const auto &data = sdr_ptr->getDense();
            for(auto z = 0u; z < data.size(); ++z) {
                dense_[z] = dense_[z] || data[z];
            }
        }
        SDR::setDenseInplace();
    }


    void SparseDistributedRepresentation::concatenate(const std::vector<const SDR*>& inputs, const UInt axis)
    {
        // Check inputs.
        NTA_CHECK( inputs.size() >= 2u )
            << "Not enough inputs to SDR::concatenate, need at least 2 SDRs got " << inputs.size() << "!";
        NTA_CHECK( axis < dimensions.size() );
        UInt concat_axis_size = 0u;
        for( const auto &sdr : inputs ) {
            NTA_CHECK( sdr != nullptr );
            NTA_CHECK( sdr->dimensions.size() == dimensions.size() )
                << "All inputs to SDR::concatenate must have the same number of dimensions as the output SDR!";
            for( auto dim = 0u; dim < dimensions.size(); dim++ ) {
                if( dim == axis ) {
                    concat_axis_size += sdr->dimensions[axis];
                }
                else {
                    NTA_CHECK( sdr->dimensions[dim] == dimensions[dim] )
                        << "All dimensions except the axis must be the same!";
                }
            }
        }
        NTA_CHECK( concat_axis_size == dimensions[axis] )
            << "Axis of concatenation dimensions do not match, inputs sum to "
            << concat_axis_size << ", output expects " << dimensions[axis] << "!";

        // Setup for copying the data as rows & strides.
        vector<ElemDense*> buffers;
        vector<UInt>       row_lengths;
        for( const auto &sdr : inputs ) {
            buffers.push_back( sdr->getDense().data() );
            UInt row = 1u;
            for(UInt d = axis; d < dimensions.size(); ++d)
                row *= sdr->dimensions[d];
            row_lengths.push_back( row );
        }

        // Get the output buffer.
        dense_.resize( size );
              auto dense_data  = dense_.data();
        const auto data_end    = dense_data + size;
        const auto n_inputs    = inputs.size();
        while( dense_data < data_end ) {
            // Copy one row from each input SDR.
            for( UInt i = 0u; i < n_inputs; ++i ) {
                const auto &buf = buffers[i];
                const auto &row = row_lengths[i];
                std::copy( buf, buf + row, dense_data );
                // Increment the pointers.
                buffers[i] += row;
                dense_data += row;
            }
        }
        SDR::setDenseInplace();
    }

    bool SparseDistributedRepresentation::operator==(const SparseDistributedRepresentation &sdr) const {
        // Check attributes
        if( sdr.size != size or dimensions.size() != sdr.dimensions.size() )
            return false;
        for( UInt i = 0; i < dimensions.size(); i++ ) {
            if( dimensions[i] != sdr.dimensions[i] )
                return false;
        }
        // Check data
        return std::equal(
            getDense().begin(),
            getDense().end(), 
            sdr.getDense().begin());
    }


    UInt SparseDistributedRepresentation::addCallback(SDR_callback_t callback) const {
        UInt index = 0;
        for( ; index < callbacks.size(); index++ ) {
            if( callbacks[index] == nullptr ) {
                callbacks[index] = callback;
                return index;
            }
        }
        callbacks.push_back( callback );
        return index;
    }

    void SparseDistributedRepresentation::removeCallback(UInt index) const {
        NTA_CHECK( index < callbacks.size() )
            << "SparseDistributedRepresentation::removeCallback, Invalid Handle!";
        NTA_CHECK( callbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeCallback, Callback already removed!";
        callbacks[index] = nullptr;
    }


    UInt SparseDistributedRepresentation::addDestroyCallback(SDR_callback_t callback) const {
        UInt index = 0;
        for( ; index < destroyCallbacks.size(); index++ ) {
            if( destroyCallbacks[index] == nullptr ) {
                destroyCallbacks[index] = callback;
                return index;
            }
        }
        destroyCallbacks.push_back( callback );
        return index;
    }

    void SparseDistributedRepresentation::removeDestroyCallback(UInt index) const {
        NTA_CHECK( index < destroyCallbacks.size() )
            << "SparseDistributedRepresentation::removeDestroyCallback, Invalid Handle!";
        NTA_CHECK( destroyCallbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeDestroyCallback, Callback already removed!";
        destroyCallbacks[index] = nullptr;
    }

} // end namespace htm
