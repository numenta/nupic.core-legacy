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
 * ----------------------------------------------------------------------
 */

/** @file
 * Implementation of the SparseDistributedRepresentation class
 * Also Known As "SDR" class
 *
 * SDR implementation in C++, refer to Sdr.hpp for detailed comments.
 */

#include "nupic/types/Sdr.hpp"

#include <numeric>
#include <algorithm> // std::sort

using namespace std;

namespace nupic {
namespace sdr {

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
            NTA_ASSERT(sparse_.size() <= size);
            for(auto idx : sparse_) {
                NTA_ASSERT(idx < size);
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
                NTA_ASSERT(coord_vec.size() <= size);
                NTA_ASSERT(coord_vec.size() == coordinates_[0].size()); // All coordinate vectors have same size.
                for(const auto &idx : coord_vec) {
                    NTA_ASSERT(idx < dimensions[dim]);
                }
            }
        #endif
        // Set the valid flags.
        clear();
        coordinates_valid = true;
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

    //constructors
    SparseDistributedRepresentation::SparseDistributedRepresentation() {}

    SparseDistributedRepresentation::SparseDistributedRepresentation( const vector<UInt> dimensions )
        { initialize( dimensions ); }

    void SparseDistributedRepresentation::initialize( const vector<UInt> dimensions ) {
        dimensions_ = dimensions;
        NTA_CHECK( dimensions.size() > 0 ) << "SDR has no dimensions!";

        // Calculate the SDR's size.
        size_ = 1;
        for(UInt dim : dimensions)
            size_ *= dim;

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
        NTA_ASSERT( value.dimensions == dimensions );
        clear();

        dense_valid = value.dense_valid;
        if( dense_valid ) {
            dense_.assign( value.dense_.begin(), value.dense_.end() );
        }
        sparse_valid = value.sparse_valid;
        if( sparse_valid ) {
            sparse_.assign( value.sparse_.begin(), value.sparse_.end() );
        }
        coordinates_valid = value.coordinates_valid;
        if( coordinates_valid ) {
            for(UInt dim = 0; dim < dimensions.size(); dim++)
                coordinates_[dim].assign( value.coordinates_[dim].begin(), value.coordinates_[dim].end() );
        }
        // Subclasses may override these getters and ignore the valid flags...
        if( !dense_valid and !sparse_valid and !coordinates_valid ) {
            const auto data = value.getSparse();
            sparse_.assign( data.begin(), data.end() );
            sparse_valid = true;
        }
        do_callbacks();
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
        //FIXME NTA_ASSERT(nbits > 0); //enable this check

        SDR_sparse_t range( size );
        iota( range.begin(), range.end(), 0u );
        sparse_ = rng.sample( range, nbits);
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
        const vector<UInt> turn_off = rng.sample(getSparse(), num_move_bits);

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


    void SparseDistributedRepresentation::intersection(std::vector<const SDR*> inputs) {
        const auto &input0 = inputs[0]->getDense();
        dense_.assign( input0.begin(), input0.end() );
        for(auto i = 1u; i < inputs.size(); ++i) {
            const auto &data = inputs[i]->getDense();
            for(auto z = 0u; z < data.size(); ++z)
                dense_[z] = dense_[z] && data[z];
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


    void SparseDistributedRepresentation::save(std::ostream &outStream) const {

        auto writeVector = [&outStream] (const vector<UInt> &vec) {
            outStream << vec.size() << " ";
            for( auto elem : vec ) {
                outStream << elem << " ";
            }
            outStream << endl;
        };

        // Write a starting marker and version.
        outStream << "SDR " << SERIALIZE_VERSION << " " << endl;

        // Store the dimensions.
        writeVector( dimensions );

        // Store the data in the flat-sparse format.
        writeVector( getSparse() );

        outStream << "~SDR" << endl;
    }

    void SparseDistributedRepresentation::load(std::istream &inStream) {

        auto readVector = [&inStream] (vector<UInt> &vec) { //TODO add to Serializable
            vec.clear();
            UInt size;
            inStream >> size;
            vec.reserve( size );
            for( UInt i = 0; i < size; ++i ) {
                UInt elem;
                inStream >> elem;
                vec.push_back( elem );
            }
        };

        // Read the starting marker and version.
        string marker;
        UInt version;
        inStream >> marker >> version;
        NTA_CHECK( marker == "SDR" );
        NTA_CHECK( version == SERIALIZE_VERSION );

        // Read the dimensions.
        readVector( dimensions_ );

        // Initialize the SDR.
        // Calculate the SDR's size.
        size_ = 1;
        for(UInt dim : dimensions)
            size_ *= dim;
        // Initialize sparse tuple.
        coordinates_.assign( dimensions.size(), {} );

        // Read the data.
        readVector( sparse_ );
        setSparseInplace();

        // Consume the end marker.
        inStream >> marker;
        NTA_CHECK( marker == "~SDR" );
        inStream.ignore(1);  // skip past endl.
    }


    UInt SparseDistributedRepresentation::addCallback(SDR_callback_t callback) {
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

    void SparseDistributedRepresentation::removeCallback(UInt index) {
        NTA_CHECK( index < callbacks.size() )
            << "SparseDistributedRepresentation::removeCallback, Invalid Handle!";
        NTA_CHECK( callbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeCallback, Callback already removed!";
        callbacks[index] = nullptr;
    }


    UInt SparseDistributedRepresentation::addDestroyCallback(SDR_callback_t callback) {
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

    void SparseDistributedRepresentation::removeDestroyCallback(UInt index) {
        NTA_CHECK( index < destroyCallbacks.size() )
            << "SparseDistributedRepresentation::removeDestroyCallback, Invalid Handle!";
        NTA_CHECK( destroyCallbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeDestroyCallback, Callback already removed!";
        destroyCallbacks[index] = nullptr;
    }

} // end namespace sdr
} // end namespace nupic
