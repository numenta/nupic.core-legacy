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
 */

#include "nupic/types/Sdr.hpp"

#include <numeric>
#include <algorithm> // std::sort

using namespace std;

namespace nupic {
// SDR implementation in .cpp, refer to .hpp for detailed comments

/**
 * SparseDistributedRepresentation class
 * Also known as "SDR" class
 */
    
    void SparseDistributedRepresentation::clear() {
        dense_valid      = false;
        flatSparse_valid = false;
        sparse_valid     = false;
    }

    void SparseDistributedRepresentation::do_callbacks() {
        for(const auto func_ptr : callbacks) {
            if( func_ptr != nullptr )
                func_ptr();
        }
    }

    void SparseDistributedRepresentation::setDenseInplace() {
        // Check data is valid.
        NTA_ASSERT( dense_.size() == size );
        // Set the valid flags.
        clear();
        dense_valid = true;
        do_callbacks();
    }

    void SparseDistributedRepresentation::setFlatSparseInplace() {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT(flatSparse_.size() <= size);
            for(auto idx : flatSparse_) {
                NTA_ASSERT(idx < size);
            }
        #endif
        // Set the valid flags.
        clear();
        flatSparse_valid = true;
        do_callbacks();
    }

    void SparseDistributedRepresentation::setSparseInplace() {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT(sparse_.size() == dimensions.size());
            for(UInt dim = 0; dim < dimensions.size(); dim++) {
                const auto coord_vec = sparse_[dim];
                NTA_ASSERT(coord_vec.size() <= size);
                NTA_ASSERT(coord_vec.size() == sparse_[0].size()); // All coordinate vectors have same size.
                for(auto idx : coord_vec) {
                    NTA_ASSERT(idx < dimensions[dim]);
                }
            }
        #endif
        // Set the valid flags.
        clear();
        sparse_valid = true;
        do_callbacks();
    }

    void SparseDistributedRepresentation::deconstruct() {
        clear();
        size_ = 0;
        dimensions_.clear();
        for( auto func : destroyCallbacks ) {
            if( func != nullptr )
                func();
        }
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
        flatSparse_valid = true;
        // Initialize the index tuple.
        sparse_.assign( dimensions.size(), {} );
        sparse_valid = true;
    }

    SparseDistributedRepresentation::SparseDistributedRepresentation( const SparseDistributedRepresentation &value )
        : SparseDistributedRepresentation( value.dimensions )
        { setSDR( value ); }

    SparseDistributedRepresentation::~SparseDistributedRepresentation()
        { deconstruct(); }


    void SparseDistributedRepresentation::zero() {
        flatSparse_.clear();
        setFlatSparseInplace();
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
            for(const auto idx : getFlatSparse()) {
                dense_[idx] = 1;
            }
            dense_valid = true;
        }
        return dense_;
    }

    Byte SparseDistributedRepresentation::at(const vector<UInt> &coordinates) const {
        UInt flat = 0;
        NTA_ASSERT(coordinates.size() == dimensions.size()) << "SDR: coordinates must have same dimensions as SDR";
        for(UInt i = 0; i < dimensions.size(); i++) {
            NTA_ASSERT( coordinates[i] < dimensions[i] );
            flat *= dimensions[i];
            flat += coordinates[i];
        }
        return getDense()[flat];
    }

    void SparseDistributedRepresentation::setFlatSparse( SDR_flatSparse_t &value ) {
        flatSparse_.swap( value );
        setFlatSparseInplace();
    }


    SDR_flatSparse_t& SparseDistributedRepresentation::getFlatSparse() const {
        if( !flatSparse_valid ) {
            flatSparse_.clear(); // Clear out any old data.
            if( sparse_valid ) {
                // Convert from sparse to flatSparse.
                const auto num_nz = size ? sparse_[0].size() : 0;
                flatSparse_.reserve( num_nz );
                for(UInt nz = 0; nz < num_nz; nz++) {
                    UInt flat = 0;
                    for(UInt dim = 0; dim < dimensions.size(); dim++) {
                        flat *= dimensions[dim];
                        flat += sparse_[dim][nz];
                    }
                    flatSparse_.push_back(flat);
                }
            }
            else if( dense_valid ) {
                // Convert from dense to flatSparse.
                for(UInt idx = 0; idx < size; idx++)
                    if( dense_[idx] != 0 )
                        flatSparse_.push_back( idx );
            }
            else
                NTA_THROW << "SDR has no data!";
            flatSparse_valid = true;
        }
        return flatSparse_;
    }

    void SparseDistributedRepresentation::setSparse( SDR_sparse_t &value ) {
        sparse_.swap( value );
        setSparseInplace();
    }


    SDR_sparse_t& SparseDistributedRepresentation::getSparse() const {
      if( !sparse_valid ) {
        // Clear out any old data.
        for( auto& vec : sparse_ ) {
          vec.clear();
        }
        // Convert from flatSparse to sparse.
        for( auto idx : getFlatSparse() ) {
          for(UInt dim = (UInt)(dimensions.size() - 1); dim > 0; dim--) {
            auto dim_sz = dimensions[dim];
            sparse_[dim].push_back( idx % dim_sz );
            idx /= dim_sz;
          }
          sparse_[0].push_back(idx);
        }
        sparse_valid = true;
      }
      return sparse_;
    }


    void SparseDistributedRepresentation::setSDR( const SparseDistributedRepresentation &value ) {
        NTA_ASSERT( value.dimensions == dimensions );
        clear();

        dense_valid = value.dense_valid;
        if( dense_valid ) {
            dense_.assign( value.dense_.begin(), value.dense_.end() );
        }
        flatSparse_valid = value.flatSparse_valid;
        if( flatSparse_valid ) {
            flatSparse_.assign( value.flatSparse_.begin(), value.flatSparse_.end() );
        }
        sparse_valid = value.sparse_valid;
        if( sparse_valid ) {
            for(UInt dim = 0; dim < dimensions.size(); dim++)
                sparse_[dim].assign( value.sparse_[dim].begin(), value.sparse_[dim].end() );
        }
        // method.  Subclasses may override these getters and ignore the valid
        // flags...
        if( !dense_valid and !flatSparse_valid and !sparse_valid ) {
            const auto data = value.getFlatSparse();
            flatSparse_.assign( data.begin(), data.end() );
            flatSparse_valid = true;
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

        SDR_flatSparse_t range( size );
        iota( range.begin(), range.end(), 0 );
        flatSparse_.resize( nbits );
        rng.sample( range.data(),      size,
                    flatSparse_.data(), nbits);
        setFlatSparseInplace();
    }


    void SparseDistributedRepresentation::addNoise(Real fractionNoise) {
        Random rng( 0 );
        addNoise( fractionNoise, rng );
    }


    void SparseDistributedRepresentation::addNoise(Real fractionNoise, Random &rng) {
        NTA_ASSERT( fractionNoise >= 0. and fractionNoise <= 1. );
        NTA_CHECK( ( 1 + fractionNoise) * getSparsity() <= 1. );

        UInt num_move_bits = (UInt) std::round( fractionNoise * getSum() );
        vector<UInt> turn_off( num_move_bits , 0 );
        rng.sample(
            (UInt*) getFlatSparse().data(), getSum(),
            (UInt*) turn_off.data(),        num_move_bits);

        auto& dns = getDense();

        vector<UInt> off_pop;
        for(UInt idx = 0; idx < size; idx++) {
            if( dns[idx] == 0 )
                off_pop.push_back( idx );
        }
        vector<UInt> turn_on( num_move_bits, 0 );
        rng.sample(
            off_pop.data(), (UInt)off_pop.size(),
            turn_on.data(), num_move_bits);

        for( auto idx : turn_on )
            dns[ idx ] = 1;
        for( auto idx : turn_off )
            dns[ idx ] = 0;

        setDenseInplace();
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
        writeVector( getFlatSparse() );

        outStream << "~SDR" << endl;
    }


    void SparseDistributedRepresentation::load(std::istream &inStream) {

        auto readVector = [&inStream] (vector<UInt> &vec) { //TODO add to Serializable
            vec.clear();
            UInt size;
            inStream >> size;
            vec.reserve( size );
            for( UInt i = 0; i < size; i++ ) {
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
        sparse_.assign( dimensions.size(), {} );

        // Read the data.
        readVector( flatSparse_ );
        setFlatSparseInplace();

        // Consume the end marker.
        inStream >> marker;
        NTA_CHECK( marker == "~SDR" );
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

} // end namespace nupic
