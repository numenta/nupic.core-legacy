/* ---------------------------------------------------------------------
 * Copyright (C) 2018, David McDougall.
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

#include <vector>
#include <nupic/ntypes/Sdr.hpp>

using namespace std;
using namespace nupic;

SDR::SparseDistributedRepresentation(const vector<UInt> dimensions) {
    this->dimensions = dimensions;
    // Initialize the dense storage.
    UInt size = 1;
    for(UInt dim : dimensions)
        size *= dim;
    dense.resize( size );
    // initialize the index tuple.
    for(UInt dim : dimensions)
        index.push_back( { } );
    // No need to initialize the flatIndex array.

    clear(); // Mark all data as invalid.
};

void SDR::clear() {
    dense_valid     = false;
    flatIndex_valid = false;
    index_valid     = false;
};

void SDR::zero() {
    clear();
    flatIndex.clear();
    flatIndex_valid = true;
};

void SDR::setDense( const vector<Byte> &value ) {
    NTA_ASSERT(value.size() == getSize());
    clear();
    dense.assign( value.begin(), value.end() );
    dense_valid = true;
};  

void SDR::setDense( const UInt *value ) {
    clear();
    dense.assign( value, value + getSize() );
    dense_valid = true;
};

void SDR::setDense( const ArrayBase *value ) {
    // TODO: Assert correct size and data type 
    clear();
    NTA_ASSERT( false /* Unimplemented */ );
    dense_valid = true;
};

void SDR::setFlatIndex( const vector<UInt> &value ) {
    NTA_ASSERT(value.size() <= getSize());
    clear();
    flatIndex.assign( value.begin(), value.end() );
    flatIndex_valid = true;
};

void SDR::setFlatIndex( const UInt *value, const UInt num_values ) {
    NTA_ASSERT(num_values <= getSize());
    clear();
    flatIndex.assign( value, value + num_values );
    flatIndex_valid = true;
};

void SDR::setIndex( const vector<vector<UInt>> &value ) {
    NTA_ASSERT(value.size() == getDimensions().size());
    for( auto coord_vec : value )
        NTA_ASSERT(coord_vec.size() <= getSize());
    clear();
    for(UInt dim = 0; dim < getDimensions().size(); dim++)
        index[dim].assign( value[dim].begin(), value[dim].end() );
    index_valid = true;
};

void SDR::assign( const SparseDistributedRepresentation &value ) {
    NTA_ASSERT( value.getDimensions() == getDimensions() );

    this->dense_valid = value.dense_valid;
    if( dense_valid )
        dense.assign( value.dense.begin(), value.dense.end() );

    this->flatIndex_valid = value.flatIndex_valid;
    if( flatIndex_valid )
        flatIndex.assign( value.flatIndex.begin(), value.flatIndex.end() );

    this->index_valid = value.index_valid;
    if( index_valid )
        for(UInt dim = 0; dim < getDimensions().size(); dim++)
            index[dim].assign( value.index[dim].begin(), value.index[dim].end() );
};

const vector<Byte>*  SDR::getDense() {
    if( !dense_valid ) {
        // Convert from flatIndex to dense.
        dense.assign( getSize(), 0 );
        for(auto idx : *getFlatIndex()) {
            NTA_ASSERT(idx < getSize());
            dense[idx] = 1;
        }
        dense_valid = true;
    }
    return &dense;
};

const vector<UInt>*  SDR::getFlatIndex() {
    if( !flatIndex_valid ) {
        flatIndex.clear();
        if( dense_valid ) {
            // Convert from dense to flatIndex.
            for(UInt idx = 0; idx < getSize(); idx++)
                if( dense[idx] != 0 )
                    flatIndex.push_back( idx );
            flatIndex_valid = true;
        }
        else if( index_valid ) {
            // Convert from index to flatIndex.
            auto num_nz = index.at(0).size();
            flatIndex.reserve( num_nz );
            for(UInt nz = 0; nz < num_nz; nz++) {
                UInt flat = 0;
                for(UInt i = 0; i < dimensions.size(); i++) {
                    flat *= dimensions[i];
                    flat += index[i][nz];
                }
                flatIndex.push_back(flat);
            }
            flatIndex_valid = true;
        }
        else
            throw logic_error("Can not get value from empty SDR.");
    }
    return &flatIndex;
};

const vector<vector<UInt>>* SDR::getIndex() {
    if( !index_valid ) {
        // Convert from flatIndex to index.
        for( auto vec : index )
            vec.clear();
        for( auto idx : *getFlatIndex() ) {
            for(UInt dim = dimensions.size() - 1; dim > 0; dim--) {
                const auto dim_sz = dimensions[dim];
                index[dim].push_back( idx % dim_sz );
                idx /= dim_sz;
            }
            index[0].push_back(idx);
        }
        index_valid = true;
    }
    return &index;
};

void SDR::save(std::ostream &stream) const {
    NTA_ASSERT( false /* Unimplemented */ );
};

void SDR::load(std::istream &stream) {
    NTA_ASSERT( false /* Unimplemented */ );
};
