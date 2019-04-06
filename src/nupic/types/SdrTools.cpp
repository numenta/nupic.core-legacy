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
 * Implementations for SDR tools & helper classes
 */

#include <nupic/types/SdrTools.hpp>

namespace nupic {
namespace sdr {

using namespace std;


Reshape::Reshape(SDR &sdr, const vector<UInt> &dimensions)
    : ReadOnly_( dimensions )
{
    clear();
    parent = &sdr;
    NTA_CHECK( size == parent->size ) << "SDR Reshape must have same size as given SDR.";
    callback_handle = parent->addCallback( [&] () {
        clear();
        do_callbacks();
    });
    destroyCallback_handle = parent->addDestroyCallback( [&] () {
        deconstruct();
    });
}

SDR_dense_t& Reshape::getDense() const {
    NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
    return parent->getDense();
}

SDR_sparse_t& Reshape::getSparse() const {
    NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
    return parent->getSparse();
}

SDR_coordinate_t& Reshape::getCoordinates() const {
    NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
    if( dimensions.size() == parent->dimensions.size() &&
        equal( dimensions.begin(), dimensions.end(),
               parent->dimensions.begin() )) {
        // All things equal, prefer reusing the parent's cached value.
        return parent->getCoordinates();
    }
    else {
        // Don't override getCoordinates().  It will call either getDense()
        // or getSparse() to get its data, and will use this SDR Reshape's
        // dimensions.
        return SDR::getCoordinates();
    }
}

void Reshape::save(std::ostream &outStream) const {
    NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
    parent->save( outStream );
}

void Reshape::deconstruct() {
    // Unlink this SDR from the parent SDR.
    if( parent != nullptr ) {
        parent->removeCallback( callback_handle );
        parent->removeDestroyCallback( destroyCallback_handle );
        parent = nullptr;
        SDR::deconstruct();
    }
}


/******************************************************************************/

void Concatenation::initialize(const vector<SDR*> inputs, const UInt axis)
{
    NTA_CHECK( inputs.size() >= 1u )
        << "Not enough inputs to SDR Concatenation, need at least 2 SDRs got " << inputs.size() << ".";
    inputs_.assign( inputs.begin(), inputs.end() );
    axis_ = axis;
    const UInt n_dim = (UInt)inputs[0]->dimensions.size();
    NTA_CHECK( axis_ < n_dim );
    // Determine dimensions & check input dimensions.
    vector<UInt> dims = inputs[0]->dimensions;
    dims[axis] = 0;
    for(auto i = 0u; i < inputs.size(); ++i) {
        NTA_CHECK( inputs[i]->dimensions.size() == n_dim )
            << "All inputs to SDR Concatenation must have the same number of dimensions!";
        for(auto d = 0u; d < n_dim; d++) {
            if( d == axis )
                dims[axis] += inputs[i]->dimensions[d];
            else
                NTA_CHECK( inputs[i]->dimensions[d] == dims[d] )
                    << "All dimensions except the axis must be the same! "
                    << "Argument #" << i << " dimension #" << d << ".";
        }
    }
    SDR::initialize( dims );

    callback_handles_.clear();
    destroyCallback_handles_.clear();
    for(SDR *inp : inputs) {
        NTA_CHECK(inp != nullptr);
        // When input SDR is assigned to, invalidate this SDR.  This SDR
        // will be recalculated next time it is accessed.
        callback_handles_.push_back( inp->addCallback( [&] ()
            { clear(); }));
        // This SDR can't survive without all of its input SDRs.
        destroyCallback_handles_.push_back( inp->addDestroyCallback( [&] ()
            { deconstruct(); }));
    }
    clear();
}

void Concatenation::clear() const {
    SDR::clear();
    // Always advertise that this SDR has dense data.
    dense_valid = true;
    // But make note that this SDR does not actually have dense data, it
    // will be computed it when it's requested.
    dense_valid_lazy = false;
}

SDR_dense_t& Concatenation::getDense() const {
    NTA_ASSERT( dense_valid );
    if( !dense_valid_lazy ) {
        // Setup for copying the data as rows & strides.
        const UInt    n_dim = (UInt)inputs[0]->dimensions.size();
        vector<ElemDense*> buffers;
        vector<UInt>  row_lengths;
        for(const auto &sdr : inputs) {
            buffers.push_back( sdr->getDense().data() );
            UInt row = 1u;
            for(UInt d = axis; d < n_dim; ++d)
                row *= sdr->dimensions[d];
            row_lengths.push_back( row );
        }
        // Get the output buffer.
        dense_.resize( size );
        auto  dense_data = dense_.data();
        const auto data_end    = dense_data + size;
        const auto n_inputs    = inputs.size();
        while( dense_data < data_end ) {
            // Copy one row from each input SDR.
            for(UInt i = 0u; i < n_inputs; ++i) {
                const auto &buf = buffers[i];
                const auto &row = row_lengths[i];
                std::copy( buf, buf + row, dense_data );
                // Increment the pointers.
                buffers[i] += row;
                dense_data += row;
            }
        }
        SDR::setDenseInplace();
        dense_valid_lazy = true;
    }
    return dense_;
}

void Concatenation::deconstruct() {
    // Unlink everything at death.
    for(auto i = 0u; i < inputs_.size(); i++) {
        inputs_[i]->removeCallback( callback_handles_[i] );
        inputs_[i]->removeDestroyCallback( destroyCallback_handles_[i] );
    }
    // Clear internal data.
    inputs_.clear();
    callback_handles_.clear();
    destroyCallback_handles_.clear();
    dense_valid_lazy = false;
    // Notify SDR parent class.
    SDR::deconstruct();
}


/******************************************************************************/

void Intersection::initialize(const vector<SDR*> inputs)
{
    NTA_CHECK( inputs.size() >= 1u )
        << "Not enough inputs to SDR Intersection, need at least 2 SDRs got " << inputs.size() << ".";
    SDR::initialize( inputs[0]->dimensions );
    inputs_.assign( inputs.begin(), inputs.end() );

    callback_handles_.clear();
    destroyCallback_handles_.clear();
    for(SDR *inp : inputs_) {
        NTA_CHECK(inp != nullptr);
        NTA_ASSERT(inp->size == size)
            << "All inputs to SDR Intersection must have the same size!";
        // When input SDR is assigned to, invalidate this SDR.  This SDR
        // will be recalculated next time it is accessed.
        callback_handles_.push_back( inp->addCallback( [&] ()
            { clear(); }));
        // This SDR can't survive without all of its input SDRs.
        destroyCallback_handles_.push_back( inp->addDestroyCallback( [&] ()
            { deconstruct(); }));
    }
    clear();
}

void Intersection::clear() const {
    SDR::clear();
    // Always advertise that this SDR has dense data.
    dense_valid = true;
    // But make note that this SDR does not actually have dense data, it
    // will be computed it when it's requested.
    dense_valid_lazy = false;
}

SDR_dense_t& Intersection::getDense() const {
    NTA_ASSERT( dense_valid );
    if( !dense_valid_lazy ) {
        const auto &input0 = inputs[0]->getDense();
        dense_.assign( input0.begin(), input0.end() );
        for(auto i = 1u; i < inputs.size(); ++i) {
            const auto &data = inputs[i]->getDense();
            for(auto z = 0u; z < data.size(); ++z)
                dense_[z] = dense_[z] && data[z];
        }
        SDR::setDenseInplace();
        dense_valid_lazy = true;
    }
    return dense_;
}

void Intersection::deconstruct() {
    // Unlink everything at death.
    for(auto i = 0u; i < inputs_.size(); i++) {
        inputs_[i]->removeCallback( callback_handles_[i] );
        inputs_[i]->removeDestroyCallback( destroyCallback_handles_[i] );
    }
    // Clear internal data.
    inputs_.clear();
    callback_handles_.clear();
    destroyCallback_handles_.clear();
    dense_valid_lazy = false;
    // Notify SDR parent class.
    SDR::deconstruct();
}


} // end namespace sdr
} // end namespace nupic
