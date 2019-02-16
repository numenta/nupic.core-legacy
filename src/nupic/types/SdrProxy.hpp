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
 * Definitions for SparseDistributedRepresentation class
 */

#ifndef SDR_PROXY_HPP
#define SDR_PROXY_HPP

#include <vector>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>

using namespace std;

namespace nupic {

/**
 * SDR_Proxy class
 *
 * ### Description

 * SDR_Proxy presents a view onto an SDR.
 *      + Proxies have the same value as their source SDR, at all times and
 *        automatically.
 *      + SDR_Proxy is a subclass of SDR and be safely typecast to an SDR.
 *      + Proxies can have different dimensions than their source SDR.
 *      + Proxies are read only.
 *
 * SDR and SDR_Proxy classes tell each other when they are created and
 * destroyed.  Proxies can be created and destroyed as needed.  Proxies will
 * throw an exception if they are used after their source SDR has been
 * destroyed.
 *
 * Example Usage:
 *      // Convert SDR dimensions from (4 x 4) to (8 x 2)
 *      SDR       A(    { 4, 4 })
 *      SDR_Proxy B( A, { 8, 2 })
 *      A.setSparse( {1, 1, 2}, {0, 1, 2}} )
 *      auto sparse = B.getSparse()  ->  {{2, 2, 5}, {0, 1, 0}}
 *
 * SDR_Proxy partially supports the Serializable interface.  SDR_Proxies can be
 * saved but can not be loaded.
 */
class SDR_Proxy : public SDR
{
public:
    /**
     * Create an SDR_Proxy object.
     *
     * @param sdr Source SDR to make a view of.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.  Optional, if not given then this Proxy will have the same
     * dimensions as the given SDR.
     */
    SDR_Proxy(SDR &sdr)
        : SDR_Proxy(sdr, sdr.dimensions)
        {}

    SDR_Proxy(SDR &sdr, const vector<UInt> &dimensions)
        : SDR( dimensions ) {
        clear();
        parent = &sdr;
        NTA_CHECK( size == parent->size ) << "SDR Proxy must have same size as given SDR.";
        callback_handle = parent->addCallback( [&] () {
            clear();
            do_callbacks();
        });
        destroyCallback_handle = parent->addDestroyCallback( [&] () {
            deconstruct();
        });
    }

    ~SDR_Proxy() override
        { deconstruct(); }

    SDR_dense_t& getDense() const override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        return parent->getDense();
    }

    SDR_flatSparse_t& getFlatSparse() const override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        return parent->getFlatSparse();
    }

    SDR_sparse_t& getSparse() const override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        if( dimensions.size() == parent->dimensions.size() &&
            equal( dimensions.begin(), dimensions.end(),
                   parent->dimensions.begin() )) {
            // All things equal, prefer reusing the parent's cached value.
            return parent->getSparse();
        }
        else {
            // Don't override getSparse().  It will call either getDense() or
            // getFlatSparse() to get its data, and will use this proxies
            // dimensions.
            return SDR::getSparse();
        }
    }

    void save(std::ostream &outStream) const override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        parent->save( outStream );
    }

    void load(std::istream &inStream) override {
        NTA_THROW << "Can not load into SDR_Proxy, SDR_Proxy is read only.";
    }

protected:
    /**
     * This SDR shall always have the same value as the parent SDR.
     */
    SDR *parent;
    UInt callback_handle;
    UInt destroyCallback_handle;

    void deconstruct() override {
        // Unlink this SDR from the parent SDR.
        if( parent != nullptr ) {
            parent->removeCallback( callback_handle );
            parent->removeDestroyCallback( destroyCallback_handle );
            parent = nullptr;
            SDR::deconstruct();
        }
    }

    const string _SDR_Proxy_setter_error_message = "SDR_Proxy is read only.";

    void setDenseInplace() const override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setFlatSparseInplace() const override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setSparseInplace() const override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setSDR( const SparseDistributedRepresentation &value ) override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
};


// class SDR_Join : public SDR
// {
// private:
//     vector<SDR&> inputs_;
//     UInt         axis_;
//     vector<UInt> callback_handles_;
//     vector<UInt> destroyCallback_handles_;
// public:
//     const vector<SDR&> &inputs = inputs_;
//     const UInt         &axis   = axis_;

//     SDR_Join(vector<SDR&> inputs, axis=0)
//     : inputs_(inputs), axis_(axis)
//     {
//         // Compute dimensions

//         initialize( dims );

//         for(auto &inp : inputs_) {
//             // When input SDR is assigned to, invalidate this SDR.  It will be
//             // recalculated next time it is accessed.
//                      TODO

//     SDR_dense_t& getDense() const override {
//         if( !denseValid ) {

//             vector<SDR_dense_t&> inputs;
//             vector<UInt> strides;
//             vector<UInt> row_lengths;
//             vector<UInt> input_offsets( inputs_, 0u );

//             for(const auto &sdr : inputs_) {
//                 inputs.push_back( sdr );
//                 strides.push_back( sdr.dimensions );
//             }
//             dense_.resize( self.size );

//             // TODO...

//             setDenseInplace();
//         }
//         return &denseValid;
//     }
//     // SDR_flatSparse_t& getFlatSparse() const override;
// }



/**
 * TODO DOCUEMNTATION
 */
class SDR_Intersection : public SDR
{
protected:
    vector<SDR*> inputs_;
    vector<UInt> callback_handles_;
    vector<UInt> destroyCallback_handles_;
    mutable bool dense_valid_lazy;

    void clear() const override {
        SDR::clear();
        // Always advertise that this SDR has dense data.
        dense_valid = true;
        // But make note that this SDR does not actually have dense data, it
        // will be computed it when it's requested.
        dense_valid_lazy = false;
    }

    void deconstruct() override {
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

public:
    const vector<SDR*> &inputs = inputs_;

    SDR_Intersection(SDR &input1, SDR &input2)
        { initialize({   &input1,     &input2}); }
    SDR_Intersection(SDR &input1, SDR &input2, SDR &input3)
        { initialize({   &input1,     &input2,     &input3}); }
    SDR_Intersection(SDR &input1, SDR &input2, SDR &input3, SDR &input4)
        { initialize({   &input1,     &input2,     &input3,     &input4}); }

    SDR_Intersection(vector<SDR*> inputs)
        { initialize(inputs); }

    void initialize(const vector<SDR*> inputs)
    {
        NTA_CHECK( inputs.size() >= 2u )
            << "Not enough inputs to SDR_Intersection, need at least 2 SDRs got " << inputs.size() << ".";
        SDR::initialize( inputs[0]->dimensions );
        inputs_.assign( inputs.begin(), inputs.end() );

        callback_handles_.clear();
        destroyCallback_handles_.clear();
        for(SDR *inp : inputs_) {
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

    ~SDR_Intersection()
        { deconstruct(); }

    SDR_dense_t& getDense() const override {
        NTA_ASSERT( dense_valid );
        if( !dense_valid_lazy ) {
            const auto &input0 = inputs[0]->getDense();
            dense_.assign( input0.begin(), input0.end() );
            for(auto i = 1u; i < inputs.size(); ++i) {
                const auto &data = inputs[i]->getDense();
                for(auto z = 0u; z < data.size(); ++z)
                    dense_[z] = dense_[z] && data[z];
            }
            setDenseInplace();
            dense_valid_lazy = true;
        }
        return dense_;
    }
};

} // end namespace nupic
#endif // end ifndef SDR_PROXY_HPP
