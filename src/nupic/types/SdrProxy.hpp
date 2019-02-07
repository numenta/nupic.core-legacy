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

    void setDenseInplace() override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setFlatSparseInplace() override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setSparseInplace() override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
    void setSDR( const SparseDistributedRepresentation &value ) override
        { NTA_THROW << _SDR_Proxy_setter_error_message; }
};

} // end namespace nupic
#endif // end ifndef SDR_PROXY_HPP
