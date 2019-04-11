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

} // end namespace sdr
} // end namespace nupic
