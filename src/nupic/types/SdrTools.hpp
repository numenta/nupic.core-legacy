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
 * Definitions for SDR tools & helper classes
 */

#ifndef SDR_TOOLS_HPP
#define SDR_TOOLS_HPP

#include <vector>
#include <nupic/types/Sdr.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/types/Serializable.hpp>

namespace nupic {
namespace sdr {


class ReadOnly_ : public SDR
{
public:
    ReadOnly_() {}

    ReadOnly_( const std::vector<UInt> dimensions )
        : SDR( dimensions ) {}

private:
    const std::string _error_message = "This SDR is read only.";

    void setDenseInplace() const override
        { NTA_THROW << _error_message; }
    void setSparseInplace() const override
        { NTA_THROW << _error_message; }
    void setCoordinatesInplace() const override
        { NTA_THROW << _error_message; }
    void setSDR( const SparseDistributedRepresentation &value ) override
        { NTA_THROW << _error_message; }
    void load(std::istream &inStream) override
        { NTA_THROW << _error_message; }
};


/**
 * Reshape class
 *
 * ### Description
 * Reshape presents a view onto an SDR with different dimensions.
 *      + Reshape is a subclass of SDR and be safely typecast to an SDR.
 *      + The resulting SDR has the same value as the source SDR, at all times
 *        and automatically.
 *      + The resulting SDR is read only.
 *
 * SDR and Reshape classes tell each other when they are created and
 * destroyed.  Reshape can be created and destroyed as needed.  Reshape
 * will throw an exception if it is used after its source SDR has been
 * destroyed.
 *
 * Example Usage:
 *      // Convert SDR dimensions from (4 x 4) to (8 x 2)
 *      SDR     A(    { 4, 4 })
 *      Reshape B( A, { 8, 2 })
 *      A.setCoordinates( {1, 1, 2}, {0, 1, 2}} )
 *      B.getCoordinates()  ->  {{2, 2, 5}, {0, 1, 0}}
 *
 * Reshape partially supports the Serializable interface.  Reshape can
 * be saved but can not be loaded.
 *
 * Note: Reshape used to be called SDR_Proxy. See PR #298
 */
class Reshape : public ReadOnly_
{
public:
    /**
     * Reshape an SDR.
     *
     * @param sdr Source SDR to make a view of.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.  Optional, if not given then this SDR will have the same
     * dimensions as the given SDR.
     */
    Reshape(SDR &sdr, const std::vector<UInt> &dimensions);

    Reshape(SDR &sdr)
        : Reshape(sdr, sdr.dimensions) {}

    SDR_dense_t& getDense() const override;

    SDR_sparse_t& getSparse() const override;

    SDR_coordinate_t& getCoordinates() const override;

    void save(std::ostream &outStream) const override;

    ~Reshape() override
        { deconstruct(); }

protected:
    /**
     * This SDR shall always have the same value as the parent SDR.
     */
    SDR *parent;
    UInt callback_handle;
    UInt destroyCallback_handle;

    void deconstruct() override;
};

} // end namespace sdr
} // end namespace nupic
#endif // end ifndef SDR_TOOLS_HPP
