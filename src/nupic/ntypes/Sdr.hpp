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
 * Definitions for SparseDistributedRepresentation in C++
 */

#ifndef SDR_HPP
#define SDR_HPP

#include <vector>
#include <nupic/types/Types.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/Serializable.hpp>

using namespace std;

namespace nupic {

/**
 * SparseDistributedRepresentation in C++
 *
 * ### Description
 * This class manages the specification and momentary value of a Sparse
 * Distributed Representation (SDR).  An SDR is a group of boolean values which
 * represent the state of a group of neurons or their associated processes. This
 * class automatically converts between the commonly used SDR data formats:
 * which are dense, index, and flat-index.  Converted values are cached by this
 * class, so getting a value in one format many times incurs no extra
 * performance cost.  This class uses C-order throughout, meaning that when
 * iterating through the SDR, the last/right-most index changes fastest.
 *
 * Dense Format: A contiguous array of boolean values, representing all of the
 * bits in the SDR.
 *
 * Index Format: A list of lists of the indices of the true bits in the SDR. The
 * outter list contains an entry for each dimension in the SDR.  The inner lists
 * contain the coordinates of each true bit.  The inner lists run in parallel.
 *
 * Flat Index Format: A list of the indices of the true bits, indexed into the
 * the flattened SDR.
 *
 * Example usage:
 *
 *    // Make an SDR with 10,000 bits, arranged in a (100 x 100) gird.
 *    SparseDistributedRepresentation X({100, 100});
 *
 *    // Assign data in any format.
 *    X.setFlatIndex( myFlatIndex ); // Assigns myFlatIndex to SDR's value.
 *    X.setDense( myDenseData );     // myDenseData overwrites myFlatIndex.
 *
 *    // Access data in different formats.
 *    indices      = X.getFlatIndex(); // Calculated from myDenseData.
 *    indicesAgain = X.getFlatIndex(); // Reuses previous result, no work done.
 *
 */
class SparseDistributedRepresentation : public Serializable
{
public:
    /**
     *
     */
    SparseDistributedRepresentation();

    /**
     *
     * @param
     */
    SparseDistributedRepresentation(const vector<UInt> dimensions);

    /**
     * Initialize this SDR as a copy of the given SDR.  This is a deep copy
     * meaning that all of the data in the given SDR is fully copied.
     *
     * @param value An SDR to replicate.
     */
    SparseDistributedRepresentation(const SparseDistributedRepresentation value);

    /**
     * @returns A vector of dimensions for the SDR.
     */
    const vector<UInt> getDimensions() const
        {return dimensions;};

    /**
     * @returns The total number of bits in the SDR.
     */
    const UInt getSize() const
        {return size;};

    /**
     * Remove the value from this SDR.  Attempting to get the value of an empty
     * SDR will raise an exception.
     */
    void clear();

    /**
     * Set all of the bits in the SDR to 0's.
     */
    void zero();

    /**
     * @param value A dense boolean vector to assign to the SDR.
     */
    void setDense( const vector<bool> value );

    /**
     * @param value A dense C-style array of UInt's to assign to the SDR.
     */
    void setDense( const UInt *value );

    /**
     * @param value A dense boolean Array to assign to the SDR.
     */
    void setDense( const ArrayBase *value );

    /**
     * Assign a vector of sparse indices to the SDR.
     *
     * @param value A vector of indices to assign to the SDR.
     */
    void setFlatIndex( const vector<UInt> value );

    /**
     * @param value A C-style array of indices to assign to the SDR.
     * @param num_values The number of elements in the 'value' array.
     */
    void setFlatIndex( const UInt *value, const UInt num_values );

    /**
     *
     */
    void setIndex( const vector<vector<UInt>> value );

    /**
     * @param value An SDR to copy the value of.
     */
    void assign( const SparseDistributedRepresentation value );

    /**
     * @returns TODO
     */
    const vector<bool> getDense();

    /**
     * @returns TODO
     */
    const vector<UInt> getFlatIndex();

    /**
     * @returns TODO
     */
    const vector<vector<UInt>> getIndex();

    /**
     * @returns TODO
     */
    const SparseDistributedRepresentation copy() const;

    /**
     * @returns The fraction of bits in the SDR which are 1's.
     */
    const Real getSparsity();

    /**
     * Save (serialize) the current state of the SDR to the specified file.
     * 
     * @param stream A valid output stream, such as an open file.
     */
    void save(std::ostream &stream) const override;

    /**
     * Load (deserialize) and initialize the SDR from the
     * specified input stream.
     *
     * @param stream A input valid istream, such as an open file.
     */
    void load(std::istream &stream) override;

private:
    vector<UInt> dimensions;
    size_t       size;

    vector<bool>         dense;
    vector<UInt>         flatIndex;
    vector<vector<UInt>> index;

    bool dense_valid;
    bool flatIndex_valid;
    bool index_valid;
};

typdef SDR SparseDistributedRepresentation;

} // end namespace nupic
#endif // end ifndef SDR_HPP
