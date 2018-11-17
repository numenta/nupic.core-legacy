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
 * Definitions for SparseDistributedRepresentation class
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
 * SparseDistributedRepresentation class
 *
 * ### Description
 * This class manages the specification and momentary value of a Sparse
 * Distributed Representation (SDR).  An SDR is a group of boolean values which
 * represent the state of a group of neurons or their associated processes. 
 *
 * This class automatically converts between the commonly used SDR data formats:
 * which are dense, index, and flat-index.  Converted values are cached by this
 * class, so getting a value in one format many times incurs no extra
 * performance cost.  Assigning to the SDR via a setter method will clear these
 * cached values and cause them to be recomputed as needed.
 *
 *    Dense Format: A contiguous array of boolean values, representing all of
 *    the bits in the SDR.  This format allows random-access queries of the SDRs
 *    values.
 *
 *    Index Format: A list of lists of the indices of the true bits in the SDR.
 *    The outter list contains an entry for each dimension in the SDR.  The
 *    inner lists contain the coordinates of each true bit.  The inner lists run
 *    in parallel. This format is useful because it contains the location of
 *    each true bit inside of the SDR's dimensional space.
 *
 *    Flat Index Format: A list of the indices of the true bits, indexed into
 *    the the flattened SDR.  This format allows for quickly accessing all of
 *    the true bits in the SDR.
 *
 * Flat Formats: This class uses C-order throughout, meaning that when iterating
 * through the SDR, the last/right-most index changes fastest.
 *
 * Boolean Data Type: Although the documentation for this class may refer to the
 * values in an SDR as bits they are in fact stored as bytes (of type signed
 * character).
 *
 * Example usage:
 *
 *    // Make an SDR with 10,000 bits, arranged in a (100 x 100) gird.
 *    SparseDistributedRepresentation X({100, 100});
 *
 *    // Assign data in any format.
 *    X.setFlatIndex( mySparseData ); // Assigns mySparseData to SDR's value.
 *    X.setDense( myDenseData );      // myDenseData overwrites mySparseData.
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
     * Create an uninitialized SDR.  The only way to initialize SDRs created by
     * this method is by calling SDR.load().  Until SDR.load() is called this
     * SDR is in an unusable state.
     */
    SparseDistributedRepresentation()
        { clear(); };   // No dimensions, size zero, no value.

    /**
     * Create an SDR object.  Initially this SDR has no value set.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.
     */
    SparseDistributedRepresentation(const vector<UInt> dimensions);

    /**
     * Initialize this SDR as a copy of the given SDR.  This is a deep copy
     * meaning that all of the data in the given SDR is fully copied, and
     * modifying either SDR will not affect the other SDR.
     *
     * @param value An SDR to replicate.
     */
    SparseDistributedRepresentation(const SparseDistributedRepresentation &value) {
        SparseDistributedRepresentation( value.getDimensions() );
        assign(value);
    };

    /**
     * @returns A list of dimensions of the SDR.
     */
    const vector<UInt> getDimensions() const
        {return dimensions;};

    /**
     * @returns The total number of boolean values in the SDR.
     */
    UInt getSize() const
        { return dense.size(); };

    /**
     * Remove the value from this SDR.  Attempting to get the value of an unset
     * SDR will raise an exception.
     */
    void clear();

    /**
     * Set all of the values in the SDR to false.  This method overwrites the
     * SDRs current value.
     */
    void zero();

    /**
     * Assigns a new value to the SDR, overwritting the current value.
     *
     * @param value A dense array in a vector<char> to assign to the SDR.
     */
    void setDense( const vector<Byte> &value );

    /**
     * Assigns a new value to the SDR, overwritting the current value.
     *
     * @param value A dense C-style array of UInt's to assign to the SDR.  The
     * given value must have as many elements as this SDR, as given by 
     * SDR.getSize().
     */
    void setDense( const UInt *value );

    /**
     * Assigns a new value to the SDR, overwritting the current value.
     *
     * @param value A dense byte Array to assign to the SDR.
     */
    void setDense( const ArrayBase *value );

    /**
     * Assigns a vector of sparse indices to the SDR.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A vector of indices to assign to the SDR.
     */
    void setFlatIndex( const vector<UInt> &value );

    /**
     * Assigns an array of sparse indices to the SDR.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A C-style array of indices to assign to the SDR.
     * @param num_values The number of elements in the 'value' array.
     */
    void setFlatIndex( const UInt *value, const UInt num_values );

    /**
     * Assign a list of indices to the SDR, overwritting the SDRs current value.
     * These indices are into the SDR space with dimensions.  The outter list
     * is indexed using an index into the dimensions list, see
     * SDR.getDimensions().  The inner lists are indexed in parallel, they
     * contain the coordinates of the true values in the SDR.
     *
     * @param value A list of lists containing the coordinates of the true
     * values to assign to the SDR.
     */
    void setIndex( const vector<vector<UInt>> &value );

    /**
     * Assigns a deep copy of the given SDR to this SDR.  This overwrites the
     * current value of this SDR.
     *
     * @param value An SDR to copy the value of.
     */
    void assign( const SparseDistributedRepresentation &value );

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value
     * is unset this raises an exception.
     *
     * @returns A list of all values in the SDR.
     */
    const vector<Byte>* getDense();

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value is
     * unset this raises an exception.
     *
     * @returns A list of the indices of the true values in the flattened SDR.
     */
    const vector<UInt>* getFlatIndex();

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value is
     * unset this raises an exception.
     *
     * @returns A list of lists of the coordinates of the true values in the
     * SDR.  The outter list has an entry for each dimension in the SDR.  The
     * inner lists have an entry for each true value in the SDR.
     */
    const vector<vector<UInt>>* getIndex();

    /**
     * Makes a deep copy of the SDR.  This SDR and the returned SDR have no
     * shared data and they can be modified without affecting each other.
     *
     * @returns An SDR which is identical to this SDR.
     */
    SparseDistributedRepresentation* copy() const
        { return new SparseDistributedRepresentation(*this); };

    /**
     * Calculates the sparsity of the SDR, which is the fraction of bits which
     * are true out of the total number of bits in the SDR.
     * I.E.  sparsity = SDR.getFlatIndex.size() / SDR.getSize()
     *
     * If the SDRs value is unset this raises an exception.
     *
     * @returns The fraction of values in the SDR which are true.
     */
    Real getSparsity()
        { return (Real) getFlatIndex()->size() / getSize(); };

    /**
     * Save (serialize) the current state of the SDR to the specified file.
     * 
     * @param stream A valid output stream, such as an open file.
     */
    void save(std::ostream &stream) const override;

    /**
     * Load (deserialize) and initialize the SDR from the specified input
     * stream.
     *
     * @param stream A input valid istream, such as an open file.
     */
    void load(std::istream &stream) override;

private:
    vector<UInt> dimensions;

    vector<Byte>         dense;
    vector<UInt>         flatIndex;
    vector<vector<UInt>> index;

    bool dense_valid;
    bool flatIndex_valid;
    bool index_valid;
};

typedef SparseDistributedRepresentation SDR;

} // end namespace nupic
#endif // end ifndef SDR_HPP
