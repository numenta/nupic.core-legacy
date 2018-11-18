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
 * Also known as "SDR" class
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
 *    // Make an SDR with 10,000 bits, arranged in a (100 x 100) grid.
 *    SDR  X({100, 100});             // SDR is an alias for SparseDistributedRepresentation
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
private:
    vector<UInt> dimensions_;
    UInt         size_;

    Byte*                 dense;
    vector<UInt>*         flatIndex;
    vector<vector<UInt>>* index;

    bool dense_valid_;
    bool flatIndex_valid_;
    bool index_valid_;

    vector<void (*)(SparseDistributedRepresentation*)> callbacks;
    void do_callbacks() {
        for(auto func_ptr : callbacks)
            func_ptr(this);
    };

public:
    /**
     * Create a zero sized SDR.  Use this method in conjunction with sdr.load().
     */
    SparseDistributedRepresentation() {
        SparseDistributedRepresentation(vector<UInt>());
    };

    /**
     * Create an SDR object.  Initially this SDR has no value set.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.
     */
    SparseDistributedRepresentation(const vector<UInt> dimensions) {
        dimensions_ = dimensions;
        // Calculate the SDR's size.
        if( dimensions.size() ) {
            size_ = 1;
            for(UInt dim : dimensions)
                size_ *= dim;
        }
        else {
            size_ = 0;
        }
        // Initialize the dense array storage.
        dense = new Byte[size];
        // Initialize the flatIndex array.
        flatIndex = new vector<UInt>;
        // Initialize the index tuple.
        index = new vector<vector<UInt>>;
        for(UInt dim : dimensions)
            index->push_back( { } );
        // Mark the current data as invalid.
        clear();
    };

    /**
     * Initialize this SDR as a shallow copy of the given SDR.  Modifying either
     * SDR will modify both SDRs.
     *
     * @param value An SDR to connect with.
     */
    SparseDistributedRepresentation(SparseDistributedRepresentation &value);

    /**
     * Initialize this SDR as a deep copy of the given SDR.
     *
     * @param value An SDR to replicate.
     */
    SparseDistributedRepresentation(const SparseDistributedRepresentation &value) {
        NTA_ASSERT( false /* Unimplemented */ );
        SparseDistributedRepresentation( value.dimensions );
        assign(value);
    };

    /**
     * @attribute dimensions A list of dimensions of the SDR.
     */
    const vector<UInt> &dimensions = dimensions_;

    /**
     * @attribute size The total number of boolean values in the SDR.
     */
    const UInt &size = size_;

    /**
     * These flags remember which data formats are up-to-date and which formats
     * need to be updated.  If all of these flags are false then the SDR has no
     * value assigned to it.
     *
     * @attribute dense_valid ...
     * @attribute flatIndex_valid ...
     * @attribute index_valid ...
     */
    const bool &dense_valid     = dense_valid_;
    const bool &flatIndex_valid = flatIndex_valid_;
    const bool &index_valid     = index_valid_;

    /**
     * Remove the value from this SDR by clearing all of the valid flags.  Does
     * not actually change any of the data.  Attempting to get the SDR's value
     * immediately after this operation will raise an exception.
     */
    void clear() {
        dense_valid_     = false;
        flatIndex_valid_ = false;
        index_valid_     = false;
    };

    /**
     * Hook for getting notified after every assignment to the SDR.
     *
     * @param func_ptr A function pointer which is called every time a setter
     * method is called on the SDR.  Function accepts one argument: a pointer to
     * the SDR.
     */
    void addAssignCallback(void (*func_ptr)(SparseDistributedRepresentation*)) {
        callbacks.push_back( func_ptr );
    };

    /**
     * Set all of the values in the SDR to false.  This method overwrites the
     * SDRs current value.
     */
    void zero() {
        clear();
        flatIndex->clear();
        flatIndex_valid_ = true;
        do_callbacks();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense array of type char to copy into the SDR.
     */
    void setDenseCopy( const Byte *value ) {
        NTA_ASSERT(value != NULL);
        std::copy(value, value + size, dense);
        setDenseInplace();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense C-style array of UInt's to copy into the SDR.
     */
    void setDenseCopy( const UInt *value ) {
        NTA_ASSERT(value != NULL);
        std::copy(value, value + size, dense);
        setDenseInplace();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense byte Array to copy into the SDR.
     */
    void setDenseCopy( const ArrayBase *value )  {
        NTA_ASSERT( false /* Unimplemented */ );
        // TODO: Assert correct size and data type.
        setDenseInplace();
    };

    /**
     * Assign a new value to the SDR by replacing the dense pointer.  This
     * overwrites the current SDR value.
     *
     * @param newDensePtr A pointer to an array containing dense-format data.
     * This array will be used from here on.
     *
     * @returns A pointer to the old array which is being replaced.  It is the
     * callers responsibility to deallocate the old array if necessary.
     */
    Byte *setDensePtr(Byte *newDensePtr) {
        auto oldDensePtr = dense;
        dense = newDensePtr;
        setDenseInplace();
        return oldDensePtr;
    };

    /**
     * Update the SDR to reflect the value currently inside of the dense array.
     * Use this method after modifying the dense buffer inplace, in order to
     * propigate any changes to the index & flatIndex formats.
     */
    void setDenseInplace() {
        clear();
        dense_valid_ = true;
        do_callbacks();
    };

    /**
     * Copy a vector of sparse indices into the SDR.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A vector of flat indices to copy into the SDR.
     */
    void setFlatIndexCopy( const vector<UInt> &value ) {
        flatIndex->assign( value.begin(), value.end() );
        setFlatIndexInplace();
    };

    /**
     * Copy an array of sparse indices into the SDR.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A C-style array of indices to copy into the SDR.
     * @param num_values The number of elements in the 'value' array.
     */
    void setFlatIndexCopy( const UInt *value, const UInt num_values ) {
        flatIndex->assign( value, value + num_values );
        setFlatIndexInplace();
    };

    /**
     * Assign a new value to the SDR by replacing the flatIndex pointer.  This
     * overwrites the current SDR value.
     *
     * @param newFlatIndexPtr A pointer to a vector containing flat index format
     * data.  This vector will be used from here on.
     *
     * @returns A pointer to the old vector which is being replaced.  It is the
     * callers responsibility to deallocate the old vector if necessary.
     */
    vector<UInt>* setFlatIndexPtr(vector<UInt>* newFlatIndexPtr) {
        auto oldFlatIndexPtr = flatIndex;
        flatIndex = newFlatIndexPtr;
        setFlatIndexInplace();
        return oldFlatIndexPtr;
    };

    /**
     * Update the SDR to reflect the value currently inside of the flatIndex
     * vector. Use this method after modifying the flatIndex vector inplace, in
     * order to propigate any changes to the dense & index formats.
     */
    void setFlatIndexInplace() {
        NTA_ASSERT(flatIndex->size() <= size);
        for(auto idx : *flatIndex) {
            NTA_ASSERT(idx < size);
        }
        clear();
        flatIndex_valid_ = true;
        do_callbacks();
    };

    /**
     * Copy a list of indices into the SDR, overwritting the SDRs current value.
     * These indices are into the SDR space with dimensions.  The outter list is
     * indexed using an index into the sdr.dimensions list.  The inner lists are
     * indexed in parallel, they contain the coordinates of the true values in
     * the SDR.
     *
     * @param value A list of lists containing the coordinates of the true
     * values to copy into the SDR.
     */
    void setIndexCopy( const vector<vector<UInt>> &value ) {
        for(UInt dim = 0; dim < dimensions.size(); dim++) {
            index->at(dim).assign( value[dim].begin(), value[dim].end() );
        }
        setIndexInplace();
    };

    /**
     * Assign a new value to the SDR by replacing the index pointer.  This
     * overwrites the current SDR value.
     *
     * @param newIndexPtr A pointer to a vector containing index format data.
     * This vector will be used from here on.
     *
     * @returns A pointer to the old vector which is being replaced.  It is the
     * callers responsibility to deallocate the old vector if necessary.
     */
    vector<vector<UInt>>* setIndexPtr(vector<vector<UInt>>* newIndexPtr) {
        auto oldIndexPtr = index;
        index = newIndexPtr;
        setIndexInplace();
        return oldIndexPtr;
    }

    /**
     * Update the SDR to reflect the value currently inside of the index
     * vector. Use this method after modifying the index vector inplace, in
     * order to propigate any changes to the dense & flatIndex formats.
     */
    void setIndexInplace() {
        NTA_ASSERT(index->size() == dimensions.size());
        for(UInt dim = 0; dim < dimensions.size(); dim++) {
            const auto coord_vec = index->at(dim);
            NTA_ASSERT(coord_vec.size() <= size);
            // TODO: Assert that all inner vectors have the same length.
            for(auto idx : coord_vec) {
                NTA_ASSERT(idx < size);
            }
        }
        clear();
        index_valid_ = true;
        do_callbacks();
    };

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
     * @returns A pointer to an array of all the values in the SDR.
     */
    const Byte* getDense()
        { return getDenseMutable(); };

    /**
     * This method does the same thing as sdr.getDense() except that it returns
     * a non-constant pointer.  After modifying the dense array you MUST call
     * sdr.setDenseInplace() in order to notify the SDR that its dense array has
     * changed and its cached data is out of date.
     *
     * @returns A pointer to an array of all the values in the SDR.
     */
    Byte* getDenseMutable() {
        if( !dense_valid ) {
            std::fill(dense, dense + size, 0);
            // Convert from flatIndex to dense.
            for(auto idx : *getFlatIndex()) {
                dense[idx] = 1;
            }
            dense_valid_ = true;
        }
        return dense;
    };

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value is
     * unset this raises an exception.
     *
     * @returns A pointer to a vector of the indices of the true values in the
     * flattened SDR.
     */
    const vector<UInt>* getFlatIndex()
        { return getFlatIndexMutable(); };

    /**
     * This method does the same thing as sdr.getFlatIndex() except that it
     * returns a non-constant pointer.  After modifying the flatIndex vector you
     * MUST call sdr.setFlatIndexInplace() in order to notify the SDR that its
     * flatIndex vector has changed and its cached data is out of date.
     *
     * @returns A pointer to a vector of the indices of the true values in the
     * flattened SDR.
     */
    vector<UInt>* getFlatIndexMutable() {
        if( !flatIndex_valid ) {
            flatIndex->clear(); // Clear out any old data.
            if( index_valid ) {
                // Convert from index to flatIndex.
                const auto num_nz = index->at(0).size();
                flatIndex->reserve( num_nz );
                for(UInt nz = 0; nz < num_nz; nz++) {
                    UInt flat = 0;
                    for(UInt i = 0; i < dimensions.size(); i++) {
                        flat *= dimensions[i];
                        flat += (*index)[i][nz];
                    }
                    flatIndex->push_back(flat);
                }
                flatIndex_valid_ = true;
            }
            else if( dense_valid ) {
                // Convert from dense to flatIndex.
                for(UInt idx = 0; idx < size; idx++)
                    if( dense[idx] != 0 )
                        flatIndex->push_back( idx );
                flatIndex_valid_ = true;
            }
            else
                throw logic_error("Can not get value from empty SDR.");
        }
        return flatIndex;
    };

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value is
     * unset this raises an exception.
     *
     * @returns A pointer to a list of lists of the coordinates of the true
     * values in the SDR.
     */
    const vector<vector<UInt>>* getIndex()
        { return getIndexMutable(); };

    /**
     * This method does the same thing as sdr.getIndex() except that it
     * returns a non-constant pointer.  After modifying the index vector you
     * MUST call sdr.setIndexInplace() in order to notify the SDR that its
     * index vector has changed and its cached data is out of date.
     *
     * @returns A pointer to a list of lists of the coordinates of the true
     * values in the SDR.
     */
    vector<vector<UInt>>* getIndexMutable() {
        if( !index_valid ) {
            // Clear out any old data.
            for( auto vec : *index ) {
                vec.clear();
            }
            // Convert from flatIndex to index.
            for( auto idx : *getFlatIndex() ) {
                for(UInt dim = dimensions.size() - 1; dim > 0; dim--) {
                    auto dim_sz = dimensions[dim];
                    (*index)[dim].push_back( idx % dim_sz );
                    idx /= dim_sz;
                }
                (*index)[0].push_back(idx);
            }
            index_valid_ = true;
        }
        return index;
    };

    /**
     * Query the value of the SDR at a single location.
     * If the SDRs value is unset this raises an exception.
     *
     * @param coordinates A list of coordinates into the SDR space to query.
     *
     * @returns The value of the SDR at the given location.
     */
    Byte at(const vector<UInt> coordinates);

    /**
     * Makes a deep copy of the SDR.  This SDR and the returned SDR have no
     * shared data and they can be modified without affecting each other.
     *
     * @returns An SDR which is identical to this SDR.
     */
    SparseDistributedRepresentation* copy() const
        { return new SparseDistributedRepresentation((const) this); };

    /**
     * Calculates the sparsity of the SDR, which is the fraction of bits which
     * are true out of the total number of bits in the SDR.
     * I.E.  sparsity = sdr.getFlatIndex.size() / sdr.size
     *
     * If the SDRs value is unset this raises an exception.
     *
     * @returns The fraction of values in the SDR which are true.
     */
    Real getSparsity()
        { return (Real) getFlatIndex()->size() / size; };

    /**
     * Make a random SDR, overwriting the current value of the SDR.  The
     * resulting has uniformly random activations.
     *
     * @param sparsity The sparsity of the randomly generated SDR.
     */
    void randomize(Real sparsity);

    /**
     * Modify the SDR by moving a fraction of the active bits to different
     * locations.  This method does not change the sparsity of the SDR, it only
     * changes which bits are active.  The resulting SDR has a controlled amount
     * of overlap with the original.
     *
     * If the SDRs value is unset this raises an exception.
     *
     * @param fractionNoise The fraction of active bits to swap out.  The
     * original and resulting SDRs have an overlap of (1 - fractionNoise).
     */
    void addNoise(Real fractionNoise);

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
};

typedef SparseDistributedRepresentation SDR;

} // end namespace nupic
#endif // end ifndef SDR_HPP
