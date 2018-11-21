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

#define SERIALIZE_VERSION 1

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
 *    Sparse Flat Index Format: Contains the indices of only the true values in
 *    the SDR.  This is a list of the indices, indexed into the flattened SDR.
 *    This format allows for quickly accessing all of the true bits in the SDR.
 *
 *    Sparse Index Format: Contains the indices of only the true values in the
 *    SDR.  This is a A list of lists: the outter list contains an entry for
 *    each dimension in the SDR. The inner lists contain the coordinates of each
 *    true bit.  The inner lists run in parallel. This format is useful because
 *    it contains the location of each true bit inside of the SDR's dimensional
 *    space.
 *
 * Flat Formats: This class uses C-order throughout, meaning that when iterating
 * through the SDR, the last/right-most index changes fastest.
 *
 * Example usage:
 *
 *    // Make an SDR with 9 values, arranged in a (3 x 3) grid.
 *    // "SDR" is an alias/typedef for SparseDistributedRepresentation.
 *    SDR  X( {3, 3} );
 *
 *    // These three statements are equivalent.
 *    X.setDense({ 0, 1, 0,
 *                 0, 1, 0,
 *                 0, 0, 1 });
 *    X.setFlatIndex({ 1, 4, 8 });
 *    X.setIndex({{ 0, 1, 2,}, { 1, 1, 2 }});
 *
 *    // Access data in any format, SDR will automatically convert data formats.
 *    X.getDense()     -> { 0, 1, 0, 0, 1, 0, 0, 0, 1 }
 *    X.getIndex()     -> {{ 0, 1, 2 }, {1, 1, 2}}
 *    x.getFlatIndex() -> { 1, 4, 8 }
 *
 *    // Data format conversions are cached, and when an SDR value changes the
 *    // cache is cleared.
 *    X.setFlatIndex({});  // Assign new data to the SDR, clearing the cache.
 *    X.getDense();        // This line will convert formats.
 *    X.getDense();        // This line will resuse the result of the previous line
 */

class SparseDistributedRepresentation : public Serializable
{
private:
    vector<UInt> dimensions_;
    UInt         size_;

    vector<Byte>         dense;
    vector<UInt>         flatIndex;
    vector<vector<UInt>> index;

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
     * Create an SDR object.  Initially this SDR has no value set.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.
     */
    SparseDistributedRepresentation( const vector<UInt> dimensions = vector<UInt>(0) ) {
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
        dense = vector<Byte>(size_);
        // Initialize the flatIndex array, nothing to do.
        // Initialize the index tuple.
        index.assign( dimensions.size(), {} );
        // Mark the current data as invalid.
        clear();
    };

    /**
     * Initialize this SDR as a deep copy of the given SDR.  This SDR and the
     * given SDR will have no shared data and they can be modified without
     * affecting each other.
     *
     * @param value An SDR to replicate.
     */
    SparseDistributedRepresentation(const SparseDistributedRepresentation &value)
        : SparseDistributedRepresentation( value.dimensions ) {
        setSDR(value);
    };

    ~SparseDistributedRepresentation() {};

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
     * @returns Boolean, true if the SDR has a value assigned to it, false if
     * there is no data in the SDR.
     */
    bool hasValue() const {
        return dense_valid or flatIndex_valid or index_valid;
    }

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
        flatIndex.clear();
        flatIndex_valid_ = true;
        do_callbacks();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense vector<char> to copy into the SDR.
     */
    void setDense( const vector<Byte> &value ) {
        NTA_ASSERT(value.size() == size);
        dense.assign( value.begin(), value.end() );
        setDenseInplace();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense array of type char to copy into the SDR.
     */
    void setDense( const Byte *value ) {
        NTA_ASSERT(value != NULL);
        dense.assign( value, value + size );
        setDenseInplace();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense C-style array of UInt's to copy into the SDR.
     */
    void setDense( const UInt *value ) {
        NTA_ASSERT(value != NULL);
        dense.assign( value, value + size );
        setDenseInplace();
    };

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense byte Array to copy into the SDR.
     */
    void setDense( const ArrayBase *value )  {
        NTA_ASSERT( false /* Unimplemented */ );
        // TODO: Assert correct size and data type.
        setDenseInplace();
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
    void setFlatIndex( const vector<UInt> &value ) {
        flatIndex.assign( value.begin(), value.end() );
        setFlatIndexInplace();
    };

    /**
     * Copy an array of sparse indices into the SDR.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A C-style array of indices to copy into the SDR.
     * @param num_values The number of elements in the 'value' array.
     */
    void setFlatIndex( const UInt *value, const UInt num_values ) {
        flatIndex.assign( value, value + num_values );
        setFlatIndexInplace();
    };

    // TODO: Overload setFlatIndex to accept an Array ...

    /**
     * Update the SDR to reflect the value currently inside of the flatIndex
     * vector. Use this method after modifying the flatIndex vector inplace, in
     * order to propigate any changes to the dense & index formats.
     */
    void setFlatIndexInplace() {
        NTA_ASSERT(flatIndex.size() <= size);
        for(auto idx : flatIndex) {
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
    void setIndex( const vector<vector<UInt>> &value ) {
        NTA_ASSERT(value.size() == dimensions.size());
        for(UInt dim = 0; dim < dimensions.size(); dim++) {
            index[dim].assign( value[dim].begin(), value[dim].end() );
        }
        setIndexInplace();
    };

    /**
     * Update the SDR to reflect the value currently inside of the index
     * vector. Use this method after modifying the index vector inplace, in
     * order to propigate any changes to the dense & flatIndex formats.
     */
    void setIndexInplace() {
        // Check data is valid.
        NTA_ASSERT(index.size() == dimensions.size());
        for(UInt dim = 0; dim < dimensions.size(); dim++) {
            const auto coord_vec = index[dim];
            NTA_ASSERT(coord_vec.size() <= size);
            NTA_ASSERT(coord_vec.size() == index[0].size()); // All coordinate vectors have same size.
            for(auto idx : coord_vec) {
                NTA_ASSERT(idx < dimensions[dim]);
            }
        }
        // Do the setter assignment.
        clear();
        index_valid_ = true;
        do_callbacks();
    };

    /**
     * Deep Copy the given SDR to this SDR.  This overwrites the current value of
     * this SDR.  This SDR and the given SDR will have no shared data and they
     * can be modified without affecting each other.
     *
     * @param value An SDR to copy the value of.
     */
    void setSDR( const SparseDistributedRepresentation &value ) {
        NTA_ASSERT( value.dimensions == dimensions );
        clear();

        dense_valid_ = value.dense_valid;
        if( dense_valid ) {
            dense.assign( value.dense.begin(), value.dense.end() );
        }
        flatIndex_valid_ = value.flatIndex_valid;
        if( flatIndex_valid ) {
            flatIndex.assign( value.flatIndex.begin(), value.flatIndex.end() );
        }
        index_valid_ = value.index_valid;
        if( index_valid ) {
            for(UInt dim = 0; dim < dimensions.size(); dim++)
                index[dim].assign( value.index[dim].begin(), value.index[dim].end() );
        }
        do_callbacks();
    };

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  If the value
     * is unset this raises an exception.
     *
     * @returns A pointer to an array of all the values in the SDR.
     */
    const vector<Byte>& getDense()
        { return getDenseMutable(); };

    /**
     * This method does the same thing as sdr.getDense() except that it returns
     * a non-constant pointer.  After modifying the dense array you MUST call
     * sdr.setDenseInplace() in order to notify the SDR that its dense array has
     * changed and its cached data is out of date.
     *
     * @returns A pointer to an array of all the values in the SDR.
     */
    vector<Byte>& getDenseMutable() {
        if( !dense_valid ) {
            // Convert from flatIndex to dense.
            dense.assign( size, 0 );
            for(const auto idx : getFlatIndex()) {
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
    const vector<UInt>& getFlatIndex()
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
    vector<UInt>& getFlatIndexMutable() {
        if( !flatIndex_valid ) {
            flatIndex.clear(); // Clear out any old data.
            if( index_valid ) {
                // Convert from index to flatIndex.
                const auto num_nz = size ? index[0].size() : 0;
                flatIndex.reserve( num_nz );
                for(UInt nz = 0; nz < num_nz; nz++) {
                    UInt flat = 0;
                    for(UInt dim = 0; dim < dimensions.size(); dim++) {
                        flat *= dimensions[dim];
                        flat += index[dim][nz];
                    }
                    flatIndex.push_back(flat);
                }
                flatIndex_valid_ = true;
            }
            else if( dense_valid ) {
                // Convert from dense to flatIndex.
                for(UInt idx = 0; idx < size; idx++)
                    if( dense[idx] != 0 )
                        flatIndex.push_back( idx );
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
    const vector<vector<UInt>>& getIndex()
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
    vector<vector<UInt>>& getIndexMutable() {
        if( !index_valid ) {
            // Clear out any old data.
            for( auto& vec : index ) {
                vec.clear();
            }
            // Convert from flatIndex to index.
            for( auto idx : getFlatIndex() ) {
                for(UInt dim = dimensions.size() - 1; dim > 0; dim--) {
                    auto dim_sz = dimensions[dim];
                    index[dim].push_back( idx % dim_sz );
                    idx /= dim_sz;
                }
                index[0].push_back(idx);
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
    Byte at(const vector<UInt> &coordinates) {
        NTA_ASSERT( size > 0 );
        UInt flat = 0;
        for(UInt i = 0; i < dimensions.size(); i++) {
            NTA_ASSERT( coordinates[i] < dimensions[i] );
            flat *= dimensions[i];
            flat += coordinates[i];
        }
        return getDense()[flat];
    }

    /**
     * Calculates the number of true / non-zero values in the SDR.  If the SDRs
     * value is unset this raises an exception.
     *
     * @returns The number of true values in the SDR.
     */
    UInt getSum()
        { return getFlatIndex().size(); };

    /**
     * Calculates the sparsity of the SDR, which is the fraction of bits which
     * are true out of the total number of bits in the SDR.
     * I.E.  sparsity = sdr.getSum() / sdr.size
     *
     * If the SDRs value is unset this raises an exception.
     *
     * @returns The fraction of values in the SDR which are true.
     */
    Real getSparsity()
        { return (Real) getSum() / size; };

    /**
     * TODO ...
     *
     * @returns ...
     */
    UInt overlap(SparseDistributedRepresentation &sdr) {
        NTA_ASSERT( false /* Unimplemented */ );
        return 0;
    };

    /**
     * Make a random SDR, overwriting the current value of the SDR.  The
     * resulting has uniformly random activations.
     *
     * @param sparsity The sparsity of the randomly generated SDR.
     */
    void randomize(Real sparsity) {
        NTA_ASSERT( false /* Unimplemented */ );
    };

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
    void addNoise(Real fractionNoise) {
        NTA_ASSERT( false /* Unimplemented */ );
    };

    bool operator==(SparseDistributedRepresentation &sdr) {
        // Check attributes
        if( sdr.size != size or dimensions.size() != sdr.dimensions.size() )
            return false;
        for( UInt i = 0; i < dimensions.size(); i++ ) {
            if( dimensions[i] != sdr.dimensions[i] )
                return false;
        }
        if( hasValue() != sdr.hasValue() )
            return false;
        if( !hasValue() )
            return true;

        // Check data
        return std::equal(
            getDense().begin(),
            getDense().end(), 
            sdr.getDense().begin());
    };

    /**
     * Save (serialize) the current state of the SDR to the specified file.
     * 
     * @param stream A valid output stream, such as an open file.
     */
    void save(std::ostream &outStream) const override {

        auto writeVector = [&outStream] (const vector<UInt> &vec) {
            outStream << vec.size() << " ";
            for( auto elem : vec ) {
                outStream << elem << " ";
            }
            outStream << endl;
        };

        // Write a starting marker and version.
        outStream << "SDR " << SERIALIZE_VERSION << endl;

        // Store the dimensions.
        writeVector( dimensions );

        // Store the data valid flags.
        if( hasValue() )
            outStream << "valid_data" << endl;
        else
            outStream << "no_data" << endl;

        // Store the data in the flat-index format.
        if( ! hasValue() )
            writeVector( {} );
        else if( flatIndex_valid )
            writeVector( flatIndex );
        else {
            SparseDistributedRepresentation constWorkAround( *this );
            writeVector( constWorkAround.getFlatIndex() );
        }

        outStream << "~SDR" << endl;
    };

    /**
     * Load (deserialize) and initialize the SDR from the specified input
     * stream.
     *
     * @param stream A input valid istream, such as an open file.
     */
    void load(std::istream &inStream) override {

        auto readVector = [&inStream] (vector<UInt> &vec) {
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
        NTA_ASSERT( marker == "SDR" );
        NTA_ASSERT( version == SERIALIZE_VERSION );

        // Read the dimensions.
        readVector( dimensions_ );


        // Read the data valid flags.
        string valid;
        inStream >> valid;

        // Read the data.
        readVector( flatIndex );

        if( valid == "valid_data" ) {
            flatIndex_valid_ = true;
        }
        else if( valid == "no_data" ) {
            clear();
        }

        // Consume the end marker.
        inStream >> marker;
        NTA_ASSERT( marker == "~SDR" );

        // Initialize the SDR.
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
        dense = vector<Byte>(size);
        // Initialize the flatIndex array, nothing to do.
        // Initialize the index tuple.
        index.assign( dimensions.size(), {} );
    };
};

typedef SparseDistributedRepresentation SDR;

}; // end namespace nupic
#endif // end ifndef SDR_HPP
