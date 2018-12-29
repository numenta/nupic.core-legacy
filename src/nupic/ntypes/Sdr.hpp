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
#include <numeric>
#include <nupic/types/Types.hpp>
#include <nupic/ntypes/Array.hpp>
#include <nupic/types/Serializable.hpp>
#include <nupic/utils/Random.hpp>
#include <functional>
#include <cmath> // std::log2 isnan
#include <regex>

using namespace std;

namespace nupic {

typedef vector<Byte>          SDR_dense_t;
typedef vector<UInt>          SDR_flatSparse_t;
typedef vector<vector<UInt>>  SDR_sparse_t;
typedef function<void()>      SDR_callback_t;

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
 * which are dense, sparse, and flat-sparse.  Converted values are cached by
 * this class, so getting a value in one format many times incurs no extra
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
 *    SDR.  This is a list of lists: the outter list contains an entry for
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
 *    X.setFlatSparse({ 1, 4, 8 });
 *    X.setSparse({{ 0, 1, 2,}, { 1, 1, 2 }});
 *
 *    // Access data in any format, SDR will automatically convert data formats.
 *    X.getDense()     -> { 0, 1, 0, 0, 1, 0, 0, 0, 1 }
 *    X.getSparse()     -> {{ 0, 1, 2 }, {1, 1, 2}}
 *    x.getFlatSparse() -> { 1, 4, 8 }
 *
 *    // Data format conversions are cached, and when an SDR value changes the
 *    // cache is cleared.
 *    X.setFlatSparse({});  // Assign new data to the SDR, clearing the cache.
 *    X.getDense();        // This line will convert formats.
 *    X.getDense();        // This line will resuse the result of the previous line
 *
 *
 * Avoiding Copying:  To avoid copying call the setter methods with the correct
 * data types and non-constant variables.  This allows for a fast swap instead
 * of a slow copy operation.  The data vectors returned by the getter methods
 * can be modified and reassigned to the SDR, or the caller can allocate their
 * own data vectors as one of the following types:
 *     vector<Byte>            aka SDR_dense_t
 *     vector<UInt>            aka SDR_flatSparse_t
 *     vector<vector<UInt>>    aka SDR_sparse_t
 *
 * Example Usage:
 *    X.zero();
 *    SDR_dense_t newData({ 1, 0, 0, 1, 0, 0, 1, 0, 0 });
 *    X.setDense( newData );
 *    // X now points to newData, and newData points to X's old data.
 *
 *    X.zero();
 *    // The "&" is really important!  Otherwise vector copies.
 *    auto & dense = X.getDense();
 *    dense[2] = true;
 *    // Notify the SDR of the changes, even if using the SDR's own data inplace.
 *    X.setDense( dense );
 *    X.getFlatSparse() -> { 2 }
 */
class SparseDistributedRepresentation : public Serializable
{
protected:
    vector<UInt> dimensions_;
    UInt         size_;

    SDR_dense_t      dense;
    SDR_flatSparse_t flatSparse;
    SDR_sparse_t     sparse;

    /**
     * These hooks are called every time the SDR's value changes.  These can be
     * NULL pointers!  See methods addCallback & removeCallback for API details.
     */
    vector<SDR_callback_t> callbacks;

    /**
     * These hooks are called when the SDR is destroyed.  These can be NULL
     * pointers!  See methods addDestroyCallback & removeDestroyCallback for API
     * details.
     */
    vector<SDR_callback_t> destroyCallbacks;

    /**
     * These flags remember which data formats are up-to-date and which formats
     * need to be updated.
     */
    bool dense_valid;
    bool flatSparse_valid;
    bool sparse_valid;

    /**
     * Remove the value from this SDR by clearing all of the valid flags.  Does
     * not actually change any of the data.  Attempting to get the SDR's value
     * immediately after this operation will raise an exception.
     */
    virtual void clear() {
        dense_valid      = false;
        flatSparse_valid = false;
        sparse_valid     = false;
    };

    /**
     * Notify everyone that this SDR's value has officially changed.
     */
    void do_callbacks() {
        for(const auto func_ptr : callbacks) {
            if( func_ptr != nullptr )
                func_ptr();
        }
    }

    /**
     * Update the SDR to reflect the value currently inside of the dense array.
     * Use this method after modifying the dense buffer inplace, in order to
     * propigate any changes to the sparse & flatSparse formats.
     */
    virtual void setDenseInplace() {
        // Check data is valid.
        NTA_ASSERT( dense.size() == size );
        // Set the valid flags.
        clear();
        dense_valid = true;
        do_callbacks();
    }

    /**
     * Update the SDR to reflect the value currently inside of the flatSparse
     * vector. Use this method after modifying the flatSparse vector inplace, in
     * order to propigate any changes to the dense & sparse formats.
     */
    virtual void setFlatSparseInplace() {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT(flatSparse.size() <= size);
            for(auto idx : flatSparse) {
                NTA_ASSERT(idx < size);
            }
        #endif
        // Set the valid flags.
        clear();
        flatSparse_valid = true;
        do_callbacks();
    }

    /**
     * Update the SDR to reflect the value currently inside of the sparse
     * vector. Use this method after modifying the sparse vector inplace, in
     * order to propigate any changes to the dense & flatSparse formats.
     */
    virtual void setSparseInplace() {
        // Check data is valid.
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT(sparse.size() == dimensions.size());
            for(UInt dim = 0; dim < dimensions.size(); dim++) {
                const auto coord_vec = sparse[dim];
                NTA_ASSERT(coord_vec.size() <= size);
                NTA_ASSERT(coord_vec.size() == sparse[0].size()); // All coordinate vectors have same size.
                for(auto idx : coord_vec) {
                    NTA_ASSERT(idx < dimensions[dim]);
                }
            }
        #endif
        // Set the valid flags.
        clear();
        sparse_valid = true;
        do_callbacks();
    }

    /**
     * Destroy this SDR.  Makes SDR unusable, should error or clearly fail if
     * used.  Also sends notification to all watchers via destroyCallbacks.
     * This is a separate method from ~SDR so that SDRs can be destroyed long
     * before they're deallocated; SDR Proxy does this.
     */
    virtual void deconstruct() {
        cerr << "SDR DECON" << endl;
        clear();
        size_ = 0;
        dimensions_.clear();
        for( auto func : destroyCallbacks ) {
            cerr << "SDR DECON 1" << endl;
            if( func != nullptr )
                func();
        }
        cerr << "SDR DECON DONE" << endl;
    }

public:
    /**
     * Use this method only in conjuction with sdr.load().
     */
    SparseDistributedRepresentation() {}

    /**
     * Create an SDR object.  Initially SDRs value is all zeros.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.  The product of the dimensions must be greater than zero.
     */
    SparseDistributedRepresentation( const vector<UInt> dimensions ) {
        dimensions_ = dimensions;
        NTA_CHECK( dimensions.size() > 0 ) << "SDR has no dimensions!";
        // Calculate the SDR's size.
        size_ = 1;
        for(UInt dim : dimensions)
            size_ *= dim;
        NTA_CHECK( size > 0 ) << "SDR size is zero!";

        // Initialize the dense array storage, when it's needed.
        dense_valid = false;
        // Initialize the flatSparse array, nothing to do.
        flatSparse_valid = true;
        // Initialize the index tuple.
        sparse.assign( dimensions.size(), {} );
        sparse_valid = true;
    }

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
    }

    virtual ~SparseDistributedRepresentation()
        { deconstruct(); }

    /**
     * @attribute dimensions A list of dimensions of the SDR.
     */
    const vector<UInt> &dimensions = dimensions_;

    /**
     * @attribute size The total number of boolean values in the SDR.
     */
    const UInt &size = size_;

    /**
     * Set all of the values in the SDR to false.  This method overwrites the
     * SDRs current value.
     */
    void zero() {
        flatSparse.clear();
        setFlatSparseInplace();
    }

    /**
     * Swap a new value into the SDR, replacng the current value.  This
     * method is fast since it copies no data.  This method modifies its
     * argument!
     *
     * @param value A dense vector<char> to swap into the SDR.
     */
    void setDense( SDR_dense_t &value ) {
        NTA_ASSERT(value.size() == size);
        dense.swap( value );
        setDenseInplace();
    }

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense vector to copy into the SDR.
     */
    template<typename T>
    void setDense( const vector<T> &value ) {
        NTA_ASSERT(value.size() == size);
        dense.assign( value.begin(), value.end() );
        setDenseInplace();
    }

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense C-style array to copy into the SDR.
     */
    template<typename T>
    void setDense( const T *value ) {
        NTA_ASSERT(value != nullptr);
        dense.assign( value, value + size );
        setDenseInplace();
    }

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense array to copy into the SDR.
     */
    void setDense( const ArrayBase &value )  {
        NTA_ASSERT( value.getCount() == size );
        BasicType::convertArray(
            getDense().data(), NTA_BasicType_Byte,
            value.getBuffer(), value.getType(),
            size);
        setDenseInplace();
    }

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * cached inside of this SDR until the SDRs value changes.  After modifying
     * the dense array you MUST call sdr.setDense() in order to notify the SDR
     * that its dense array has changed and its cached data is out of date.
     *
     * @returns A reference to an array of all the values in the SDR.
     */
    virtual SDR_dense_t& getDense() {
        if( !dense_valid ) {
            // Convert from flatSparse to dense.
            dense.assign( size, 0 );
            for(const auto idx : getFlatSparse()) {
                dense[idx] = 1;
            }
            dense_valid = true;
        }
        return dense;
    }

    /**
     * Query the value of the SDR at a single location.
     *
     * @param coordinates A list of coordinates into the SDR space to query.
     *
     * @returns The value of the SDR at the given location.
     */
    Byte at(const vector<UInt> &coordinates) {
        UInt flat = 0;
        for(UInt i = 0; i < dimensions.size(); i++) {
            NTA_ASSERT( coordinates[i] < dimensions[i] );
            flat *= dimensions[i];
            flat += coordinates[i];
        }
        return getDense()[flat];
    }

    /**
     * Swap a new value into the SDR, replacing the current value.  This
     * method is fast since it copies no data.  This method modifies its
     * argument!
     *
     * @param value A sparse vector<UInt> to swap into the SDR.
     */
    void setFlatSparse( SDR_flatSparse_t &value ) {
        flatSparse.swap( value );
        setFlatSparseInplace();
    }

    /**
     * Copy a vector of sparse indices of true values.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A vector of flat indices to copy into the SDR.
     */
    template<typename T>
    void setFlatSparse( const vector<T> &value ) {
        flatSparse.assign( value.begin(), value.end() );
        setFlatSparseInplace();
    }

    /**
     * Copy an array of sparse indices of true values.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A C-style array of indices to copy into the SDR.
     * @param num_values The number of elements in the 'value' array.
     */
    template<typename T>
    void setFlatSparse( const T *value, const UInt num_values ) {
        flatSparse.assign( value, value + num_values );
        setFlatSparseInplace();
    }

    /**
     * Copy an array of sparse indices of true values.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value An array of flat indices to copy into the SDR.
     */
    void setFlatSparse( const ArrayBase &value ) {
        getFlatSparse().resize( value.getCount() );
        BasicType::convertArray(
            getFlatSparse().data(),
            NTA_BasicType_UInt,
            value.getBuffer(), value.getType(),
            value.getCount());
        setFlatSparseInplace();
    }

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  After modifying
     * the flatSparse vector you MUST call sdr.setFlatSparse() in order to
     * notify the SDR that its flatSparse vector has changed and its cached data
     * is out of date.
     *
     * @returns A reference to a vector of the indices of the true values in the
     * flattened SDR.
     */
    virtual SDR_flatSparse_t& getFlatSparse() {
        if( !flatSparse_valid ) {
            flatSparse.clear(); // Clear out any old data.
            if( sparse_valid ) {
                // Convert from sparse to flatSparse.
                const auto num_nz = size ? sparse[0].size() : 0;
                flatSparse.reserve( num_nz );
                for(UInt nz = 0; nz < num_nz; nz++) {
                    UInt flat = 0;
                    for(UInt dim = 0; dim < dimensions.size(); dim++) {
                        flat *= dimensions[dim];
                        flat += sparse[dim][nz];
                    }
                    flatSparse.push_back(flat);
                }
                flatSparse_valid = true;
            }
            else if( dense_valid ) {
                // Convert from dense to flatSparse.
                for(UInt idx = 0; idx < size; idx++)
                    if( dense[idx] != 0 )
                        flatSparse.push_back( idx );
                flatSparse_valid = true;
            }
            else
                NTA_THROW << "SDR has no data!";
        }
        return flatSparse;
    }

    /**
     * Swap a list of indices into the SDR, replacing the SDRs current value.
     * These indices are into the SDR space with dimensions.  The outter list is
     * indexed using an index into the sdr.dimensions list.  The inner lists are
     * indexed in parallel, they contain the coordinates of the true values in
     * the SDR.
     *
     * This method is fast since it swaps the vector content, however it does
     * modify its argument!
     *
     * @param value A vector<vector<UInt>> containing the coordinates of the
     * true values to swap into the SDR.
     */
    void setSparse( vector<vector<UInt>> &value ) {
        sparse.swap( value );
        setSparseInplace();
    }

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
    template<typename T>
    void setSparse( const vector<vector<T>> &value ) {
        NTA_ASSERT(value.size() == dimensions.size());
        for(UInt dim = 0; dim < dimensions.size(); dim++) {
            sparse[dim].assign( value[dim].begin(), value[dim].end() );
        }
        setSparseInplace();
    }

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.
     *
     * @returns A reference to a list of lists which contain the coordinates of
     * the true values in the SDR.     
     */
    virtual SDR_sparse_t& getSparse() {
        if( !sparse_valid ) {
            // Clear out any old data.
            for( auto& vec : sparse ) {
                vec.clear();
            }
            // Convert from flatSparse to sparse.
            for( auto idx : getFlatSparse() ) {
                for(UInt dim = dimensions.size() - 1; dim > 0; dim--) {
                    auto dim_sz = dimensions[dim];
                    sparse[dim].push_back( idx % dim_sz );
                    idx /= dim_sz;
                }
                sparse[0].push_back(idx);
            }
            sparse_valid = true;
        }
        return sparse;
    }

    /**
     * Deep Copy the given SDR to this SDR.  This overwrites the current value of
     * this SDR.  This SDR and the given SDR will have no shared data and they
     * can be modified without affecting each other.
     *
     * @param value An SDR to copy the value of.
     */
    virtual void setSDR( const SparseDistributedRepresentation &value ) {
        NTA_ASSERT( value.dimensions == dimensions );
        clear();

        dense_valid = value.dense_valid;
        if( dense_valid ) {
            dense.assign( value.dense.begin(), value.dense.end() );
        }
        flatSparse_valid = value.flatSparse_valid;
        if( flatSparse_valid ) {
            flatSparse.assign( value.flatSparse.begin(), value.flatSparse.end() );
        }
        sparse_valid = value.sparse_valid;
        if( sparse_valid ) {
            for(UInt dim = 0; dim < dimensions.size(); dim++)
                sparse[dim].assign( value.sparse[dim].begin(), value.sparse[dim].end() );
        }
        do_callbacks();
    }

    /**
     * Calculates the number of true / non-zero values in the SDR.
     *
     * @returns The number of true values in the SDR.
     */
    UInt getSum()
        { return getFlatSparse().size(); }

    /**
     * Calculates the sparsity of the SDR, which is the fraction of bits which
     * are true out of the total number of bits in the SDR.
     * I.E.  sparsity = sdr.getSum() / sdr.size
     *
     * @returns The fraction of values in the SDR which are true.
     */
    Real getSparsity()
        { return (Real) getSum() / size; }

    /**
     * Calculates the number of true bits which both SDRs have in common.
     *
     * @param sdr, An SDR to compare with, both SDRs must have the same
     * dimensons.
     *
     * @returns Integer, the number of true values which both SDRs have in
     * common.
     */
    UInt overlap(SparseDistributedRepresentation &sdr) {
        #ifdef NTA_ASSERTIONS_ON
            NTA_ASSERT( dimensions.size() == sdr.dimensions.size() );
            for( UInt i = 0u; i < dimensions.size(); i++ )
                NTA_ASSERT( dimensions[i] == sdr.dimensions[i] );
        #endif

        UInt ovlp = 0u;
        const auto a = this->getDense();
        const auto b = sdr.getDense();
        for( UInt i = 0u; i < size; i++ )
            ovlp += a[i] && b[i];
        return ovlp;
    }

    /**
     * Make a random SDR, overwriting the current value of the SDR.  The
     * result has uniformly random activations.
     *
     * @param sparsity The sparsity of the randomly generated SDR.
     *
     * Both of the following parameters are optional, if neither is given then
     * the seed 0 is used.
     * @param seed The seed for the random number generator.
     * @param rng The random number generator to draw from.
     */
    void randomize(Real sparsity) {
        Random rng( 0 );
        randomize( sparsity, rng );
    }

    void randomize(Real sparsity, Random &rng) {
        NTA_ASSERT( sparsity >= 0. and sparsity <= 1. );
        UInt nbits = size * sparsity + .5;

        SDR_flatSparse_t range( size );
        iota( range.begin(), range.end(), 0 );
        flatSparse.resize( nbits );
        rng.sample( range.data(),      size,
                    flatSparse.data(), nbits);
        setFlatSparseInplace();
    }

    /**
     * Modify the SDR by moving a fraction of the active bits to different
     * locations.  This method does not change the sparsity of the SDR, it moves
     * the locations of the true values.  The resulting SDR has a controlled
     * amount of overlap with the original.
     *
     * @param fractionNoise The fraction of active bits to swap out.  The
     * original and resulting SDRs have an overlap of (1 - fractionNoise).
     *
     * Both of the following parameters are optional, if neither is given then
     * the seed 0 is used.
     * @param seed The seed for the random number generator.
     * @param rng The random number generator to draw from.
     */
    void addNoise(Real fractionNoise) {
        Random rng( 0 );
        addNoise( fractionNoise, rng );
    }

    void addNoise(Real fractionNoise, Random &rng) {
        NTA_ASSERT( fractionNoise >= 0. and fractionNoise <= 1. );
        NTA_CHECK( ( 1 + fractionNoise) * getSparsity() <= 1. );

        UInt num_move_bits = fractionNoise * getSum() + .5;
        vector<UInt> turn_off( num_move_bits , 0 );
        rng.sample(
            (UInt*) getFlatSparse().data(), getSum(),
            (UInt*) turn_off.data(),        num_move_bits);

        auto& dns = getDense();

        vector<UInt> off_pop;
        for(UInt idx = 0; idx < size; idx++) {
            if( dns[idx] == 0 )
                off_pop.push_back( idx );
        }
        vector<UInt> turn_on( num_move_bits, 0 );
        rng.sample(
            off_pop.data(), off_pop.size(),
            turn_on.data(), num_move_bits);

        for( auto idx : turn_on )
            dns[ idx ] = 1;
        for( auto idx : turn_off )
            dns[ idx ] = 0;

        setDenseInplace();
    }

    /**
     * Print a human readable version of the SDR, defaults to STDOUT.
     *
     * @param stream The output to write to, defaults to std::cout.
     */
    void print(std::ostream &stream = std::cout) {
        stream << "SDR( ";
        for( UInt i = 0; i < dimensions.size(); i++ ) {
            stream << dimensions[i];
            if( i + 1 != dimensions.size() )
                stream << ", ";
        }
        stream << " ) ";
        auto data = getFlatSparse();
        for( UInt i = 0; i < data.size(); i++ ) {
            stream << data[i];
            if( i + 1 != data.size() )
                stream << ", ";
        }
        stream << endl;
    }

    bool operator==(SparseDistributedRepresentation &sdr) {
        // Check attributes
        if( sdr.size != size or dimensions.size() != sdr.dimensions.size() )
            return false;
        for( UInt i = 0; i < dimensions.size(); i++ ) {
            if( dimensions[i] != sdr.dimensions[i] )
                return false;
        }
        // Check data
        return std::equal(
            getDense().begin(),
            getDense().end(), 
            sdr.getDense().begin());
    }

    bool operator!=(SparseDistributedRepresentation &sdr)
        { return not ((*this) == sdr); }

    /**
     * Save (serialize) the current state of the SDR to the specified file.
     * This method can NOT save callbacks!  Only the dimensions and current data
     * are saved.
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

        // Store the data in the flat-sparse format.
        if( flatSparse_valid )
            writeVector( flatSparse );
        else {
            SparseDistributedRepresentation constWorkAround( *this );
            writeVector( constWorkAround.getFlatSparse() );
        }

        outStream << "~SDR" << endl;
    }

    /**
     * Load (deserialize) and initialize the SDR from the specified input
     * stream.  This method does NOT load callbacks!  If the original SDR had
     * callbacks then the user must re-add them after saving & loading the SDR.
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
        NTA_CHECK( marker == "SDR" );
        NTA_CHECK( version == SERIALIZE_VERSION );

        // Read the dimensions.
        readVector( dimensions_ );

        // Initialize the SDR.
        // Calculate the SDR's size.
        size_ = 1;
        for(UInt dim : dimensions)
            size_ *= dim;
        // Initialize sparse tuple.
        sparse.assign( dimensions.size(), {} );

        // Read the data.
        readVector( flatSparse );
        setFlatSparseInplace();

        // Consume the end marker.
        inStream >> marker;
        NTA_CHECK( marker == "~SDR" );
    }

    /**
     * Callbacks notify you when this SDR's value changes.
     *
     * Note: callbacks are NOT serialized; after saving & loading an SDR the
     * user must setup their callbacks again.
     *
     * @param callback A function to call every time this SDRs value changes.
     * function accepts no arguments and returns void.
     *
     * @returns UInt Handle for the given callback, needed to remove callback.
     */
    UInt addCallback(SDR_callback_t callback) {
        UInt index = 0;
        for( ; index < callbacks.size(); index++ ) {
            if( callbacks[index] == nullptr ) {
                callbacks[index] = callback;
                return index;
            }
        }
        callbacks.push_back( callback );
        return index;
    }

    /**
     * Remove a previously registered callback.
     *
     * @param UInt Handle which was returned by addCallback when you registered
     * your callback.
     */
    void removeCallback(UInt index) {
        NTA_CHECK( index < callbacks.size() )
            << "SDR::removeCallback, Invalid Handle!";
        NTA_CHECK( callbacks[index] != nullptr )
            << "SDR::removeCallback, Callback already removed!";
        callbacks[index] = nullptr;
    }

    /**
     * This callback notifies you when this SDR is deconstructed and freed from
     * memory.
     *
     * Note: callbacks are NOT serialized; after saving & loading an SDR the
     * user must setup their callbacks again.
     *
     * @param callback A function to call when this SDR is destroyed.  Function
     * accepts no arguments and returns void.
     *
     * @returns UInt Handle for the given callback, needed to remove callback.
     */
    UInt addDestroyCallback(SDR_callback_t callback) {
        UInt index = 0;
        for( ; index < destroyCallbacks.size(); index++ ) {
            if( destroyCallbacks[index] == nullptr ) {
                destroyCallbacks[index] = callback;
                return index;
            }
        }
        destroyCallbacks.push_back( callback );
        return index;
    }

    /**
     * Remove a previously registered destroy callback.
     *
     * @param UInt Handle which was returned by addDestroyCallback when you
     * registered your callback.
     */
    void removeDestroyCallback(UInt index) {
        NTA_CHECK( index < destroyCallbacks.size() )
            << "SDR::removeDestroyCallback, Invalid Handle!";
        NTA_CHECK( destroyCallbacks[index] != nullptr )
            << "SDR::removeDestroyCallback, Callback already removed!";
        destroyCallbacks[index] = nullptr;
    }
};

typedef SparseDistributedRepresentation SDR;

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
 * Save/Load: SDR_Proxy does not support the Serializable interface.  Users
 * should instead serialize the source SDR and recreate the proxy.
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

    SDR_dense_t& getDense() override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        return parent->getDense();
    }

    SDR_flatSparse_t& getFlatSparse() override {
        NTA_CHECK( parent != nullptr ) << "Parent SDR has been destroyed!";
        return parent->getFlatSparse();
    }

    SDR_sparse_t& getSparse() override {
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

    const string _SDR_save_load_error_message =
        "SDR_Proxy does not support serialization, save/load the source SDR instead.";
    void save(std::ostream &outStream) const override
        { NTA_THROW << _SDR_save_load_error_message; }
    void load(std::istream &inStream) override
        { NTA_THROW << _SDR_save_load_error_message; }
};

/**
 * Helper for SDR metrics trackers, including: SDR_Sparsity,
 * SDR_ActivationFrequency, and SDR_Overlap classes.
 *
 * Subclasses must override method "callback".
 */
class _SDR_MetricsHelper {
protected:
    SDR* dataSource_;
    UInt period_;
    int  samples_;
    UInt  callback_handle_;
    UInt  destroyCallback_handle_;

    /**
     * @param dataSource SDR to track.  Add data to the metric by assigning to
     * this SDR.  This class deals with adding a callback to this SDR so that
     * your SDR-MetricsTracker is notified after every update to the SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    _SDR_MetricsHelper( SDR &dataSource, UInt period ) {
        cerr << "METRIC CONSTRUCT" << endl;
        NTA_CHECK( period > 0u );
        dataSource_ = &dataSource;
        period_     = period;
        samples_    = 0;
        cerr << "METRIC CONSTRUCT 1" << endl;
        callback_handle_ = dataSource_->addCallback( [&](){
            // cerr << "METRIC CALLBACK" << endl;
            samples_++;
            callback( *dataSource_, 1.0f / std::min( period_, (UInt) samples_ ));
            // cerr << "METRIC CALLBACK DONE" << endl;
        });
        cerr << "METRIC CONSTRUCT 2" << endl;
        destroyCallback_handle_ = dataSource_->addDestroyCallback( [&](){
            cerr << "METRIC DESTROY CALLBACK" << endl;
            deconstruct();
        });
        cerr << "METRIC CONSTRUCT DONE" << endl;
    }

    ~_SDR_MetricsHelper() {
        cerr << "METRIC DESTROY" << endl;
        deconstruct();
    }

    void deconstruct() {
        cerr << "METRIC DECON" << endl;
        if( dataSource_ != nullptr ) {
            cerr << "METRIC DECON 1" << endl;
            dataSource_->removeCallback( callback_handle_ );
            cerr << "METRIC DECON 2" << endl;
            dataSource_->removeDestroyCallback( destroyCallback_handle_ );
            cerr << "METRIC DECON 3" << endl;
            dataSource_ = nullptr;
        }
        cerr << "METRIC DECON DONE" << endl;
    }

    /**
     * Add another datum to the metric.
     *
     * @param dataSource SDR to add to the metric.
     *
     * @param alpha Weight for the new datum.  Metrics trackers should use an
     * exponential weighting scheme, so that they can efficiently process
     * streaming data.  This class deals with finding a suitable weight for each
     * sample.
     */
    virtual void callback( SDR &dataSource, Real alpha ) = 0;

public:
    const int  &samples = samples_;
    const UInt &period  = period_;
};

/**
 * SDR_Sparsity class
 *
 * ### Description
 * Measures the sparsity of an SDR.  This accumulates measurements using an
 * exponential moving average, and outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      SDR_Sparsity B( A, 1000 )
 *      A.randomize( 0.01 )
 *      A.randomize( 0.15 )
 *      A.randomize( 0.05 )
 *      B.sparsity ->  0.05
 *      B.min()    ->  0.01
 *      B.max()    ->  0.15
 *      B.mean()   -> ~0.07
 *      B.std()    -> ~0.06
 */
class SDR_Sparsity : public _SDR_MetricsHelper {
private:
    Real min_;
    Real max_;
    Real mean_;
    Real var_;
    Real sparsity_;

    void callback(SDR &dataSource, Real alpha) override {
        sparsity_ = dataSource.getSparsity();
        min_ = std::min( min_, sparsity_ );
        max_ = std::max( max_, sparsity_ );
        // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        // See section 9.
        const Real diff   = sparsity_ - mean_;
        const Real incr   = alpha * diff;
                   mean_ += incr;
                   var_   = (1.0f - alpha) * (var_ + diff * incr);
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this sparsity metric by
     * assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Sparsity( SDR &dataSource, UInt period )
        : _SDR_MetricsHelper( dataSource, period )
    {
        sparsity_   =  1234.56789f;
        min_        =  1234.56789f;
        max_        = -1234.56789f;
        mean_       =  1234.56789f;
        var_        =  1234.56789f;
    }

    const Real &sparsity = sparsity_;
    Real min() const { return min_; }
    Real max() const { return max_; }
    Real mean() const { return mean_; }
    Real std() const { return std::sqrt( var_ ); }

    void print(std::ostream &stream = std::cout) const
    {
        stream << "Sparsity Min/Mean/Std/Max "
            << min() << " / " << mean() << " / "
            << std() << " / " << max() << endl;
    }
};

/**
 * SDR_ActivationFrequency class
 *
 * ### Description
 * Measures the activation frequency of each value in an SDR.  This accumulates
 * measurements using an exponential moving average, and outputs a summary of
 * results.
 *
 * Activation frequencies are Real numbers in the range [0, 1], where zero
 * indicates never active, and one indicates always active.
 *
 * Example Usage:
 *      SDR A( 2 )
 *      SDR_ActivationFrequency B( A, 1000 )
 *      A.setDense({ 0, 0 })
 *      A.setDense({ 1, 1 })
 *      A.setDense({ 0, 1 })
 *      B.activationFrequency -> { 0.33, 0.66 }
 *      B.min()     -> ~0.33
 *      B.max()     -> ~0.66
 *      B.mean()    ->  0.50
 *      B.std()     -> ~0.16
 *      B.entropy() -> ~0.92
 */
class SDR_ActivationFrequency : public _SDR_MetricsHelper {
private:
    vector<Real> activationFrequency_;

    void callback(SDR &dataSource, Real alpha) override
    {
        const auto decay = 1.0f - alpha;
        for(auto &value : activationFrequency_)
            value *= decay;

        const auto &sparse = dataSource.getFlatSparse();
        for(const auto &idx : sparse)
            activationFrequency_[idx] += alpha;
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_ActivationFrequency
     * instance by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_ActivationFrequency( SDR &dataSource, UInt period )
        : _SDR_MetricsHelper( dataSource, period )
    {
        activationFrequency_.assign( dataSource.size, 1234.56789f );
    }

    const vector<Real> &activationFrequency = activationFrequency_;

    Real min() const {
        return *std::min_element(activationFrequency_.begin(),
                                 activationFrequency_.end());
    }

    Real max() const {
        return *std::max_element(activationFrequency_.begin(),
                                 activationFrequency_.end());
    }

    Real mean() const  {
        const auto sum = std::accumulate( activationFrequency_.begin(),
                                          activationFrequency_.end(),
                                          0.0f);
        return (Real) sum / activationFrequency_.size();
    }

    Real std() const {
        const auto mean_ = mean();
        auto sum_squares = 0.0f;
        for(auto &frequency : activationFrequency) {
            const auto displacement = frequency - mean_;
            sum_squares += displacement * displacement;
        }
        const auto variance = sum_squares / activationFrequency.size();

        return std::sqrt( variance );
    }

    static Real binary_entropy_(const vector<Real> &frequencies) {
        Real accumulator = 0.0f;
        for(const auto &p  : frequencies) {
            const auto  p_ = 1.0f - p;
            const auto  e  = -p * std::log2( p ) - p_ * std::log2( p_ );
            accumulator   += isnan(e) ? 0.0f : e;
        }
        return accumulator / frequencies.size();
    }

    /**
     * Binary entropy is a measurement of information.  It measures how well the
     * SDR utilizes its resources (bits).  A low entropy indicates that many
     * bits in the SDR are under-utilized and do not transmit as much
     * information as they could.  A high entropy indicates that the SDR
     * optimally utilizes its resources.  The most optimal use of SDR resources
     * is when all bits have an equal activation frequency.  For convenience,
     * the entropy is scaled by the theoretical maximum into the range [0, 1].
     *
     * @returns Binary entropy of SDR, scaled to range [0, 1].
     */
    Real entropy() const {
        const auto max_extropy = binary_entropy_({ mean() });
        if( max_extropy == 0.0f )
            return 0.0f;
        return binary_entropy_( activationFrequency ) / max_extropy;
    }

    void print(std::ostream &stream = std::cout) const
    {
        stream << "Activation Frequency Min/Mean/Std/Max "
            << min() << " / " << mean() << " / "
            << std() << " / " << max() << endl;
        stream << "Entropy " << entropy() << endl;
    }
};


/**
 * SDR_Overlap class
 *
 * ### Description
 * Measures the overlap between successive assignments to an SDR.  This class
 * accumulates measurements using an exponential moving average, and outputs a
 * summary of results.
 *
 * This class normalizes the overlap into the range [0, 1] by dividing by the
 * number of active values.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      SDR_Overlap B( A, 1000 )
 *      A.randomize( 0.05 )
 *      A.addNoise( 0.95 )  ->  5% overlap
 *      A.addNoise( 0.55 )  -> 45% overlap
 *      A.addNoise( 0.72 )  -> 28% overlap
 *      B.overlap   ->  0.28
 *      B.min()     ->  0.05
 *      B.max()     ->  0.45
 *      B.mean()    ->  0.26
 *      B.std()     -> ~0.16
 */
class SDR_Overlap : public _SDR_MetricsHelper {
private:
    SDR  previous_;
    Real overlap_;
    Real min_;
    Real max_;
    Real mean_;
    Real var_; // TODO: Rename to variance_, in all classes!

    void callback(SDR &dataSource, Real alpha) override {
        const auto nbits = std::max( previous_.getSum(), dataSource.getSum() );
        const auto overlap = (nbits == 0u) ? 0.0f
                               : (Real) previous_.overlap( dataSource ) / nbits;
        previous_.setSDR( dataSource );
        // Ignore first data point, need two to compute.  Account for the
        // initial decrement to samples counter.
        if( samples + 1 < 2 ) return;
        overlap_ = overlap; // Don't overwrite initial value until have valid data.
        min_     = std::min( min_, overlap );
        max_     = std::max( max_, overlap );
        // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        // See section 9.
        const Real diff   = overlap - mean_;
        const Real incr   = alpha * diff;
                   mean_ += incr;
                   var_   = (1.0f - alpha) * (var_ + diff * incr);
    }

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_Overlap instance
     * by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Overlap( SDR &dataSource, UInt period )
        : _SDR_MetricsHelper( dataSource, period ),
          previous_( dataSource.dimensions )
    {
        // This class needs two samples before its data is valid, instead of one
        // sample like _SDR_MetricsHelper class  expects, so start the samples
        // counter one behind.
        samples_   -=  1;
        overlap_    =  1234.56789f;
        min_        =  1234.56789f;
        max_        = -1234.56789f;
        mean_       =  1234.56789f;
        var_        =  1234.56789f;
    }

    const Real &overlap = overlap_; // TODO: Should this be a method, for consistency with min/mean/max?
    Real min() const { return min_; }
    Real max() const { return max_; }
    Real mean() const { return mean_; }
    Real std() const { return std::sqrt( var_ ); }

    void print(std::ostream &stream = std::cout) const
    {
        stream << "Overlap Min/Mean/Std/Max "
            << min() << " / " << mean() << " / "
            << std() << " / " << max() << endl;
    }
};

/**
 * SDR_Metrics class
 *
 * ### Description
 * Measures an SDR.  This applies the following three metrics:
 *      SDR_Sparsity
 *      SDR_ActivationFrequency
 *      SDR_Overlap
 *
 * This accumulates measurements using an exponential moving average, and
 * outputs a summary of results.
 *
 * Example Usage:
 *      SDR A( dimensions )
 *      SDR_Metrics M( A, 1000 )
 *
 *      Run program:
 *          A.setData( ... )
 *
 *      M.print()
 */
class SDR_Metrics {
private:
    vector<UInt>            dimensions_;
    SDR_Sparsity            sparsity_;
    SDR_ActivationFrequency activationFrequency_;
    SDR_Overlap             overlap_;

public:
    /**
     * @param dataSource SDR to track.  Add data to this SDR_Metrics instance
     * by assigning to this SDR.
     *
     * @param period Time scale for exponential moving average.
     */
    SDR_Metrics( SDR &dataSource, UInt period )
        : sparsity_( dataSource, period ),
          activationFrequency_( dataSource, period ),
          overlap_( dataSource, period )
    {
        // TODO: Add flags to enable/disable which metrics are turned on by
        // default.
        dimensions_ = dataSource.dimensions;
    }

    const SDR_Sparsity            &sparsity            = sparsity_;
    const SDR_ActivationFrequency &activationFrequency = activationFrequency_;
    const SDR_Overlap             &overlap             = overlap_;

    void print(std::ostream &stream = std::cout) const {
        // Introduction line:  "SDR ( dimensions )"
        stream << "SDR( ";
        for(const auto &dim : dimensions_)
            stream << dim << " ";
        stream << ")" << endl;

        // Print data to temporary area for formatting.
        stringstream data_stream;

        sparsity.print( data_stream );
        activationFrequency.print( data_stream );
        overlap.print( data_stream );

        // Indent all of the data.
        string data = data_stream.str();
        // Append tabs to all newlines
        data = regex_replace( data, regex("\n"), "\n\r    " );
        // Strip trailing whitespace
        data = regex_replace( data, regex("\\s*$"), "" );
        stream << "    ";
        stream << data << endl;
    }
};

} // end namespace nupic
#endif // end ifndef SDR_HPP
