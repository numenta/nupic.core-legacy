/* ---------------------------------------------------------------------
 * Copyright (C) 2018-2019, David McDougall.
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

#include "nupic/types/Sdr.hpp"

#include <numeric>
#include <algorithm> // std::sort

using namespace std;

namespace nupic {
// SDR implementation in .cpp, refer to .hpp for detailed comments

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
    
    /**
     * Remove the value from this SDR by clearing all of the valid flags.  Does
     * not actually change any of the data.  Attempting to get the SDR's value
     * immediately after this operation will raise an exception.
     */
    void SparseDistributedRepresentation::clear() {
        dense_valid      = false;
        flatSparse_valid = false;
        sparse_valid     = false;
    }

    /**
     * Notify everyone that this SDR's value has officially changed.
     */
    void SparseDistributedRepresentation::do_callbacks() {
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
    void SparseDistributedRepresentation::setDenseInplace() {
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
    void SparseDistributedRepresentation::setFlatSparseInplace() {
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
    void SparseDistributedRepresentation::setSparseInplace() {
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
    void SparseDistributedRepresentation::deconstruct() {
        clear();
        size_ = 0;
        dimensions_.clear();
        for( auto func : destroyCallbacks ) {
            if( func != nullptr )
                func();
        }
    }

    /**
     * Use this method only in conjuction with sdr.initialize() or sdr.load().
     */
    SparseDistributedRepresentation::SparseDistributedRepresentation() {}

    /**
     * Create an SDR object.  The initial value is all zeros.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.  The product of the dimensions must be greater than zero.
     */
    SparseDistributedRepresentation::SparseDistributedRepresentation( const vector<UInt> dimensions )
        { initialize( dimensions ); }

    void SparseDistributedRepresentation::initialize( const vector<UInt> dimensions ) {
        dimensions_ = dimensions;
        NTA_CHECK( dimensions.size() > 0 ) << "SDR has no dimensions!";
        // Calculate the SDR's size.
        size_ = 1;
        for(UInt dim : dimensions)
            size_ *= dim;
	NTA_CHECK(size_ > 0) << "SDR: all dimensions must be > 0";

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
    SparseDistributedRepresentation::SparseDistributedRepresentation( const SparseDistributedRepresentation &value )
        : SparseDistributedRepresentation( value.dimensions )
        { setSDR( value ); }

    SparseDistributedRepresentation::~SparseDistributedRepresentation()
        { deconstruct(); }


    /**
     * Set all of the values in the SDR to false.  This method overwrites the
     * SDRs current value.
     */
    void SparseDistributedRepresentation::zero() {
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
    void SparseDistributedRepresentation::setDense( SDR_dense_t &value ) {
        NTA_ASSERT(value.size() == size);
        dense.swap( value );
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
    SDR_dense_t& SparseDistributedRepresentation::getDense() const {
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
    Byte SparseDistributedRepresentation::at(const vector<UInt> &coordinates) const {
        UInt flat = 0;
	NTA_ASSERT(coordinates.size() == dimensions.size()) << "SDR: coordinates must have same dimensions as SDR";
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
    void SparseDistributedRepresentation::setFlatSparse( SDR_flatSparse_t &value ) {
        flatSparse.swap( value );
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
    SDR_flatSparse_t& SparseDistributedRepresentation::getFlatSparse() const {
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
            }
            else if( dense_valid ) {
                // Convert from dense to flatSparse.
                for(UInt idx = 0; idx < size; idx++)
                    if( dense[idx] != 0 )
                        flatSparse.push_back( idx );
            }
            else
                NTA_THROW << "SDR has no data!";
            flatSparse_valid = true;
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
    void SparseDistributedRepresentation::setSparse( SDR_sparse_t &value ) {
        sparse.swap( value );
        setSparseInplace();
    }


    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.
     *
     * @returns A reference to a list of lists which contain the coordinates of
     * the true values in the SDR.     
     */
    SDR_sparse_t& SparseDistributedRepresentation::getSparse() const {
      if( !sparse_valid ) {
        // Clear out any old data.
        for( auto& vec : sparse ) {
          vec.clear();
        }
        // Convert from flatSparse to sparse.
        for( auto idx : getFlatSparse() ) {
          for(UInt dim = (UInt)(dimensions.size() - 1); dim > 0; dim--) {
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
    void SparseDistributedRepresentation::setSDR( const SparseDistributedRepresentation &value ) {
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
        // method.  Subclasses may override these getters and ignore the valid
        // flags...
        if( !dense_valid and !flatSparse_valid and !sparse_valid ) {
            const auto data = value.getFlatSparse();
            flatSparse.assign( data.begin(), data.end() );
            flatSparse_valid = true;
        }
        do_callbacks();
    }

    /**
     * Calculates the number of true bits which both SDRs have in common.
     *
     * @param sdr, An SDR to compare with, both SDRs must have the same
     * dimensons.
     *
     * @returns Integer, the number of true values which both SDRs have in
     * common.
     */
    UInt SparseDistributedRepresentation::getOverlap(const SparseDistributedRepresentation &sdr) const {
        NTA_ASSERT( dimensions == sdr.dimensions );

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
     * @param rng The random number generator to draw from.  If not given, this
     * makes one using the magic seed 0.
     */
    void SparseDistributedRepresentation::randomize(Real sparsity) {
        Random rng( 0 );
        randomize( sparsity, rng );
    }

    void SparseDistributedRepresentation::randomize(Real sparsity, Random &rng) {
        NTA_ASSERT( sparsity >= 0.0f and sparsity <= 1.0f );
        UInt nbits = (UInt) std::round( size * sparsity );

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
     * @param rng The random number generator to draw from.  If not given, this
     * makes one using the magic seed 0.
     */
    void SparseDistributedRepresentation::addNoise(Real fractionNoise) {
        Random rng( 0 );
        addNoise( fractionNoise, rng );
    }

    void SparseDistributedRepresentation::addNoise(Real fractionNoise, Random &rng) {
        NTA_ASSERT( fractionNoise >= 0. and fractionNoise <= 1. );
        NTA_CHECK( ( 1 + fractionNoise) * getSparsity() <= 1. );

        UInt num_move_bits = (UInt) std::round( fractionNoise * getSum() );
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
            off_pop.data(), (UInt)off_pop.size(),
            turn_on.data(), num_move_bits);

        for( auto idx : turn_on )
            dns[ idx ] = 1;
        for( auto idx : turn_off )
            dns[ idx ] = 0;

        setDenseInplace();
    }


    bool SparseDistributedRepresentation::operator==(const SparseDistributedRepresentation &sdr) const {
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

    /**
     * Save (serialize) the current state of the SDR to the specified file.
     * This method can NOT save callbacks!  Only the dimensions and current data
     * are saved.
     * 
     * @param stream A valid output stream, such as an open file.
     */
    void SparseDistributedRepresentation::save(std::ostream &outStream) const {

        auto writeVector = [&outStream] (const vector<UInt> &vec) {
            outStream << vec.size() << " ";
            for( auto elem : vec ) {
                outStream << elem << " ";
            }
            outStream << endl;
        };

        // Write a starting marker and version.
        outStream << "SDR " << SERIALIZE_VERSION << " " << endl;

        // Store the dimensions.
        writeVector( dimensions );

        // Store the data in the flat-sparse format.
        writeVector( getFlatSparse() );

        outStream << "~SDR" << endl;
    }

    /**
     * Load (deserialize) and initialize the SDR from the specified input
     * stream.  This method does NOT load callbacks!  If the original SDR had
     * callbacks then the user must re-add them after saving & loading the SDR.
     *
     * @param stream A input valid istream, such as an open file.
     */
    void SparseDistributedRepresentation::load(std::istream &inStream) {

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
    UInt SparseDistributedRepresentation::addCallback(SDR_callback_t callback) {
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
    void SparseDistributedRepresentation::removeCallback(UInt index) {
        NTA_CHECK( index < callbacks.size() )
            << "SparseDistributedRepresentation::removeCallback, Invalid Handle!";
        NTA_CHECK( callbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeCallback, Callback already removed!";
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
    UInt SparseDistributedRepresentation::addDestroyCallback(SDR_callback_t callback) {
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
    void SparseDistributedRepresentation::removeDestroyCallback(UInt index) {
        NTA_CHECK( index < destroyCallbacks.size() )
            << "SparseDistributedRepresentation::removeDestroyCallback, Invalid Handle!";
        NTA_CHECK( destroyCallbacks[index] != nullptr )
            << "SparseDistributedRepresentation::removeDestroyCallback, Callback already removed!";
        destroyCallbacks[index] = nullptr;
    }

} // end namespace nupic
