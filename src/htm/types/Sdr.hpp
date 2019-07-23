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
 * ---------------------------------------------------------------------- */

/** @file
 * Definitions for SparseDistributedRepresentation class
 */

#ifndef SDR_HPP
#define SDR_HPP

#define SERIALIZE_VERSION 1

#include <algorithm> //sort
#include <functional>
#include <vector>

#include <htm/types/Types.hpp>
#include <htm/types/Serializable.hpp>
#include <htm/utils/Random.hpp>

namespace htm {

using ElemDense        = Byte; //TODO allow changing this
using ElemSparse       = UInt32; //must match with connections::CellIdx 

using SDR_dense_t      = std::vector<ElemDense>;
using SDR_sparse_t     = std::vector<ElemSparse>;
using SDR_coordinate_t = std::vector<std::vector<UInt>>;
using SDR_callback_t   = std::function<void()>;

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
 * which are dense, sparse, and coordinates.  Converted values are cached by
 * this class, so getting a value in one format many times incurs no extra
 * performance cost.  Assigning to the SDR via a setter method will clear these
 * cached values and cause them to be recomputed as needed.
 *
 *    Dense Format: A contiguous array of boolean values, representing all of
 *    the bits in the SDR.  This format allows random-access queries of the SDRs
 *    values.
 *
 *    Sparse Index Format: Contains the indices of only the true values in
 *    the SDR.  This is a list of the indices, indexed into the flattened SDR.
 *    This format allows for quickly accessing all of the true bits in the SDR.
 *
 *    Coordinate Format: Contains the indices of only the true values in the
 *    SDR.  This is a list of lists: the outter list contains an entry for each
 *    dimension in the SDR. The inner lists contain the coordinates of each true
 *    bit in that dimension.  The inner lists run in parallel. This format is
 *    useful because it contains the location of each true bit inside of the
 *    SDR's dimensional space.
 *
 * Array Memory Layout: This class uses C-order throughout, meaning that when
 * iterating through the SDR, the last/right-most index changes fastest.
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
 *    X.setSparse({ 1, 4, 8 });
 *    X.setCoordinates({{ 0, 1, 2,}, { 1, 1, 2 }});
 *
 *    // Access data in any format, SDR will automatically convert data formats.
 *    X.getDense()       -> { 0, 1, 0, 0, 1, 0, 0, 0, 1 }
 *    X.getCoordinates() -> {{ 0, 1, 2 }, {1, 1, 2}}
 *    x.getSparse()      -> { 1, 4, 8 }
 *
 *    // Data format conversions are cached, and when an SDR value changes the
 *    // cache is cleared.
 *    X.setSparse({});  // Assign new data to the SDR, clearing the cache.
 *    X.getDense();     // This line will convert formats.
 *    X.getDense();     // This line will resuse the result of the previous line
 *
 *
 * Avoiding Copying:  To avoid copying call the setter methods with the correct
 * data types and non-constant variables.  This allows for a fast swap instead
 * of a slow copy operation.  The data vectors returned by the getter methods
 * can be modified and reassigned to the SDR, or the caller can allocate their
 * own data vectors as one of the following types:
 *     vector<Byte>            aka SDR_dense_t
 *     vector<UInt>            aka SDR_sparse_t
 *     vector<vector<UInt>>    aka SDR_coordinate_t
 *
 * Example Usage With Out Copying:
 *    SDR  X( {3, 3} );
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
 *    X.getSparse() -> { 2 }
 */
class SparseDistributedRepresentation : public Serializable
{
private:
    mutable std::vector<UInt> dimensions_;
    UInt                      size_;

protected:
    /**
     * Internal representations in each data format.  Not all must match at a
     * time, see *_valid below.
     */
    mutable SDR_dense_t      dense_;
    mutable SDR_sparse_t     sparse_;
    mutable SDR_coordinate_t coordinates_;

    /**
     * These flags remember which data formats are up-to-date and which formats
     * need to be updated.
     */
    mutable bool dense_valid;
    mutable bool sparse_valid;
    mutable bool coordinates_valid;

private:
    /**
     * These hooks are called every time the SDR's value changes.  These can be
     * NULL pointers!  See methods addCallback & removeCallback for API details.
     */
    mutable std::vector<SDR_callback_t> callbacks;

    /**
     * These hooks are called when the SDR is destroyed.  These can be NULL
     * pointers!  See methods addDestroyCallback & removeDestroyCallback for API
     * details.
     */
    mutable std::vector<SDR_callback_t> destroyCallbacks;

protected:
    /**
     * Remove the value from this SDR by clearing all of the valid flags.  Does
     * not actually change any of the data.  Attempting to get the SDR's value
     * immediately after this operation will raise an exception.
     */
    virtual void clear() const;

    /**
     * Notify everyone that this SDR's value has officially changed.
     */
    void do_callbacks() const;

    /**
     * Update the SDR to reflect the value currently inside of the dense array.
     * Use this method after modifying the dense buffer inplace, in order to
     * propigate any changes to the sparse & coordinate formats.
     */
    virtual void setDenseInplace() const;

    /**
     * Update the SDR to reflect the value currently inside of the flatSparse
     * vector. Use this method after modifying the flatSparse vector inplace, in
     * order to propigate any changes to the dense & coordinate formats.
     */
    virtual void setSparseInplace() const;

    /**
     * Update the SDR to reflect the value currently inside of the sparse
     * vector. Use this method after modifying the sparse vector inplace, in
     * order to propigate any changes to the dense & sparse formats.
     */
    virtual void setCoordinatesInplace() const;

    /**
     * Destroy this SDR.  Makes SDR unusable, should error or clearly fail if
     * used.  Also sends notification to all watchers via destroyCallbacks.
     * This is a separate method from ~SDR so that SDRs can be destroyed long
     * before they're deallocated.
     */
    virtual void deconstruct();

public:
    /**
     * Use this method only in conjuction with sdr.initialize() or sdr.load().
     */
    SparseDistributedRepresentation();

    /**
     * Create an SDR object.  The initial value is all zeros.
     *
     * @param dimensions A list of dimension sizes, defining the shape of the
     * SDR.  The product of the dimensions must be greater than zero.
     */
    SparseDistributedRepresentation( const std::vector<UInt> &dimensions );

    void initialize( const std::vector<UInt> &dimensions );

    /**
     * Initialize this SDR as a deep copy of the given SDR.  This SDR and the
     * given SDR will have no shared data and they can be modified without
     * affecting each other.
     *
     * @param value An SDR to replicate.
     */
    SparseDistributedRepresentation( const SparseDistributedRepresentation &value );

    virtual ~SparseDistributedRepresentation();

    /**
     * @attribute dimensions A list of dimensions of the SDR.
     */
    const std::vector<UInt> &dimensions = dimensions_;

    /**
     * @attribute size The total number of boolean values in the SDR.
     */
    const UInt &size = size_;

    /**
     * Change the dimensions of the SDR.  The total size must not change.
     */
    void reshape(const std::vector<UInt> &dimensions) const;

    /**
     * Set all of the values in the SDR to false.  This method overwrites the
     * SDRs current value.
     */
    void zero();

    /**
     * Swap a new value into the SDR, replacng the current value.  This
     * method is fast since it copies no data.  This method modifies its
     * argument!
     *
     * @param value A dense vector<char> to swap into the SDR.
     */
    void setDense( SDR_dense_t &value );

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense vector to copy into the SDR.
     */
     template<typename T>
     void setDense( const std::vector<T> &value ) {
       NTA_ASSERT(value.size() == size);
       setDense(value.data());
     }

    /**
     * Copy a new value into the SDR, overwritting the current value.
     *
     * @param value A dense C-style array to copy into the SDR.
     */
     template<typename T>
     void setDense( const T *value ) {
       NTA_ASSERT(value != nullptr);
       dense_.resize( size );
       const T zero = (T) 0;
       for(auto i = 0u; i < size; i++)
         dense_[i] = value[i] != zero;
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
    virtual SDR_dense_t& getDense() const;

    /**
     * Query the value of the SDR at a single location.
     *
     * @param coordinates A list of coordinates into the SDR space to query.
     *
     * @returns The value of the SDR at the given location.
     */
    Byte at(const std::vector<UInt> &coordinates) const;

    /**
     * Swap a new value into the SDR, replacing the current value.  This
     * method is fast since it copies no data.  This method modifies its
     * argument!
     *
     * @param value A sparse vector<UInt> to swap into the SDR.
     * @throws Sparse data must be sorted and contain no duplicates.
     */
    void setSparse( SDR_sparse_t &value );

    /**
     * Copy a vector of sparse indices of true values.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A vector of flat indices to copy into the SDR.
     * @throws Sparse data must be sorted and contain no duplicates.
     */
    template<typename T>
    void setSparse( const std::vector<T> &value ) {
      sparse_.assign( value.begin(), value.end() );
      setSparseInplace();
    }

    /**
     * Copy an array of sparse indices of true values.  These indicies are into
     * the flattened SDR space.  This overwrites the SDR's current value.
     *
     * @param value A C-style array of indices to copy into the SDR.
     * @throws Sparse data must be sorted and contain no duplicates.
     *
     * @param num_values The number of elements in the 'value' array.
     */
    template<typename T>
    void setSparse( const T *value, const UInt num_values ) {
      sparse_.assign( value, value + num_values );
      setSparseInplace();
    }

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.  After modifying
     * the sparse vector you MUST call sdr.setSparse() in order to notify the
     * SDR that its flatSparse vector has changed and its cached data is out of
     * date.
     *
     * @returns A reference to a vector of the indices of the true values in the
     * flattened SDR.
     */
    virtual SDR_sparse_t& getSparse() const;

    /**
     * Swap a list of coordinates into the SDR, replacing the SDRs current
     * value.  These are indices into the SDR space with dimensions.  This
     * accepts a list of lists, containing the coordinates of the true values in
     * the SDR.
     *
     * This method is fast since it swaps the vector content, however it does
     * modify its argument!
     *
     * @param value A vector<vector<UInt>> containing the coordinates of the true
     * values to swap into the SDR.  The outter list is indexed using an index
     * into the sdr.dimensions list.  The inner lists are indexed in parallel.
     *
     * @throws Coordinate data must be sorted and contain no duplicates.
     */
    void setCoordinates( SDR_coordinate_t &value );

    /**
     * Copy a list of coordinates into the SDR, overwritting the SDRs current
     * value.  These are indices into the SDR space with dimensions.  This
     * accepts a list of lists, containing the coordinates of the true values in
     * the SDR.
     *
     * @param value A list of lists containing the coordinates of the true
     * values to copy into the SDR.  The outter list is indexed using an index
     * into the sdr.dimensions list.  The inner lists are indexed in parallel.
     *
     * @throws Coordinate data must be sorted and contain no duplicates.
     */
    template<typename T>
    void setCoordinates( const std::vector<std::vector<T>> &value ) {
      NTA_ASSERT(value.size() == dimensions.size());
      for(UInt dim = 0; dim < dimensions.size(); dim++) {
        coordinates_[dim].clear();
		    coordinates_[dim].resize(value[dim].size());
        // Use an explicit type cast.  Otherwise Microsoft Visual Studio will
        // print an excessive number of warnings.  Do NOT replace this with:
        // coordinates_[dim].assign(value[dim].cbegin(), value[dim].cend());
        for (UInt i = 0; i < value[dim].size(); i++)
          coordinates_[dim][i] = (UInt)value[dim][i];
      }
      setCoordinatesInplace();
    }

    /**
     * Gets the current value of the SDR.  The result of this method call is
     * saved inside of this SDR until the SDRs value changes.
     *
     * @returns A reference to a list of lists which contain the coordinates of
     * the true values in the SDR.
     */
    virtual SDR_coordinate_t& getCoordinates() const;

    /**
     * Deep Copy the given SDR to this SDR.  This overwrites the current value of
     * this SDR.  This SDR and the given SDR will have no shared data and they
     * can be modified without affecting each other.
     *
     * @param value An SDR to copy the value of.
     */
    virtual void setSDR( const SparseDistributedRepresentation &value );

    SparseDistributedRepresentation& operator=(const SparseDistributedRepresentation& value);

    /**
     * Calculates the number of true / non-zero values in the SDR.
     *
     * @returns The number of true values in the SDR.
     */
    inline UInt getSum() const
        { return (UInt)getSparse().size(); }

    /**
     * Calculates the sparsity of the SDR, which is the fraction of bits which
     * are true out of the total number of bits in the SDR.
     * I.E.  sparsity = sdr.getSum() / sdr.size
     *
     * @returns The fraction of values in the SDR which are true.
     */
    inline Real getSparsity() const
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
    UInt getOverlap(const SparseDistributedRepresentation &sdr) const;

    /**
     * Make a random SDR, overwriting the current value of the SDR.  The
     * result has uniformly random activations.
     *
     * @param sparsity The sparsity of the randomly generated SDR.
     *
     * @param rng The random number generator to draw from.  If not given, this
     * makes one using the magic seed 0.
     */
    void randomize(Real sparsity);

    void randomize(Real sparsity, Random &rng);

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
    void addNoise(Real fractionNoise);

    void addNoise(Real fractionNoise, Random &rng);

    /**
     * Modify the SDR by setting a fraction of the bits to zero.
     *
     * @param fraction The fraction of bits to set to zero.  Must be between 0
     * and 1 (inclusive).
     *
     * @param seed The seed for the random number generator to draw from.  If not
     * given, this uses the magic seed 0.  Use the same seed to consistently
     * kill the same cells.
     */
    void killCells(const Real fraction, const UInt seed=0u);

    /**
     * This method calculates the set intersection of the active bits in each
     * input SDR.
     *
     * @params This method has two overloads:
     *          1) Accepts two SDRs, for convenience.
     *          2) Accepts a list of SDRs, must contain at least two SDRs, can
     *             contain as many SDRs as needed.
     *
     * @returns In both cases the output is stored in this SDR.  This method
     * modifies this SDR and discards its current value!
     *
     * Example Usage:
     *     SDR A({ 10 });
     *     SDR B({ 10 });
     *     SDR C({ 10 });
     *     A.setSparse({0, 1, 2, 3});
     *     B.setSparse(      {2, 3, 4, 5});
     *     C.intersection(A, B);
     *     C.getSparse() -> {2, 3}
     */
    void intersection(const SparseDistributedRepresentation &input1,
                      const SparseDistributedRepresentation &input2);

    void intersection(std::vector<const SparseDistributedRepresentation*> inputs);

    /**
     * This method calculates the set union of the active bits in all input SDRs.
     *
     * @params This method has two overloads:
     *          1) Accepts two SDRs, for convenience.
     *          2) Accepts a list of SDRs, must contain at least two SDRs, can
     *             contain as many SDRs as needed.
     *
     * @returns In both cases the output is stored in this SDR.  This method
     * modifies this SDR and discards its current value!
     *
     * Example Usage:
     *     SDR A({ 10 });
     *     SDR B({ 10 });
     *     SDR C({ 10 });
     *     A.setSparse({0, 1, 2, 3});
     *     B.setSparse(      {2, 3, 4, 5});
     *     C.set_union(A, B);
     *     C.getSparse() -> {0, 1, 2, 3, 4, 5}
     */
    void set_union(const SparseDistributedRepresentation &input1,
                   const SparseDistributedRepresentation &input2);

    void set_union(std::vector<const SparseDistributedRepresentation*> inputs);

    /**
     * Concatenates SDRs and stores the result in this SDR.
     *
     * @params This method has two overloads:
     *          1) Accepts two SDRs, for convenience.
     *          2) Accepts a list of SDR*, must contain at least two SDRs, can
     *             contain as many SDRs as needed.
     *
     * @param UInt axis: This can concatenate along any axis, as long as the
     * result has the same dimensions as this SDR.  The default axis is 0.
     *
     * @returns In both overloads the output is stored in this SDR.  This method
     * modifies this SDR and discards its current value!
     *
     * Example Usage:
     *      SDR A({ 10 });
     *      SDR B({ 10 });
     *      SDR C({ 20 });
     *      A.setSparse({ 0, 1, 2 });
     *      B.setSparse({ 0, 1, 2 });
     *      C.concatenate( A, B );
     *      C.getSparse() -> {0, 1, 2, 10, 11, 12}
     */
    inline void concatenate(const SparseDistributedRepresentation &inp1,
                            const SparseDistributedRepresentation &inp2,
                            const UInt  axis = 0u) {
      this->concatenate({&inp1, &inp2}, axis);
    }

    void concatenate(const std::vector<const SparseDistributedRepresentation*>& inputs,
                     const UInt axis = 0u);

    /**
     * Print a human readable version of the SDR.
     * Sample output:  
     *   "SDR( 200 ) 190, 172, 23, 118, 178, 129, 113, 71, 185, 182\n"
     */
    friend std::ostream& operator<< (std::ostream& stream, const SparseDistributedRepresentation &sdr)
    {
        stream << "SDR( ";
        for( UInt i = 0; i < (UInt)sdr.dimensions.size(); i++ ) {
            stream << sdr.dimensions[i];
            if( i + 1 != (UInt)sdr.dimensions.size() )
                stream << ", ";
        }
        stream << " ) ";
        auto data = sdr.getSparse();
        for( UInt i = 0; i < data.size(); i++ ) {
            stream << data[i];
            if( i + 1 != data.size() )
                stream << ", ";
        }
        return stream << std::endl;
    }

    bool operator==(const SparseDistributedRepresentation &sdr) const;
    inline bool operator!=(const SparseDistributedRepresentation &sdr) const
        {  return not ((*this) == sdr); }

    /**
     * Serialization routines.  See Serializable.hpp
     */
    CerealAdapter;

    template<class Archive>
    void save_ar(Archive & ar) const
    {
        getSparse(); // to make sure sparse is valid.
        ar(cereal::make_nvp("dimensions", dimensions_), cereal::make_nvp("sparse", sparse_) );
    }

    template<class Archive>
    void load_ar(Archive & ar)
    {
        ar( dimensions_, sparse_ );
        initialize( dimensions_ );
        setSparseInplace();
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
    UInt addCallback(SDR_callback_t callback) const;

    /**
     * Remove a previously registered callback.
     *
     * @param UInt Handle which was returned by addCallback when you registered
     * your callback.
     */
    void removeCallback(UInt index) const;

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
    UInt addDestroyCallback(SDR_callback_t callback) const;

    /**
     * Remove a previously registered destroy callback.
     *
     * @param UInt Handle which was returned by addDestroyCallback when you
     * registered your callback.
     */
    void removeDestroyCallback(UInt index) const;

};

typedef SparseDistributedRepresentation SDR;

} // end namespace htm
#endif // end ifndef SDR_HPP
