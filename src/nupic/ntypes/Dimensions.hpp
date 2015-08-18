
/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
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
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file 
 * Interface for the Dimensions class
 */

#ifndef NTA_DIMENSIONS_HPP
#define NTA_DIMENSIONS_HPP

#include <vector>
#include <sstream>

namespace nupic
{
  /**
   * @typedef Coordinate
   * 
   * A Coordinate is the location of a single cell in an n-dimensional
   * grid described by a Dimensions object.
   *
   * It's a direct @c typedef, so it has the exactly the same interface as 
   * @c std::vector<size_t> . A value with the index of `i` in the vector 
   * represents the location of the cell along the `i`th dimension.
   *
   * @note It must have the same number of dimensions as its corresponding 
   * Dimensions object.
   *
   * @internal
   *  
   * Because a vector of a basic type can be directly wrapped
   * by swig, we do not need a separate class. 
   *
   * @endinternal
   */
  typedef std::vector<size_t> Coordinate;

  /**
   * Represents the dimensions of a Region.
   *
   * A Dimensions object is an n-dimensional grid, consists of many cells, and 
   * each dimension has a size, i.e. how many cells can there be along this dimension. 
   * 
   * A node within a Region is represented by a cell of a n-dimensional grid,
   * identified by a Coordinate.
   *
   * It's implemented by a @c vector of @c size_t plus a few methods for 
   * convenience and for wrapping.
   *
   * @nosubgrouping
   *
   */
  class Dimensions : public std::vector<size_t>
  {
  public:
    /** 
     *
     * @name Constructors
     *
     * @{
     * 
     */
    
    /**
     * Create a new Dimensions object.
     *
     * @note Default dimensions are unspecified, see isUnspecified()
     * 
     */
    Dimensions();

    /** 
     * Create a new Dimensions object from a @c std::vector<size_t>.
     * 
     * @param v
     *        A @c std::vector of @c size_t, the value with the index of @a n
     *        is the size of the @a n th dimension 
     *        
     */
    Dimensions(std::vector<size_t> v);

    /** Create a new 1-dimension Dimensions object.

     * @param x
     *        The size of the 1st dimension
     * 
     */
    Dimensions(size_t x);

    /** 
     * Create a new 2-dimension Dimensions.
     * 
     * @param x
     *        The size of the 1st dimension
     * @param y
     *        The size of the 2nd dimension
     */
    Dimensions(size_t x, size_t y);

    /** 
     * Create a new 3-dimension Dimensions.
     * 
     * @param x
     *        The size of the 1st dimension
     * @param y
     *        The size of the 2nd dimension
     * @param z
     *        The size of the 3rd dimension
     */
    Dimensions(size_t x, size_t y, size_t z);

    /** 
     *
     * @}
     * 
     * @name Properties
     *
     * @{
     * 
     */

    /**
     * Get the count of cells in the grid, which is the product of the sizes of 
     * the dimensions.
     * 
     * @returns 
     *        The count of cells in the grid.
     */
    size_t 
    getCount() const;

    /**
     *
     * Get the number of dimensions.
     * 
     * @returns number of dimensions
     * 
     */
    size_t
    getDimensionCount() const;

    /**
     * Get the size of a dimension.
     * 
     * @param index
     *        The index of the dimension
     *
     * @returns
     *        The size of the dimension with the index of @a index
     *
     * @note Do not confuse @a index with "linear index" as in getIndex()
     */
    size_t
    getDimension(size_t index) const;

    /** 
     *
     * @}
     * 
     * @name Boolean properties
     *
     * There are two "special" values for dimensions:
     * 
     * * Dimensions of `[]` (`dims.size()==0`) means "not yet known" aka 
     * "unspecified", see isUnspecified()
     * * Dimensions of `[0]`  (`dims.size()==1 && dims[0] == 0`) means 
     * "don't care", see isDontcare()
     *
     * @{
     * 
     */

    /**
     * Tells whether the Dimensions object is "unspecified".
     * 
     * @returns
     *     Whether the Dimensions object is "unspecified"
     *
     * @see isSpecified()
     */
    bool
    isUnspecified() const;

    /**
     *
     * Tells whether the Dimensions object is "don't care".
     * 
     * @returns
     *     Whether the Dimensions object is "don't care"
     */
    bool
    isDontcare() const;

    /**
     * Tells whether the Dimensions object is "specified".
     *
     * A "specified" Dimensions object satisfies all following conditions:
     *
     *   * "valid"
     *   * NOT "unspecified"
     *   * NOT "don't care"
     *   
     * @returns
     *       Whether the Dimensions object is "specified"
     *
     * @note It's not the opposite of isUnspecified()!
     */
    bool
    isSpecified() const;

    /**
     * Tells whether the sizes of all dimensions are 1.
     *
     * @returns
     *       Whether the sizes of all dimensions are 1, e.g. [1], [1 1], [1 1 1], etc.
     */
    bool 
    isOnes() const;

    /**
     * Tells whether Dimensions is "valid".
     * 
     * A Dimensions object is valid if it specifies actual dimensions, i.e. all
     * dimensions have a size greater than 0, or is a special value 
     * ("unspecified"/"don't care"). 
     * 
     * A Dimensions object is invalid if any dimensions are 0 (except for "don't care")
     * 
     * @returns
     *       Whether Dimensions is "valid"
     */
    bool
    isValid() const;

    /** 
     *
     * @}
     * 
     * @name Coordinate<->index mapping
     *
     * Coordinate<->index mapping is in lower-major order, i.e.
     * for Region with dimensions `[2,3]`:
     * 
     *     [0,0] -> index 0
     *     [1,0] -> index 1
     *     [0,1] -> index 2
     *     [1,1] -> index 3
     *     [0,2] -> index 4
     *     [1,2] -> index 5
     *
     * @{
     * 
     */

    /**
     * Convert a Coordinate to a linear index (in lower-major order).
     *
     * @param coordinate
     *        The coordinate to be converted
     *   
     * @returns
     *        The linear index corresponding to @a coordinate
     */
    size_t
    getIndex(const Coordinate& coordinate) const;

    /**
     * Convert a linear index (in lower-major order) to a Coordinate.
     *
     * @param index
     *        The linear index to be converted
     * 
     * @returns
     *        The Coordinate corresponding to @a index
     */
    Coordinate
    getCoordinate(const size_t index) const;

    /** 
     *
     * @}
     * 
     * @name Misc
     *
     * @{
     * 
     */
    
    /**
     *
     * Convert the Dimensions object to string representation.
     * 
     * In most cases, we want a human-readable string, but for
     * serialization we want only the actual dimension values
     * 
     * @param humanReadable
     *        The default is @c true, make the string human-readable,
     *        set to @c false for serialization
     *        
     * @returns 
     *        The string representation of the Dimensions object
     */
    std::string
    toString(bool humanReadable=true) const;

    /**
     * Promote the Dimensions object to a new dimensionality.
     * 
     * @param newDimensionality
     *        The new dimensionality to promote to, it can be greater than, 
     *        smaller than or equal to current dimensionality
     *
     * @note The sizes of all dimensions must be 1( i.e. isOnes() returns true), 
     * or an exception will be thrown.
     */
    void
    promote(size_t newDimensionality);

    /**
     * The equivalence operator.
     *
     * Two Dimensions objects will be considered equivalent, if any of the 
     * following satisfies:
     *
     * * They have the same number of dimensions and the same size for every 
     * dimension.
     * * Both of them have the size of 1 for everything dimensions, despite of 
     * how many dimensions they have, i.e. isOnes() returns @c true for both 
     * of them. Some linking scenarios require us to treat [1] equivalent to [1 1] etc.
     *
     * @param dims2
     *        The Dimensions object being compared
     *
     * @returns
     *        Whether this Dimensions object is equivalent to @a dims2.
     *       
     */
    bool
    operator == (const Dimensions& dims2) const;

    /**
     * The in-equivalence operator, the opposite of operator==().
     *
     * @param dims2
     *        The Dimensions object being compared
     *
     * @returns
     *        Whether this Dimensions object is not equivalent to @a dims2.
     */
    bool
    operator != (const Dimensions& dims2) const;

    /** 
     *
     * @}
     *
     */


#ifdef NTA_INTERNAL
    friend std::ostream& operator<<(std::ostream& f, const Dimensions&);
#endif


  };

}


#endif // NTA_DIMENSIONS_HPP
