/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2016, Numenta, Inc.
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
 * Topology helpers
 */

#ifndef NTA_TOPOLOGY_HPP
#define NTA_TOPOLOGY_HPP

#include <vector>
#include <functional>

#include <htm/types/Types.hpp>
#include <htm/types/Sdr.hpp>
#include <htm/utils/Random.hpp>

namespace htm {

/**
 * Topology_t is a function which returns the pool of potential synapses for a
 * given cell.
 *
 * Argument 1: is an SDR representing the postsynaptic cell.  Topology functions
 * return the inputs which may connect to this cell.  This SDR contains a single
 * true bit.
 *
 * Argument 2: is the dimensions of the presynaptic cells.
 *
 * Argument 3: is a random number generator to use for reproducible results.
 *
 * Returns: an SDR containing all presynaptic cells which are allowed to connect
 * to the postsynaptic cell.  The dimensions of this SDR must equal argument 2.
 *
 * Example Usage:
 *    // Here is the implementation of the NoTopology function.
 *
 *    Topology_t NoTopology( Real potentialPct )
 *    {
 *      // Define the topology as a lambda function.
 *      return [=]( const SDR          &cell,
 *                  const vector<UInt> &potentialPoolDimensions,
 *                  Random             &rng) -> SDR
 *      {
 *        SDR potentialPool( potentialPoolDimensions );
 *        potentialPool.randomize( potentialPct, rng );
 *        return potentialPool;
 *      };
 *    }
 */
typedef std::function<SDR (const SDR&, const std::vector<UInt>&, Random&)> Topology_t;

/**
 * @param potentialRadius: This parameter determines the extent of the
 *       input that each output can potentially be connected to. This
 *       can be thought of as the input bits that are visible to each
 *       output, or a 'receptive field' of the field of vision. A large
 *       enough value will result in global coverage, meaning
 *       that each output can potentially be connected to every input
 *       bit. This parameter defines a square (or hyper square) area: an
 *       output will have a max square potential pool with sides of
 *       length (2 * potentialRadius + 1).
 *
 * @param potentialPct: The percent of the inputs, within a output's
 *       potential radius, that an output can be connected to. If set to
 *       1, the output will be connected to every input within its
 *       potential radius. This parameter is used to give each output a
 *       unique potential pool when a large potentialRadius causes
 *       overlap between the outputs. At initialization time we choose
 *       ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
 *       input bits to comprise the output's potential pool.
 *
 * @param wrapAround: boolean value that determines whether or not inputs
 *       at the beginning and end of an input dimension are considered
 *       neighbors for the purpose of mapping inputs to outputs.
 */
Topology_t  DefaultTopology(Real potentialPct,
                            Real potentialRadius,
                            bool wrapAround);

/**
 * All inputs have a uniformly equal probability of being sampled into an
 * outputs potential pool.  This does not depend on the relative locations of
 * the input and output.
 *
 * @param potentialPct: The percent of inputs which each output is potentially
 *        connected to.
 */
Topology_t NoTopology(Real potentialPct);

/**
 * Translate an index into coordinates, using the given coordinate system.
 *
 * @param index
 * The index of the point. The coordinates are expressed as a single index
 * by using the dimensions as a mixed radix definition. For example, in
 * dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.
 *
 * @param dimensions
 * The coordinate system.
 *
 * @returns
 * A vector of coordinates of length dimensions.size().
 */
std::vector<UInt> coordinatesFromIndex(UInt index,
                                       const std::vector<UInt> &dimensions);

/**
 * Translate coordinates into an index, using the given coordinate system.
 *
 * @param coordinates
 * A vector of coordinates of length dimensions.size().
 *
 * @param dimensions
 * The coordinate system.
 *
 * @returns
 * The index of the point. The coordinates are expressed as a single index
 * by using the dimensions as a mixed radix definition. For example, in
 * dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.
 */
UInt indexFromCoordinates(const std::vector<UInt> &coordinates,
                          const std::vector<UInt> &dimensions);

/**
 * A class that lets you iterate over all points within the neighborhood
 * of a point.
 *
 * Usage:
 *   UInt center = 42;
 *   for (UInt neighbor : Neighborhood(center, 10, {100, 100}))
 *   {
 *     if (neighbor == center)
 *     {
 *       // Note that the center is included in the neighborhood!
 *     }
 *     else
 *     {
 *       // Do something with the neighbor.
 *     }
 *   }
 *
 * A point's neighborhood is the n-dimensional hypercube with sides
 * ranging [center - radius, center + radius], inclusive. For example,
 * if there are two dimensions and the radius is 3, the neighborhood is
 * 6x6. Neighborhoods are truncated when they are near an edge.
 *
 * Dimensions aren't copied -- a reference is saved. Make sure the
 * dimensions don't get overwritten while this Neighborhood instance
 * exists.
 *
 * This is designed to be fast. It walks the list of points in the
 * neighborhood without ever creating a list of points.
 *
 * This still could be faster. Because it handles an arbitrary number of
 * dimensions, it has to allocate vectors. It would be faster to have a
 * Neighborhood1D, Neighborhood2D, etc., so that all computation could
 * occur on the stack, but this would put a burden on callers to handle
 * different dimensions counts. Or it would require using polymorphism,
 * using pointers/references and putting the Neighborhood on the heap,
 * which defeats the purpose of avoiding the vector allocations.
 *
 * @param centerIndex
 * The center of this neighborhood. The coordinates are expressed as a
 * single index by using the dimensions as a mixed radix definition. For
 * example, in dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.
 *
 * @param radius
 * The radius of this neighborhood about the centerIndex.
 *
 * @param dimensions
 * The dimensions of the world outside this neighborhood.
 *
 * @returns
 * An object which supports C++ range-based for loops. Each iteration of
 * the loop returns a point in the neighborhood. Each point is expressed
 * as a single index.
 */
class Neighborhood {
public:
  Neighborhood(UInt centerIndex, UInt radius,
               const std::vector<UInt> &dimensions);

  class Iterator {
  public:
    Iterator(const Neighborhood &neighborhood, bool end);
    bool operator!=(const Iterator &other) const;
    UInt operator*() const;
    const Iterator &operator++();

  private:
    void advance_();

    const Neighborhood &neighborhood_;
    std::vector<Int> offset_;
    bool finished_;
  };

  Iterator begin() const;
  Iterator end() const;

private:
  const std::vector<UInt> centerPosition_;
  const std::vector<UInt> &dimensions_;
  const UInt radius_;
};

/**
 * Like the Neighborhood class, except that the neighborhood isn't
 * truncated when it's near an edge. It wraps around to the other side.
 *
 * @param centerIndex
 * The center of this neighborhood. The coordinates are expressed as a
 * single index by using the dimensions as a mixed radix definition. For
 * example, in dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.
 *
 * @param radius
 * The radius of this neighborhood about the centerIndex.
 *
 * @param dimensions
 * The dimensions of the world outside this neighborhood.
 *
 * @returns
 * An object which supports C++ range-based for loops. Each iteration of
 * the loop returns a point in the neighborhood. Each point is expressed
 * as a single index.
 */
class WrappingNeighborhood {
public:
  WrappingNeighborhood(UInt centerIndex, UInt radius,
                       const std::vector<UInt> &dimensions);

  class Iterator {
  public:
    Iterator(const WrappingNeighborhood &neighborhood, bool end);
    bool operator!=(const Iterator &other) const;
    UInt operator*() const;
    const Iterator &operator++();

  private:
    void advance_();

    const WrappingNeighborhood &neighborhood_;
    std::vector<Int> offset_;
    bool finished_;
  };

  Iterator begin() const;
  Iterator end() const;

private:
  const std::vector<UInt> centerPosition_;
  const std::vector<UInt> &dimensions_;
  const UInt radius_;
};

} // end namespace htm

#endif // NTA_TOPOLOGY_HPP
