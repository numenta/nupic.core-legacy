/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.
 *               2019, David McDougall
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
 * --------------------------------------------------------------------- */

/** @file
 * PyBind11 bindings for Topology functions.
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <vector>
using namespace std;

#include <htm/utils/Topology.hpp>
using namespace htm;

namespace htm_ext {

    void init_Topology(py::module& m)
    {
        m.def("DefaultTopology", &DefaultTopology,
R"(Argument potentialRadius: This parameter determines the extent of the
    input that each output can potentially be connected to. This
    can be thought of as the input bits that are visible to each
    output, or a 'receptive field' of the field of vision. A large
    enough value will result in global coverage, meaning
    that each output can potentially be connected to every input
    bit. This parameter defines a square (or hyper square) area: an
    output will have a max square potential pool with sides of
    length (2 * potentialRadius + 1).

Argument potentialPct: The percent of the inputs, within a output's
    potential radius, that an output can be connected to. If set to
    1, the output will be connected to every input within its
    potential radius. This parameter is used to give each output a
    unique potential pool when a large potentialRadius causes
    overlap between the outputs. At initialization time we choose
    ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
    input bits to comprise the output's potential pool.

Argument wrapAround: boolean value that determines whether or not inputs
    at the beginning and end of an input dimension are considered
    neighbors for the purpose of mapping inputs to outputs.)",
            py::arg("potentialPct"),
            py::arg("potentialRadius"),
            py::arg("wrapAround"));

        m.def("NoTopology", &NoTopology,
R"(All inputs have a uniformly equal probability of being sampled into an
outputs potential pool.  This does not depend on the relative locations of
the input and output.

Argument potentialPct: The percent of inputs which each output is potentially
     connected to.)",
            py::arg("potentialPct"));


        m.def("coordinatesFromIndex", &coordinatesFromIndex,
R"(Translate an index into coordinates, using the given coordinate system.

Argument index of the point. The coordinates are expressed as a
single index by using the dimensions as a mixed radix definition. For example,
in dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.

Argument dimensions is the coordinate system.

Returns a vector of coordinates of length len(dimensions).)",
            py::arg("index"),
            py::arg("dimensions"));

        m.def("indexFromCoordinates", &indexFromCoordinates,
R"(Translate coordinates into an index, using the given coordinate system.

Argument coordinates is a vector of coordinates of length len(dimensions).

Argument dimensions is the coordinate system.

Returns the index of the point. The coordinates are expressed as a single index
by using the dimensions as a mixed radix definition. For example, in
dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.)",
            py::arg("index"),
            py::arg("dimensions"));

        m.def("neighborhood", [](UInt centerIndex, UInt radius,
                                 const std::vector<UInt> &dimensions)
        {
            vector<UInt> neighbors;
            for( auto idx : Neighborhood( centerIndex, radius, dimensions )) {
                neighbors.push_back( idx );
            }
            return neighbors;
        },
R"(Returns all points within the neighborhood of a point.

A point's neighborhood is the n-dimensional hypercube with sides
ranging [center - radius, center + radius], inclusive. For example,
if there are two dimensions and the radius is 3, the neighborhood is
6x6. Neighborhoods are truncated when they are near an edge.

Argument centerIndex is the center of this neighborhood. The coordinates are
expressed as a single index by using the dimensions as a mixed radix definition.
For example, in dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.

Argument radius is of this neighborhood about the centerIndex.

Argument dimensions are of the world outside this neighborhood.

Returns List of points in the neighborhood. Each point is expressed
as a single index.)");

        m.def("wrappingNeighborhood", [](UInt centerIndex, UInt radius,
                                         const std::vector<UInt> &dimensions)
        {
            vector<UInt> neighbors;
            for( auto idx : WrappingNeighborhood( centerIndex, radius, dimensions )) {
                neighbors.push_back( idx );
            }
            return neighbors;
        },
R"(Like the Neighborhood class, except that the neighborhood isn't
truncated when it's near an edge. It wraps around to the other side.

Argument centerIndex is the center of this neighborhood. The coordinates are
expressed as a single index by using the dimensions as a mixed radix definition.
For example, in dimensions 42x10, the point [1, 4] is index 1*10 + 4 = 14.

Argument radius is of this neighborhood about the centerIndex.

Argument dimensions are of the world outside this neighborhood.

Returns List of points in the neighborhood. Each point is expressed
as a single index.)");

    } // End function init_Topology
} // End namespace htm_ext
