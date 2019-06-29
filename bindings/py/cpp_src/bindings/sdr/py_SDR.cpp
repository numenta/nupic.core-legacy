/* ----------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, David McDougall
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

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/types/Sdr.hpp>

#include <memory> // shared_ptr

namespace py = pybind11;

using namespace std;
using namespace htm;

namespace htm_ext
{
    void init_SDR(py::module& m)
    {
        py::class_<SDR, shared_ptr<SDR>> py_SDR(m, "SDR",
R"(Sparse Distributed Representation

This class manages the specification and momentary value of a Sparse Distributed
Representation (SDR).  An SDR is a group of boolean values which represent the
state of a group of neurons or their associated processes.

SDR's have three commonly used data formats which are:
*   dense
*   sparse
*   coordinates
The SDR class has three magic properties, one for each of these data formats.
These properties are the primary way of accessing the SDR's data.  When these
properties are read from, the data is automatically converted to the requested
format and is cached so getting a value in one format many times incurs no extra
performance cost.  Assigning to the SDR via any one of these properties clears
the cached values and causes them to be recomputed as needed.

Example usage:
    # Make an SDR with 9 values, arranged in a (3 x 3) grid.
    X = SDR(dimensions = (3, 3))

    # These three statements are equivalent.
    X.dense  = [[0, 1, 0],
                [0, 1, 0],
                [0, 0, 1]]
    X.sparse = [ 1, 4, 8 ]
    X.coordinates = [[0, 1, 2], [1, 1, 2]]

    # Access data in any format, SDR will automatically convert data formats,
    # even if it was not the format used by the most recent assignment to the
    # SDR.
    X.dense  -> [[ 0, 1, 0 ],
                 [ 0, 1, 0 ],
                 [ 0, 0, 1 ]]
    x.sparse -> [ 1, 4, 8 ]
    X.coordinates -> [[ 0, 1, 2 ], [1, 1, 2 ]]

    # Data format conversions are cached, and when an SDR value changes the
    # cache is cleared.
    X.sparse = [1, 2, 3] # Assign new data to the SDR, clearing the cache.
    X.dense     # This line will convert formats.
    X.dense     # This line will resuse the result of the previous line

Assigning a value to the SDR requires copying the data from Python into C++. To
avoid this copy operation: modify sdr.dense inplace, and assign it to itself.
This class will detect that it's being given it's own data and will omit the
copy operation.

Example Usage of In-Place Assignment:
    X    = SDR((1000, 1000))   # Initial value is all zeros
    data = X.dense
    data[  0,   4] = 1
    data[444, 444] = 1
    X.dense = data
    X.sparse -> [ 4, 444444 ]

Data Validity Warning:  After assigning a new value to the SDR, all existing
numpy arrays of data are invalid.  In order to get the latest copy of the data,
re-access the data from the SDR.  Examples:
    A = SDR( dimensions )
    out_of_date = A.dense
    A.sparse = []
    # The variable "out_of_date" is now liable to be overwritten.
    A.dense = out_of_date   # This does not work, since the data is invalid.
)");

        py_SDR.def(
            py::init<vector<UInt>>(),
R"(Create an SDR object.  The initial value is all zeros.

Argument dimensions is a list of dimension sizes, defining the shape of the SDR.
The product of the dimensions must be greater than zero.)",
            py::arg("dimensions"));

        py_SDR.def(
            py::init( [](UInt dimensions) {
                return new SDR({ dimensions }); }),
R"(Create an SDR object.  The initial value is all zeros.

Argument dimensions is a single integer dimension size, defining a 1-dimensional
SDR.  Must be greater than zero.)",
            py::arg("dimensions"));

        py_SDR.def(
            py::init<SDR>(),
R"(Initialize this SDR as a deep copy of the given SDR.  This SDR and the given
SDR will have no shared data and they can be modified without affecting each
other.)",
            py::arg("sdr"));

        py_SDR.def_property_readonly("dimensions",
            [](const SDR &self) {
                return self.dimensions; },
            "A list of dimensions of the SDR.");

        py_SDR.def_property_readonly("size",
            [](const SDR &self) {
                return self.size; },
            "The total number of boolean values in the SDR.");

        py_SDR.def("zero", [](SDR *self) { self->zero(); return self; },
R"(Set all of the values in the SDR to false.  This method overwrites the SDRs
current value.)");

        py_SDR.def_property("dense",
            [](shared_ptr<SDR> self) {
                auto destructor = py::capsule( new shared_ptr<SDR>( self ),
                    [](void *keepAlive) {
                        delete reinterpret_cast<shared_ptr<SDR>*>(keepAlive); });
                vector<UInt> strides( self->dimensions.size(), 0u );
                auto z = sizeof(Byte);
                for(int i = (int)self->dimensions.size() - 1; i >= 0; --i) {
                    strides[i] = (UInt)z;
                    z *= self->dimensions[i];
                }
                return py::array(self->dimensions, strides, self->getDense().data(), destructor);
            },
            [](SDR &self, py::array_t<Byte> dense) {
                py::buffer_info buf = dense.request();
                if( buf.ndim == 1 ) {
                    NTA_CHECK( (UInt) buf.shape[0] == self.size )
                        << "Bad input array size! expected " << self.size << ", got " << buf.shape[0];
                }
                else if( (UInt) buf.ndim == self.dimensions.size() ) {
                    for(auto dim = 0u; dim < self.dimensions.size(); dim++) {
                        NTA_CHECK( (UInt) buf.shape[dim] == self.dimensions[dim] );
                    }
                }
                else {
                    NTA_THROW << "Invalid input dimensions!";
                }
                Byte *data = (Byte*) buf.ptr;
                if( data == self.getDense().data() )
                    // We got our own data back, set inplace instead of copying.
                    self.setDense( self.getDense() );
                else
                    self.setDense( data ); },
R"(A numpy array of boolean values, representing all of the bits in the SDR.
This format allows random-access queries of the SDRs values.

After modifying this array you MUST assign the array back into the SDR, in order
to notify the SDR that its dense array has changed and its cached data is out of
date.  If you did't copy this data, then SDR won't copy either.)");

        py_SDR.def_property("sparse",
            [](shared_ptr<SDR> self) {
                auto destructor = py::capsule( new shared_ptr<SDR>( self ),
                    [](void *keepAlive) {
                        delete reinterpret_cast<shared_ptr<SDR>*>(keepAlive); });
                return py::array(self->getSum(), self->getSparse().data(), destructor);
            },
            [](SDR &self, SDR_sparse_t data) {
                NTA_CHECK( data.size() <= self.size );
                // Sort data and check for duplicates.
                if( ! is_sorted( data.begin(), data.end() ))
                    sort( data.begin(), data.end() );
                UInt previous = -1;
                for( const UInt idx : data ) {
                    NTA_CHECK( idx != previous )
                        << "Sparse data must not contain duplicates!";
                    previous = idx;
                }
                self.setSparse( data ); },
R"(A numpy array containing the indices of only the true values in the SDR.
These are indices into the flattened SDR. This format allows for quickly
accessing all of the true bits in the SDR.

Sparse data must contain no duplicates.)");

        py_SDR.def_property("coordinates",
            [](shared_ptr<SDR> self) {
                auto destructor = py::capsule( new shared_ptr<SDR>( self ),
                    [](void *keepAlive) {
                        delete reinterpret_cast<shared_ptr<SDR>*>(keepAlive); });
                auto outer   = py::list();
                auto coords  = self->getCoordinates().data();
                for(auto dim = 0u; dim < self->dimensions.size(); ++dim) {
                    auto vec = py::array(coords[dim].size(), coords[dim].data(), destructor);
                    outer.append(vec);
                }
                return outer;
            },
            [](SDR &self, SDR_coordinate_t data) {
                NTA_CHECK( data.size() == self.dimensions.size() );
                self.setCoordinates( data ); },
R"(List of numpy arrays, containing the coordinates of only the true values in
the SDR.  This is a list of lists: the outter list contains an entry for each
dimension in the SDR. The inner lists contain the coordinates of each true bit.
The inner lists run in parallel. This format is useful because it contains the
location of each true bit inside of the SDR's dimensional space.

Coordinate data must be sorted and contain no duplicates.)");

        py_SDR.def("setSDR", [](SDR *self, SDR &other) {
            NTA_CHECK( self->dimensions == other.dimensions );
            self->setSDR( other );
            return self; },
R"(Deep Copy the given SDR to this SDR.  This overwrites the current value of this
SDR.  This SDR and the given SDR will have no shared data and they can be
modified without affecting each other.)");

        py_SDR.def("getSum", &SDR::getSum,
            "Calculates the number of true values in the SDR.");

        py_SDR.def("getSparsity", &SDR::getSparsity,
R"(Calculates the sparsity of the SDR, which is the fraction of bits which are
true out of the total number of bits in the SDR.
I.E.  sparsity = sdr.getSum() / sdr.size)");

        py_SDR.def("getOverlap", [](SDR &self, SDR &other) {
            NTA_CHECK( self.dimensions == other.dimensions );
            return self.getOverlap( other ); },
"Calculates the number of true bits which both SDRs have in common.");

        py_SDR.def("randomize",
            [](SDR *self, Real sparsity, UInt seed) {
            Random rng( seed );
            self->randomize( sparsity, rng );
            return self; },
R"(Make a random SDR, overwriting the current value of the SDR.  The result has
uniformly random activations.

Argument sparsity is the fraction of bits to set to true.  After calling this
method sdr.getSparsity() will return this sparsity, rounded to the nearest
fraction of self.size.

Optional argument seed is used for the random number generator.  Seed 0 is
special, it is replaced with the system time  The default seed is 0.)",
            py::arg("sparsity"),
            py::arg("seed") = 0u);

        py::module::import("htm.bindings.math");
        py_SDR.def("randomize",
            [](SDR *self, Real sparsity, Random rng) {
            self->randomize( sparsity, rng );
            return self; },
R"(This overload accepts Random Number Generators (RNG) intead of a random seed.
RNGs must be instances of "htm.bindings.math.Random".)",
                py::arg("sparsity"),
                py::arg("rng"));

        py_SDR.def("addNoise", [](SDR *self, Real fractionNoise, UInt seed) {
            Random rng( seed );
            self->addNoise( fractionNoise, rng );
            return self; },
R"(Modify the SDR by moving a fraction of the active bits to different
locations.  This method does not change the sparsity of the SDR, it moves
the locations of the true values.  The resulting SDR has a controlled
amount of overlap with the original.

Argument fractionNoise is the fraction of active bits to swap out.  The original
and resulting SDRs have the following relationship:
    originalSDR.getOverlap( newSDR ) / sparsity == 1 - fractionNoise

Optional argument seed is used for the random number generator.  Seed 0 is
special, it is replaced with the system time.  The default seed is 0.)",
            py::arg("fractionNoise"),
            py::arg("seed") = 0u);

        py_SDR.def("killCells", [](SDR *self, Real fraction, UInt seed) {
            self->killCells( fraction, seed ); return self; },
R"(Modify the SDR by setting a fraction of the bits to zero.

Argument fraction must be between 0 and 1 (inclusive).  This fraction of the
cells in the SDR will be set to zero, regardless of their current state.

Argument seed is for a random number generator.  If not given, this uses the
magic seed 0.  Use the same seed to consistently kill the same cells.)",
            py::arg("fraction"),
            py::arg("seed") = 0u);

        py_SDR.def("__str__", [](SDR &self){
            stringstream buf;
            buf << self;
            return py::str( buf.str() ).attr("strip")(); });

        py_SDR.def("__eq__", [](SDR &self, SDR &other){ return self == other; });
        py_SDR.def("__ne__", [](SDR &self, SDR &other){ return self != other; });

        py_SDR.def(py::pickle(
            [](const SDR& self) {
                std::stringstream ss;
                self.save(ss);
                return py::bytes(ss.str());
        },
            [](const py::bytes& s) {
                std::istringstream ss(s);
                SDR self;
                self.load(ss);
                return self;
        }));

        py_SDR.def("reshape", [](SDR *self, const vector<UInt> &dimensions)
            { self->reshape( dimensions ); return self; },
R"(Change the dimensions of the SDR.  The total size must not change.)");

        py_SDR.def("flatten", [](SDR *self)
            { self->reshape({ self->size }); return self; },
R"(Change the dimensions of the SDR into one big dimension.)");

        py_SDR.def("intersection", [](SDR *self, SDR& inp1, SDR& inp2)
            { self->intersection({ &inp1, &inp2}); return self; },
R"(This method calculates the set intersection of the active bits in each input
SDR.

This method has two overloads:
    1) Accepts two SDRs, for convenience.
    2) Accepts a list of SDRs, must contain at least two SDRs, can contain as
       many SDRs as needed.

In both cases the output is stored in this SDR.  This method modifies this SDR
and discards its current value!

Example Usage:
    A = SDR( 10 )
    B = SDR( 10 )
    X = SDR( 10 )
    A.sparse = [0, 1, 2, 3]
    B.sparse =       [2, 3, 4, 5]
    X.intersection( A, B )
    X.sparse -> [2, 3]
)");
        py_SDR.def("intersection", [](SDR *self, vector<const SDR*> inputs)
            { self->intersection(inputs); return self; });

        py_SDR.def("union", [](SDR *self, SDR& inp1, SDR& inp2)
            { self->set_union({ &inp1, &inp2}); return self; },
R"(This method calculates the set union of the active bits in each input SDR.

The output is stored in this SDR.  This method discards the SDRs current value!

Example Usage:
    A = SDR( 10 )
    B = SDR( 10 )
    U = SDR( 10 )
    A.sparse = [0, 1, 2, 3]
    B.sparse =       [2, 3, 4, 5]
    U.union( A, B )
    U.sparse -> [0, 1, 2, 3, 4, 5]
)");
        py_SDR.def("union", [](SDR *self, vector<const SDR*> inputs)
            { self->set_union(inputs); return self; });

        py_SDR.def("concatenate", [](SDR *self, const SDR& inp1, const SDR& inp2, UInt axis)
            { self->concatenate(inp1, inp2, axis); return self; },
R"(Concatenates SDRs and stores the result in this SDR.

This method has two overloads:
    1) Accepts two SDRs, for convenience.
    2) Accepts a list of SDRs, must contain at least two SDRs, can
       contain as many SDRs as needed.

Argument axis: This can concatenate along any axis, as long as the
result has the same dimensions as this SDR.  The default axis is 0.

The output is stored in this SDR.  This method modifies this SDR
and discards its current value!

Example Usage:
    A = SDR( 10 )
    B = SDR( 10 )
    C = SDR( 20 )
    A.sparse = [0, 1, 2]
    B.sparse = [0, 1, 2]
    C.concatenate( A, B )
    C.sparse == [0, 1, 2, 10, 11, 12]
)",
                py::arg("input1"),
                py::arg("input2"),
                py::arg("axis") = 0u );

        py_SDR.def("concatenate", [](SDR *self, vector<const SDR*> inputs, UInt axis)
            { self->concatenate(inputs, axis); return self; },
                py::arg("inputs"),
                py::arg("axis") = 0u );
    }
}
