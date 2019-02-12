/* ----------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2019, David McDougall
 * The following terms and conditions apply:
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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/types/Sdr.hpp>
#include <nupic/types/SdrProxy.hpp>
#include <nupic/utils/StringUtils.hpp>  // trim

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_SDR(py::module& m)
    {
        py::class_<SDR> py_SDR(m, "SDR",
R"(Sparse Distributed Representation

This class manages the specification and momentary value of a Sparse Distributed
Representation (SDR).  An SDR is a group of boolean values which represent the
state of a group of neurons or their associated processes.

SDR's have three commonly used data formats which are:
*   dense
*   sparse
*   flatSparse
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
    X.sparse = [[0, 1, 2], [1, 1, 2]]
    X.flatSparse = [ 1, 4, 8 ]

    # Access data in any format, SDR will automatically convert data formats,
    # even if it was not the format used by the most recent assignment to the
    # SDR.
    X.dense      -> [[ 0, 1, 0 ],
                     [ 0, 1, 0 ],
                     [ 0, 0, 1 ]]
    X.sparse     -> [[ 0, 1, 2 ], [1, 1, 2 ]]
    x.flatSparse -> [ 1, 4, 8 ]

    # Data format conversions are cached, and when an SDR value changes the
    # cache is cleared.
    X.flatSparse = [1, 2, 3] # Assign new data to the SDR, clearing the cache.
    X.dense     # This line will convert formats.
    X.dense     # This line will resuse the result of the previous line

Assigning a value to the SDR requires copying the data from Python into C++. To
avoid this copy operation: modify sdr.dense inplace, and assign it to itself.
This class will detect that it's being given it's own data and will omit the
copy operation.

Example Usage of In-Place Assignment:
    X    = SDR((1000, 1000))
    data = X.dense
    data[  0,   4] = 1
    data[444, 444] = 1
    X.dense = data
    X.flatSparse -> [ 4, 444444 ]

Data Validity Warning:  The SDR allocates and frees its data when it is
constructed and deconstructed, respectively.  If you have a numpy array which
came from an SDR, then you either need to copy the data or ensure that the SDR
remains in scope by holding a reference to it.
Examples of Invalid Data Accesses:
    A = SDR( dimensions )
    use_after_free = A.dense
    del A
    # The variable "use_after_free" now references data which has been deallocated.
    # Another way this can happen is:
    use_after_free = SDR( dimensions ).sparse
    use_after_free = SDR( dimensions ).flatSparse

Data Validity Warning:  After assigning a new value to the SDR, all existing
numpy arrays of data are invalid.  In order to get the latest copy of the data,
re-access the data from the SDR.  Examples:
    A = SDR( dimensions )
    out_of_date = A.dense
    A.flatSparse = []
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
                return SDR({ dimensions }); }),
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

        py_SDR.def("zero", &SDR::zero,
R"(Set all of the values in the SDR to false.  This method overwrites the SDRs
current value.)");

        py_SDR.def_property("dense",
            [](SDR &self) {
                auto capsule = py::capsule(&self, [](void *self) {});
                vector<UInt> strides( self.dimensions.size(), 0u );
                auto z = sizeof(Byte);
                for(int i = self.dimensions.size() - 1; i >= 0; --i) {
                    strides[i] = z;
                    z *= self.dimensions[i];
                }
                return py::array(self.dimensions, strides, self.getDense().data(), capsule);
            },
            [](SDR &self, py::array_t<Byte> dense) {
                py::buffer_info buf = dense.request();
                NTA_CHECK( (UInt) buf.ndim == self.dimensions.size() );
                for(auto dim = 0u; dim < self.dimensions.size(); dim++) {
                    NTA_CHECK( (UInt) buf.shape[dim] == self.dimensions[dim] );
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

        py_SDR.def_property("flatSparse",
            [](SDR &self) {
                auto capsule = py::capsule(&self, [](void *self) {});
                return py::array(self.getSum(), self.getFlatSparse().data(), capsule);
            },
            [](SDR &self, SDR_flatSparse_t data) {
                NTA_CHECK( data.size() <= self.size );
                self.setFlatSparse( data ); },
R"(A numpy array containing the indices of only the true values in the SDR.
These are indices into the flattened SDR. This format allows for quickly
accessing all of the true bits in the SDR.)");

        py_SDR.def_property("sparse",
            [](SDR &self) {
                auto capsule = py::capsule(&self, [](void *self) {});
                auto outer   = py::list();
                auto sparse  = self.getSparse().data();
                for(auto dim = 0u; dim < self.dimensions.size(); dim++) {
                    auto vec = py::array(sparse[dim].size(), sparse[dim].data(), capsule);
                    outer.append(vec);
                }
                return outer;
            },
            [](SDR &self, SDR_sparse_t data) {
                NTA_CHECK( data.size() == self.dimensions.size() );
                self.setSparse( data ); },
R"(List of numpy arrays, containing the indices of only the true values in the
SDR.  This is a list of lists: the outter list contains an entry for each
dimension in the SDR. The inner lists contain the coordinates of each true bit.
The inner lists run in parallel. This format is useful because it contains the
location of each true bit inside of the SDR's dimensional space.)");

        py_SDR.def("setSDR", [](SDR &self, SDR &other) {
            NTA_CHECK( self.dimensions == other.dimensions );
            self.setSDR( other ); },
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
            [](SDR &self, Real sparsity, UInt seed) {
            Random rng( seed );
            self.randomize( sparsity, rng ); },
R"(Make a random SDR, overwriting the current value of the SDR.  The result has
uniformly random activations.

Argument sparsity is the fraction of bits to set to true.  After calling this
method sdr.getSparsity() will return this sparsity, rounded to the nearest
fraction of self.size.

Optional argument seed is used for the random number generator.  Seed 0 is
special, it is replaced with the system time  The default seed is 0.)",
            py::arg("sparsity"),
            py::arg("seed") = 0u);

        py_SDR.def("addNoise", [](SDR &self, Real fractionNoise, UInt seed = 0) {
            Random rng( seed );
            self.addNoise( fractionNoise, rng ); },
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

        py_SDR.def("__str__", [](SDR &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() ); });

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


        py::class_<SDR_Proxy, SDR> py_Proxy(m, "SDR_Proxy",
R"(SDR_Proxy presents a view onto an SDR.
    * Proxies always have the same value as their source SDR.
    * Proxies can be used in place of an SDR.
    * Proxies are read only.
    * Proxies can have different dimensions than their source SDR.

Example Usage:
    # Convert SDR dimensions from (4 x 4) to (8 x 2)
    A = SDR([ 4, 4 ])
    B = SDR_Proxy( A, [8, 2])
    A.sparse =  ([1, 1, 2], [0, 1, 2])
    B.sparse -> ([2, 2, 5], [0, 1, 0])

SDR_Proxy supports pickle, however loading a pickled SDR proxy will return an
SDR, not an SDR_Proxy.)");

        py_Proxy.def( py::init<SDR&, vector<UInt>>(),
R"(Argument sdr is the data source make a view of.

Argument dimensions A list of dimension sizes, defining the shape of the SDR.
Optional, if not given then this Proxy will have the same dimensions as the
given SDR.)",
            py::arg("sdr"), py::arg("dimensions"));

        py_Proxy.def( py::init<SDR&>(), py::arg("sdr"));
    }
}
