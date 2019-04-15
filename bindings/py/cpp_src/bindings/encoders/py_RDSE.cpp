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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

#include <nupic/encoders/RandomDistributedScalarEncoder.hpp>

namespace py = pybind11;

using namespace nupic;
using namespace nupic::encoders;

namespace nupic_ext
{
    void init_RDSE(py::module& m)
    {
        py::class_<RDSE_Parameters> py_RDSE_args(m, "RDSE_Parameters",
R"(Parameters for the RandomDistributedScalarEncoder (RDSE)

Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
of them.

Members "radius" & "resolution" are mutually exclusive, specify exactly one of
them.)");

        py_RDSE_args.def(py::init<>());

        py_RDSE_args.def_readwrite("size", &RDSE_Parameters::size,
R"(Member "size" is the total number of bits in the encoded output SDR.)");

        py_RDSE_args.def_readwrite("sparsity", &RDSE_Parameters::sparsity,
R"(Member "sparsity" is the fraction of bits in the encoded output which this
encoder will activate. This is an alternative way to specify the member
"activeBits".)");

        py_RDSE_args.def_readwrite("activeBits", &RDSE_Parameters::activeBits,
R"(Member "activeBits" is the number of true bits in the encoded output SDR.)");

        py_RDSE_args.def_readwrite("radius", &RDSE_Parameters::radius,
R"(Two inputs separated by more than the radius will have non-overlapping
representations. Two inputs separated by less than the radius will in general
overlap in at least some of their bits. You can think of this as the radius of
the input.)");

        py_RDSE_args.def_readwrite("resolution", &RDSE_Parameters::resolution,
R"(Two inputs separated by greater than, or equal to the resolution are
guaranteed to have different representations.)");

        py_RDSE_args.def_readwrite("seed", &RDSE_Parameters::seed,
R"(Member "seed" forces different encoders to produce different outputs, even if
the inputs and all other parameters are the same.  Two encoders with the same
seed, parameters, and input will produce identical outputs.

The seed 0 is special.  Seed 0 is replaced with a random number.)");


        py::class_<RDSE> py_RDSE(m, "RDSE",
R"(Encodes a real number as a set of randomly generated activations.

The Random Distributed Scalar Encoder (RDSE) encodes a numeric scalar (floating
point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
This encoder does not need to know the minimum and maximum of the input
range.  It does not assign an input->output mapping at construction.  Instead
the encoding is determined at runtime.

Note: This implementation differs from Numenta's original RDSE.  The original
RDSE saved all associations between inputs and active bits for the lifetime
of the encoder.  This allowed it to guarantee a good set of random
activations which didn't conflict with any previous encoding.  It also allowed
the encoder to decode an SDR into the input value which likely created it.
This RDSE does not save the association between inputs and active bits.  This
is faster and uses less memory.  It relies on the random & distributed nature
of SDRs to prevent conflicts between different encodings.  This method does
not allow for decoding SDRs into the inputs which likely created it.

To inspect this run:
$ python -m nupic.encoders.rdse --help)");
        py_RDSE.def(py::init<RDSE_Parameters>());

        py_RDSE.def_property_readonly("parameters",
            [](RDSE &self) { return self.parameters; },
R"(Contains the parameter structure which this encoder uses internally. All
fields are filled in automatically.)");

        py_RDSE.def_property_readonly("dimensions",
            [](RDSE &self) { return self.dimensions; });
        py_RDSE.def_property_readonly("size",
            [](RDSE &self) { return self.size; });

        py_RDSE.def("encode", &RDSE::encode, R"()");

        py_RDSE.def("encode", [](RDSE &self, Real64 value) {
            auto sdr = new sdr::SDR({self.size});
            self.encode(value, *sdr);
            return sdr;
        });
    }
}
