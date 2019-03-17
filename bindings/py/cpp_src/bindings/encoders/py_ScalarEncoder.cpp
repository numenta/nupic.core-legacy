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
namespace py = pybind11;

#include <nupic/encoders/ScalarEncoder.hpp>
#include <nupic/types/Sdr.hpp>

namespace nupic_ext
{
  using namespace nupic::encoders;
  using nupic::sdr::SDR;

  void init_ScalarEncoder(py::module& m)
  {
    py::class_<ScalarEncoderParameters> py_ScalarEncParams(m, "ScalarEncoderParameters",
        R"(

The following three (3) members define the total number of bits in the output:
     size,
     radius,
     resolution.

These are mutually exclusive and only one of them should be non-zero when
constructing the encoder.)");

    py_ScalarEncParams.def(py::init<>(), R"()");

    py_ScalarEncParams.def_readwrite("minimum", &ScalarEncoderParameters::minimum,
R"(This defines the range of the input signal. These endpoints are inclusive.)");

    py_ScalarEncParams.def_readwrite("maximum", &ScalarEncoderParameters::maximum,
R"(This defines the range of the input signal. These endpoints are inclusive.)");

    py_ScalarEncParams.def_readwrite("clipInput", &ScalarEncoderParameters::clipInput,
R"(This determines whether to allow input values outside the
range [minimum, maximum].
If true, the input will be clipped into the range [minimum, maximum].
If false, inputs outside of the range will raise an error.)");

    py_ScalarEncParams.def_readwrite("periodic", &ScalarEncoderParameters::periodic,
R"(This controls what happens near the edges of the input range.

If true, then the minimum & maximum input values are adjacent and the first and
last bits of the output SDR are also adjacent.  The contiguous block of 1's
wraps around the end back to the begining.

If false, then minimum & maximum input values are the endpoints of the input
range, are not adjacent, and activity does not wrap around.)");

    py_ScalarEncParams.def_readwrite("activeBits", &ScalarEncoderParameters::activeBits,
R"(This is the number of true bits in the encoded output SDR. The output
encodings will have a contiguous block of this many 1's.)");

    py_ScalarEncParams.def_readwrite("sparsity", &ScalarEncoderParameters::sparsity,
R"(This is an alternative way to specify the the number of active bits.
Sparsity requires that the size to also be specified.
Specify only one of: activeBits or sparsity.)");

    py_ScalarEncParams.def_readwrite("size", &ScalarEncoderParameters::size,
R"(This is the total number of bits in the encoded output SDR.)");

    py_ScalarEncParams.def_readwrite("radius", &ScalarEncoderParameters::radius,
R"(Two inputs separated by more than the radius have non-overlapping
representations. Two inputs separated by less than the radius will in general
overlap in at least some of their bits. You can think of this as the radius of
the input.)");

    py_ScalarEncParams.def_readwrite("resolution", &ScalarEncoderParameters::resolution,
R"(Two inputs separated by greater than, or equal to the resolution are guaranteed
to have different representations.)");


    py::class_<ScalarEncoder> py_ScalarEnc(m, "ScalarEncoder",
R"(Encodes a real number as a contiguous block of 1's.

The ScalarEncoder encodes a numeric (floating point) value into an array of
bits. The output is 0's except for a contiguous block of 1's. The location of
this contiguous block varies continuously with the input value.

TODO, Example Usage & unit test for it.)");

    py_ScalarEnc.def(py::init<ScalarEncoderParameters&>(), R"()");
    py_ScalarEnc.def_property_readonly("parameters",
        [](const ScalarEncoder &self) { return self.parameters; },
R"(Contains the parameter structure which this encoder uses internally. All
fields are filled in automatically.)");

    py_ScalarEnc.def("encode", &ScalarEncoder::encode, R"()");

    py_ScalarEnc.def("encode", [](ScalarEncoder &self, nupic::Real64 value) {
        auto output = new SDR( self.dimensions );
        self.encode( value, *output );
        return output; },
R"()");
  }
}
