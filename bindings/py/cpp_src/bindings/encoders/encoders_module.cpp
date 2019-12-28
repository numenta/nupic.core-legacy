/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, chhenning
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
 * Encoders bindings Module file for pybind11
 */

#include <bindings/suppress_register.hpp>  //must be before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace htm_ext
{
    void init_ScalarEncoder(py::module&);
    void init_RDSE(py::module&);
    void init_SimHashDocumentEncoder(py::module&);
    void init_DateEncoder(py::module&);
}

using namespace htm_ext;

PYBIND11_MODULE(encoders, m) {
    m.doc() =
R"(Encoders convert values into sparse distributed representation.

There are several critical properties which all encoders must have:

    1) Semantic similarity:  Similar inputs should have high overlap.  Overlap
    decreases smoothly as inputs become less similar.  Dissimilar inputs have
    very low overlap so that the output representations are not easily confused.

    2) Stability:  The representation for an input does not change during the
    lifetime of the encoder.

    3) Sparsity: The output SDR should have a similar sparsity for all inputs and
    have enough active bits to handle noise and subsampling.

Reference: https://arxiv.org/pdf/1602.05925.pdf


CategoryEncoders:

    To encode categories of input, make a ScalarEncoder or a Random Distributed
Scalar Encoder (RDSE), and set the parameter category=True.  Then enumerate your
categories into integers before encoding them. )";

    init_ScalarEncoder(m);
    init_RDSE(m);
    init_SimHashDocumentEncoder(m);
    init_DateEncoder(m);
}
