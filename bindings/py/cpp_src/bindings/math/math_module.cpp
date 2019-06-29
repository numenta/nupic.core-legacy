/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
 *               2018, chhenning
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
 * PyBind11 Module for Math
 */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace htm_ext
{
    void init_Random(py::module&);
    void init_reals(py::module&);
    void init_Topology(py::module&);
} // namespace htm_ext

using namespace htm_ext;

PYBIND11_MODULE(math, m) {
    m.doc() = 
R"(This module defines several topology functions.  Topology functions return the
pool of potential synapses for a given cell.  Functions DefaultTopology and
NoTopology return topology functions.  Topology functions accept 3 arguments:
  * Argument 1: is an SDR representing the postsynaptic cell.  Topology
    functions return the inputs which may connect to this cell.  This SDR
    contains a single true bit.
  * Argument 2: is the dimensions of the presynaptic cells.
  * Argument 3: is a random number generator to use for reproducible results.
  * Returns: an SDR containing all presynaptic cells which are allowed to
    connect to the postsynaptic cell.  The dimensions of this SDR must equal
    argument 2.)";

    init_Random(m);
    init_reals(m);
    init_Topology(m);
}
