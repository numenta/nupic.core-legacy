/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
 *
 * Author: @chhenning, 2018
 * ---------------------------------------------------------------------
 */

/** @file
PyBind11 bindings for SparseTensor class
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseTensor.hpp>

#include "SparseTensorIndex.hpp"
#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;
using namespace nupic;

namespace nupic_ext
{
    void init_TensorIndex(py::module& m)
    {
        py::class_<PyBindTensorIndex> py_TensorIndex(m, "TensorIndex");

    }

    void init_SparseTensor(py::module& m)
    {
        typedef nupic::SparseTensor<PyBindTensorIndex, nupic::Real> Tensor_t;

        py::class_<Tensor_t> py_SparseTensor(m, "SparseTensor");

        py_SparseTensor.def(py::init([](const std::string& state)
        {
            size_t rank = 0;
            {
                std::stringstream forRank(state);
                forRank.exceptions(std::ios::failbit | std::ios::badbit);
                forRank >> rank;
            };

            PyBindTensorIndex index(rank, (const size_t *)0);
            for (size_t i = 0; i<rank; ++i) {
                index[i] = 1;
            }
            Tensor_t tensor(index);
            std::stringstream toRead(state);
            tensor.fromStream(toRead);

            return tensor;
        }));
    }

} // namespace nupic_ext
