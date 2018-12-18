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
PyBind11 bindings for algorithms classes
*/



#include <bindings/suppress_register.hpp>  //must be before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>

#include <nupic/algorithms/Segment.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

namespace nupic_ext
{
    void init_algorithms(py::module& m)
    {

        m.def("getSegmentActivityLevel", [](py::list seg
            , py::array_t<nupic::Byte>& py_state
            , bool connectedSynapsesOnly
            , nupic::Real32 connectedPerm)
        {
            nupic::UInt32 activity = 0;

            auto state = get_it(py_state);
            auto stride0 = py_state.request().strides[0];

            // see algorithms.i[748]
            if (connectedSynapsesOnly)
            {
                for (py::size_t i = 0; i < seg.size(); ++i)
                {
                    py::list syn = seg[i];

                    nupic::Real32 p = syn[2].cast<nupic::Real32>();

                    if (p >= connectedPerm)
                    {
                        nupic::UInt32 c = syn[0].cast<nupic::UInt32>();
                        nupic::UInt32 j = syn[1].cast<nupic::UInt32>();

                        activity += state[c * stride0 + j];
                    }
                }
            }
            else
            {
                for (py::size_t i = 0; i < seg.size(); ++i)
                {
                    py::list syn = seg[i];

                    nupic::UInt32 c = syn[0].cast<nupic::UInt32>();
                    nupic::UInt32 j = syn[1].cast<nupic::UInt32>();

                    activity += state[c * stride0 + j];
                }
            }

            return activity;
        });

        m.def("isSegmentActive", [](py::list seg, py::array_t<nupic::Byte>& py_state,
            nupic::Real32 connectedPerm,
            nupic::UInt32 activationThreshold)
        {
            auto state = get_it(py_state);

            auto stride0 = py_state.request().strides[0];

            nupic::UInt32 activity = 0;

            nupic::UInt32 n = static_cast<nupic::UInt32>(seg.size());

            if (n < activationThreshold)
            {
                return false;
            }

            for (py::size_t i = 0; i < seg.size(); ++i)
            {
                py::list syn = seg[i];

                nupic::Real32 p = syn[2].cast<nupic::Real32>();
                if (p >= connectedPerm)
                {
                    nupic::UInt32 c = syn[0].cast<nupic::UInt32>();
                    nupic::UInt32 j = syn[1].cast<nupic::UInt32>();

                    activity += state[c * stride0 + j];
                    if (activity >= activationThreshold)
                    {
                        return true;
                    }
                }
            }

            return false;
        });


    }

} // namespace nupic_ext
