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
PyBind11 bindings for HTM class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/algorithms/TemporalMemory.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;
using namespace nupic;

namespace nupic_ext
{
    void init_HTM(py::module& m)
    {
        typedef nupic::algorithms::temporal_memory::TemporalMemory HTM_t;

        py::class_<HTM_t> py_HTM(m, "TemporalMemory");

        py_HTM.def(py::init<>())
            .def(py::init<std::vector<UInt>
                , UInt, UInt
                , Permanence, Permanence
                , UInt, UInt
                , Permanence, Permanence, Permanence
                , Int
                , UInt, UInt
                , bool>()
                , py::arg("columnDimensions")
                , py::arg("cellsPerColumn") = 32
                , py::arg("activationThreshold") = 13
                , py::arg("initialPermanence") = 0.21
                , py::arg("connectedPermanence") = 0.5
                , py::arg("minThreshold") = 10
                , py::arg("maxNewSynapseCount") = 20
                , py::arg("permanenceIncrement") = 0.1
                , py::arg("permanenceDecrement") = 0.1
                , py::arg("predictedSegmentDecrement") = 0.0
                , py::arg("seed") = 42
                , py::arg("maxSegmentsPerCell") = 255
                , py::arg("maxSynapsesPerSegment") = 255
                , py::arg("checkInputs") = true
            );


        // pickle
        // https://github.com/pybind/pybind11/issues/1061
        py_HTM.def(py::pickle(
            [](const HTM_t& self) -> std::string
        {
            // __getstate__
            std::ostringstream os;
            //s << self.persistentSize();

            os.flags(std::ios::scientific);
            os.precision(std::numeric_limits<double>::digits10 + 1);

            self.save(os);

            return os.str();
        },
            [](const std::string& str) -> HTM_t
        {
            // __setstate__
            if (str.empty())
            {
                throw std::runtime_error("Empty state");
            }

            std::istringstream is(str);

            HTM_t htm;
            htm.load(is);

            return htm;
        }
        ));


        py_HTM.def("activateCells", [](HTM_t& self, py::array_t<nupic::UInt32>& activeColumns, bool learn)
        {
            self.activateCells(activeColumns.size(), get_it(activeColumns), learn);
        }, "Calculate the active cells, using the current active columns and dendrite segments.Grow and reinforce synapses."
            , py::arg("activeColumns"), py::arg("learn") = true);

        py_HTM.def("compute", [](HTM_t& self, py::array_t<nupic::UInt32>& activeColumns, bool learn)
        {
            self.compute(activeColumns.size(), get_it(activeColumns), learn);
        }, "Perform one time step of the Temporal Memory algorithm."
            , py::arg("activeColumns"), py::arg("learn") = true);

        py_HTM.def("getActiveCells", [](const HTM_t& self)
        {
            auto activeCells = self.getActiveCells();

            return py::array_t<nupic::UInt32>(activeCells.size(), activeCells.data());
        });

        py_HTM.def("getPredictiveCells", [](const HTM_t& self)
        {
            auto predictiveCells = self.getPredictiveCells();

            return py::array_t<nupic::UInt32>(predictiveCells.size(), predictiveCells.data());
        });

        py_HTM.def("getWinnerCells", [](const HTM_t& self)
        {
            auto winnerCells = self.getWinnerCells();

            return py::array_t<nupic::UInt32>(winnerCells.size(), winnerCells.data());
        });

        py_HTM.def("getActiveSegments", [](const HTM_t& self)
        {
            auto activeSegments = self.getActiveSegments();

            return py::array_t<nupic::UInt32>(activeSegments.size(), activeSegments.data());
        });

        py_HTM.def("getMatchingSegments", [](const HTM_t& self)
        {
            auto matchingSegments = self.getMatchingSegments();

            return py::array_t<nupic::UInt32>(matchingSegments.size(), matchingSegments.data());
        });

        py_HTM.def("cellsForColumn", [](HTM_t& self, UInt columnIdx)
        {
            auto cells = self.cellsForColumn(columnIdx);

            return py::array_t<nupic::UInt32>(cells.size(), cells.data());
        });

        py_HTM.def("convertedActivateCells", [](HTM_t& self, py::array_t<nupic::UInt32>& activeColumns, bool learn)
        {
            self.activateCells(activeColumns.size(), get_it(activeColumns), learn);
        }, ""
            , py::arg("activeColumns"), py::arg("learn") = true);

        py_HTM.def("convertedCompute", [](HTM_t& self, py::array_t<nupic::UInt32>& activeColumns, bool learn)
        {
            self.compute(activeColumns.size(), get_it(activeColumns), learn);
        }, "", py::arg("activeColumns"), py::arg("learn") = true);

    }

} // namespace nupic_ext
