/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.
 *               2018, chhenning
 *               2019, David McDougall
 *
 * Unless you have an agreement with Numenta, Inc., for a separate license for
 * this software code, the following terms and conditions apply:
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
 * --------------------------------------------------------------------- */

/** @file
 * PyBind11 bindings for HTM class
 */

#include <bindings/suppress_register.hpp>  // include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/algorithms/TemporalMemory.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;
using namespace htm;

namespace htm_ext
{
    void init_TemporalMemory(py::module& m)
    {
        typedef TemporalMemory HTM_t;

        py::class_<HTM_t> py_HTM(m, "TemporalMemory",
R"(Temporal Memory implementation in C++.

Example usage:
    TODO
)");

        py_HTM.def(py::init<>());
        py_HTM.def(py::init<std::vector<CellIdx>
                , CellIdx
                , SynapseIdx
                , Permanence
                , Permanence
                , SynapseIdx
                , SynapseIdx
                , Permanence
                , Permanence
                , Permanence
                , Int
                , SegmentIdx
                , SynapseIdx
                , bool
                , UInt>(),
R"(Initialize the temporal memory (TM) using the given parameters.

Argument columnDimensions
    Dimensions of the mini-column space

Argument cellsPerColumn
   Number of cells per mini-column

Argument activationThreshold
    If the number of active connected synapses on a segment is at least
    this threshold, the segment is actived.

Argument initialPermanence
    Initial permanence of a new synapse.

Argument connectedPermanence
    If the permanence value for a synapse is greater than this value, then it
    is connected.

Argument minThreshold
    If the number of potential synapses active on a segment is at least
    this threshold, it is said to be "matching" and is eligible for
    learning.

Argument maxNewSynapseCount
    The maximum number of synapses added to a segment during learning.

Argument permanenceIncrement
    Amount by which permanences of synapses are incremented during learning.

Argument permanenceDecrement
    Amount by which permanences of synapses are decremented during learning.

Argument predictedSegmentDecrement
    Amount by which segments are punished for incorrect predictions.
    A good value is just a bit larger than (the column-level sparsity *
    permanenceIncrement). So, if column-level sparsity is 2% and
    permanenceIncrement is 0.01, this parameter should be something like 4% *
    0.01 = 0.0004

Argument seed
    Seed for the random number generator.

Argument maxSegmentsPerCell
    The maximum number of segments per cell.

Argument maxSynapsesPerSegment
    The maximum number of synapses per segment.

Argument checkInputs
    Whether to check that the activeColumns are sorted without
    duplicates. Disable this for a small speed boost.

Argument externalPredictiveInputs
    Number of external predictive inputs.  These values are not related to this
    TM, they represent input from a different region.  This TM will form
    synapses with these inputs in addition to the cells which are part of this
    TemporalMemory.  If this is given (and greater than 0) then the active
    cells and winner cells of these external inputs must be given to methods
    TM.compute and TM.activateDendrites
)"
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
                , py::arg("externalPredictiveInputs") = 0u
            );

        py_HTM.def("printParameters",
            [](const HTM_t& self)
                { self.printParameters( std::cout ); },
            py::call_guard<py::scoped_ostream_redirect,
                           py::scoped_estream_redirect>());

        // pickle
        // https://github.com/pybind/pybind11/issues/1061
        py_HTM.def(py::pickle(
            [](const HTM_t& self) -> std::string
        {
            // __getstate__
            std::ostringstream os;

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


        py_HTM.def("activateCells", [](HTM_t& self, const SDR& activeColumns, bool learn)
        {
            self.activateCells(activeColumns, learn);
        },
R"(Calculate the active cells, using the current active columns and
dendrite segments.  Grow and reinforce synapses.)"
            , py::arg("activeColumns"), py::arg("learn") = true);

        py_HTM.def("compute", [](HTM_t& self, const SDR &activeColumns, bool learn)
            { self.compute(activeColumns, learn); },
                py::arg("activeColumns"),
                py::arg("learn") = true);

        py_HTM.def("compute", [](HTM_t& self, const SDR &activeColumns, bool learn,
                                 const SDR &externalPredictiveInputsActive, const SDR &externalPredictiveInputsWinners)
            { self.compute(activeColumns, learn, externalPredictiveInputsActive, externalPredictiveInputsWinners); },
R"(Perform one time step of the Temporal Memory algorithm.

This method calls activateDendrites, then calls activateCells. Using
the TemporalMemory via its compute method ensures that you'll always
be able to call getActiveCells at the end of the time step.

Argument activeColumns
    SDR of active mini-columns.

Argument learn
    Whether or not learning is enabled.

Argument externalPredictiveInputsActive
    (optional) SDR of active external predictive inputs.  
    TM must be set up with the 'externalPredictiveInputs' constructor parameter for this use.

Argument externalPredictiveInputsWinners
    (optional) SDR of winning external predictive inputs.  When learning, only these
    inputs are considered active.  
    externalPredictiveInputsWinners must be a subset of externalPredictiveInputsActive.
)",
                py::arg("activeColumns"),
                py::arg("learn") = true,
                py::arg("externalPredictiveInputsActive"),
                py::arg("externalPredictiveInputsWinners"));

        py_HTM.def("reset", &HTM_t::reset,
R"(Indicates the start of a new sequence.
Resets sequence state of the TM.)");

        py_HTM.def("getActiveCells", [](const HTM_t& self)
        {
            auto dims = self.getColumnDimensions();
            dims.push_back( static_cast<UInt32>(self.getCellsPerColumn()) );
            SDR *cells = new SDR( dims );
            self.getActiveCells(*cells);
            return cells;
        });

        py_HTM.def("activateDendrites", [](HTM_t &self, bool learn) {
            SDR externalPredictiveInputs({ self.externalPredictiveInputs });
            self.activateDendrites(learn, externalPredictiveInputs, externalPredictiveInputs);
        },
            py::arg("learn"));

        py_HTM.def("activateDendrites",
            [](HTM_t &self, bool learn,const SDR &externalPredictiveInputsActive, const SDR &externalPredictiveInputsWinners)
                { self.activateDendrites(learn, externalPredictiveInputsActive, externalPredictiveInputsWinners); },
R"(Calculate dendrite segment activity, using the current active cells.  Call
this method before calling getPredictiveCells, getActiveSegments, or
getMatchingSegments.  In each time step, only the first call to this
method has an effect, subsequent calls assume that the prior results are
still valid.

Argument learn
    If true, segment activations will be recorded. This information is
    used during segment cleanup.

Argument externalPredictiveInputsActive
    (optional) SDR of active external predictive inputs.

Argument externalPredictiveInputsWinners
    (optional) SDR of winning external predictive inputs.  When learning, only
    these inputs are considered active.
    externalPredictiveInputsWinners must be a subset of externalPredictiveInputsActive.

See TM.compute() for details of the parameters.)",
            py::arg("learn"),
            py::arg("externalPredictiveInputsActive"),
            py::arg("externalPredictiveInputsWinners"));

        py_HTM.def("getPredictiveCells", [](const HTM_t& self)
            { return self.getPredictiveCells();},
R"()");

        py_HTM.def("getWinnerCells", [](const HTM_t& self)
        {
            auto dims = self.getColumnDimensions();
            dims.push_back( static_cast<UInt32>(self.getCellsPerColumn()) );
            SDR *winnerCells = new SDR( dims );
            self.getWinnerCells(*winnerCells);
            return winnerCells;
        },
R"()");

        py_HTM.def("getActiveSegments", [](const HTM_t& self)
            { return self.getActiveSegments(); },
R"()");

        py_HTM.def("getMatchingSegments", [](const HTM_t& self)
            { return self.getMatchingSegments(); },
R"()");

        py_HTM.def("cellsForColumn", [](HTM_t& self, UInt columnIdx)
        {
            auto cells = self.cellsForColumn(columnIdx);

            return py::array_t<htm::UInt32>(cells.size(), cells.data());
        },
R"(Returns list of indices of cells that belong to a mini-column.

Argument column is sparse index of a mini-column.)");

        py_HTM.def("columnForCell", &HTM_t::columnForCell,
R"(Returns the index of the mini-column that a cell belongs to.

Argument (int) cell index
Returns (int) mini-column index)");

        py_HTM.def("createSegment", &HTM_t::createSegment,
R"(Create a segment on the specified cell. This method calls
createSegment on the underlying connections, and it does some extra
bookkeeping. Unit tests should call this method, and not
connections.createSegment().

Argument cell
    Index of Cell to add a segment to.

Returns the created segment (index handle).)");

        py_HTM.def("numberOfCells",   &HTM_t::numberOfCells,
R"(Returns the number of cells in this TemporalMemory.)");

        py_HTM.def("numberOfColumns", &HTM_t::numberOfColumns,
R"(Returns the total number of mini-columns.)");

        py_HTM.def_property_readonly("connections", [](const HTM_t &self)
            { return self.connections; },
R"(Internal Connections object. Danger!
Modifying this may detrimentally effect the TM.
The Connections class API is subject to change.)");

        py_HTM.def_property_readonly("externalPredictiveInputs", [](const HTM_t &self)
            { return self.externalPredictiveInputs; },
R"()");

        py_HTM.def_property_readonly("anomaly", [](const HTM_t &self) { return self.anomaly; },
          "Anomaly score updated with each TM::compute() call. "
        );

        py_HTM.def("__str__",
            [](HTM_t &self) {
                std::stringstream buf;
                buf << self;
                return buf.str(); });
    }

} // namespace htm_ext
