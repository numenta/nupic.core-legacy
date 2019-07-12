/* ---------------------------------------------------------------------
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
 * --------------------------------------------------------------------- */

/** @file
 * PyBind11 bindings for Connections class
 */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <sstream>

#include <htm/algorithms/Connections.hpp>

namespace py = pybind11;
using namespace htm;

namespace htm_ext
{
  void init_Connections(py::module& m)
  {
    py::class_<Connections> py_Connections(m, "Connections",
R"(Compatibility Warning: This classes API is unstable and may change without warning.)");

    py_Connections.def(py::init<UInt, Permanence, bool>(),
        py::arg("numCells"),
        py::arg("connectedThreshold"),
        py::arg("timeseries") = false);

    py_Connections.def_property_readonly("connectedThreshold",
        [](const Connections &self) { return self.getConnectedThreshold(); });

    py_Connections.def("createSegment", &Connections::createSegment,
        py::arg("cell"),
	py::arg("maxSegmentsPerCell") = 0
	);

    py_Connections.def("destroySegment", &Connections::destroySegment);

    py_Connections.def("iteration", &Connections::iteration);

    py_Connections.def("createSynapse", &Connections::createSynapse,
        py::arg("segment"),
        py::arg("presynaticCell"),
        py::arg("permanence"));

    py_Connections.def("destroySynapse", &Connections::destroySynapse);

    py_Connections.def("updateSynapsePermanence", &Connections::updateSynapsePermanence,
        py::arg("synapse"),
        py::arg("permanence"));

    py_Connections.def("segmentsForCell", &Connections::segmentsForCell);

    py_Connections.def("synapsesForSegment", &Connections::synapsesForSegment);

    py_Connections.def("cellForSegment", &Connections::cellForSegment);

    py_Connections.def("idxOnCellForSegment", &Connections::idxOnCellForSegment);

    py_Connections.def("segmentForSynapse", &Connections::segmentForSynapse);

    py_Connections.def("permanenceForSynapse",
        [](Connections &self, Synapse idx) {
            auto &synData = self.dataForSynapse( idx );
            return synData.permanence; });

    py_Connections.def("presynapticCellForSynapse",
        [](Connections &self, Synapse idx) {
            auto &synData = self.dataForSynapse( idx );
            return synData.presynapticCell; });

    py_Connections.def("getSegment", &Connections::getSegment);

    py_Connections.def("segmentFlatListLength", &Connections::segmentFlatListLength);

    py_Connections.def("synapsesForPresynapticCell", &Connections::synapsesForPresynapticCell);

    py_Connections.def("reset", &Connections::reset);

    py_Connections.def("computeActivity",
        [](Connections &self, SDR &activePresynapticCells, bool learn=true) {
            // Allocate buffer to return & make a python destructor object for it.
            auto activeConnectedSynapses =
                new std::vector<SynapseIdx>( self.segmentFlatListLength(), 0u );
            auto destructor = py::capsule( activeConnectedSynapses,
                [](void *dataPtr) {
                    delete reinterpret_cast<std::vector<SynapseIdx>*>(dataPtr); });
            
	    // Call the C++ method.
            self.computeActivity(*activeConnectedSynapses, activePresynapticCells.getSparse(), learn);
            
	    // Wrap vector in numpy array.
            return py::array(activeConnectedSynapses->size(),
                             activeConnectedSynapses->data(),
                             destructor);
        },
R"(Returns numActiveConnectedSynapsesForSegment)");

    py_Connections.def("computeActivityFull",
        [](Connections &self, SDR &activePresynapticCells, bool learn=true) {
            // Allocate buffer to return & make a python destructor object for it.
            auto activeConnectedSynapses =
                new std::vector<SynapseIdx>( self.segmentFlatListLength(), 0u );
            auto connectedDestructor = py::capsule( activeConnectedSynapses,
                [](void *dataPtr) {
                    delete reinterpret_cast<std::vector<SynapseIdx>*>(dataPtr); });
            // Allocate buffer to return & make a python destructor object for it.
            auto activePotentialSynapses =
                new std::vector<SynapseIdx>( self.segmentFlatListLength(), 0u );
            auto potentialDestructor = py::capsule( activePotentialSynapses,
                [](void *dataPtr) {
                    delete reinterpret_cast<std::vector<SynapseIdx>*>(dataPtr); });
            
	    // Call the C++ method.
            self.computeActivity(*activeConnectedSynapses, 
			         *activePotentialSynapses,
                                 activePresynapticCells.getSparse(),
				 learn);

            // Wrap vector in numpy array.
            return py::make_tuple(
                    py::array(activeConnectedSynapses->size(),
                              activeConnectedSynapses->data(),
                              connectedDestructor),
                    py::array(activePotentialSynapses->size(),
                              activePotentialSynapses->data(),
                              potentialDestructor));
        },
R"(Returns pair of:
    numActiveConnectedSynapsesForSegment
    numActivePotentialSynapsesForSegment)");

    py_Connections.def("adaptSegment", &Connections::adaptSegment);

    py_Connections.def("raisePermanencesToThreshold", &Connections::raisePermanencesToThreshold);

    py_Connections.def("synapseCompetition", &Connections::synapseCompetition);

    py_Connections.def("bumpSegment", &Connections::bumpSegment);

    py_Connections.def("destroyMinPermanenceSynapses", &Connections::destroyMinPermanenceSynapses);

    py_Connections.def("numCells", &Connections::numCells);

    py_Connections.def("numSegments",
        [](Connections &self) { return self.numSegments(); });
    py_Connections.def("numSegments",
        [](Connections &self, CellIdx cell) { return self.numSegments(cell); });

    py_Connections.def("numSynapses",
        [](Connections &self) { return self.numSynapses(); });
    py_Connections.def("numSynapses",
        [](Connections &self, Segment seg) { return self.numSynapses(seg); });

    py_Connections.def("numConnectedSynapses",
        [](Connections &self, Segment seg) {
            auto &segData = self.dataForSegment( seg );
            return segData.numConnected; });

    py_Connections.def("__str__",
        [](Connections &self) {
            std::stringstream buf;
            buf << self;
            return buf.str(); });

    py_Connections.def(py::pickle(
        [](const Connections &self) {   // Save
            std::stringstream buf;
            self.save( buf );
            return py::bytes( buf.str() );
        },
        [](const py::bytes &data) {     // Load
            std::stringstream buf( data.cast<std::string>() );
            auto C = new Connections();
            C->load( buf );
            return C;
        } ));

    py_Connections.def("save",
        [](const Connections &self) {
            std::stringstream buf;
            self.save( buf );
            return py::bytes( buf.str() ); });

    py_Connections.def("load",
        [](const py::bytes &data) {
            std::stringstream buf( data.cast<std::string>() );
            auto C = new Connections();
            C->load( buf );
            return C; } );

  } // End function init_Connections
}   // End namespace htm_ext
