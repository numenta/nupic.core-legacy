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
PyBind11 bindings for SpatialPooler class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/algorithms/SpatialPooler.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;
using namespace nupic;
using namespace nupic::algorithms::spatial_pooler;

namespace nupic_ext
{
    void init_Spatial_Pooler(py::module& m)
    {
        py::class_<SpatialPooler> py_SpatialPooler(m, "SpatialPooler");

        py_SpatialPooler.def(
            py::init<vector<UInt>
            , vector<UInt>
            , UInt
            , Real
            , bool
            , Real
            , UInt
            , UInt
            , Real
            , Real
            , Real
            , Real
            , UInt
            , Real
            , Int
            , UInt
            , bool>()
            , py::arg("inputDimensions") = vector<UInt>({ 32, 32 })
            , py::arg("columnDimensions") = vector<UInt>({ 64, 64 })
            , py::arg("potentialRadius") = 16
            , py::arg("potentialPct") = 0.5
            , py::arg("globalInhibition") = false
            , py::arg("localAreaDensity") = -1.0
            , py::arg("numActiveColumnsPerInhArea") = 10
            , py::arg("stimulusThreshold") = 0
            , py::arg("synPermInactiveDec") = 0.01
            , py::arg("synPermActiveInc") = 0.1
            , py::arg("synPermConnected") = 0.1
            , py::arg("minPctOverlapDutyCycle") = 0.001
            , py::arg("dutyCyclePeriod") = 1000
            , py::arg("boostStrength") = 0.0
            , py::arg("seed") = -1
            , py::arg("spVerbosity") = 0
            , py::arg("wrapAround") = true
        );

        py_SpatialPooler.def("initialize", &SpatialPooler::initialize
            , py::arg("inputDimensions") = vector<UInt>({ 32, 32 })
            , py::arg("columnDimensions") = vector<UInt>({ 64, 64 })
            , py::arg("potentialRadius") = 16
            , py::arg("potentialPct") = 0.5
            , py::arg("globalInhibition") = false
            , py::arg("localAreaDensity") = -1.0
            , py::arg("numActiveColumnsPerInhArea") = 10
            , py::arg("stimulusThreshold") = 0
            , py::arg("synPermInactiveDec") = 0.01
            , py::arg("synPermActiveInc") = 0.1
            , py::arg("synPermConnected") = 0.1
            , py::arg("minPctOverlapDutyCycle") = 0.001
            , py::arg("dutyCyclePeriod") = 1000
            , py::arg("boostStrength") = 0.0
            , py::arg("seed") = -1
            , py::arg("spVerbosity") = 0
            , py::arg("wrapAround") = true);


        py_SpatialPooler.def("getColumnDimensions", &SpatialPooler::getColumnDimensions);
        py_SpatialPooler.def("getInputDimensions", &SpatialPooler::getInputDimensions);
        py_SpatialPooler.def("getNumColumns", &SpatialPooler::getNumColumns);
        py_SpatialPooler.def("getNumInputs", &SpatialPooler::getNumInputs);
        py_SpatialPooler.def("getPotentialRadius", &SpatialPooler::getPotentialRadius);
        py_SpatialPooler.def("setPotentialRadius", &SpatialPooler::setPotentialRadius);
        py_SpatialPooler.def("getPotentialPct", &SpatialPooler::getPotentialPct);
        py_SpatialPooler.def("setPotentialPct", &SpatialPooler::setPotentialPct);
        py_SpatialPooler.def("getGlobalInhibition", &SpatialPooler::getGlobalInhibition);
        py_SpatialPooler.def("setGlobalInhibition", &SpatialPooler::setGlobalInhibition);


        py_SpatialPooler.def("getNumActiveColumnsPerInhArea", &SpatialPooler::getNumActiveColumnsPerInhArea);
        py_SpatialPooler.def("setNumActiveColumnsPerInhArea", &SpatialPooler::setNumActiveColumnsPerInhArea);
        py_SpatialPooler.def("getLocalAreaDensity", &SpatialPooler::getLocalAreaDensity);
        py_SpatialPooler.def("setLocalAreaDensity", &SpatialPooler::setLocalAreaDensity);
        py_SpatialPooler.def("getStimulusThreshold", &SpatialPooler::getStimulusThreshold);
        py_SpatialPooler.def("setStimulusThreshold", &SpatialPooler::setStimulusThreshold);
        py_SpatialPooler.def("getInhibitionRadius", &SpatialPooler::getInhibitionRadius);
        py_SpatialPooler.def("setInhibitionRadius", &SpatialPooler::setInhibitionRadius);
        py_SpatialPooler.def("getDutyCyclePeriod", &SpatialPooler::getDutyCyclePeriod);
        py_SpatialPooler.def("setDutyCyclePeriod", &SpatialPooler::setDutyCyclePeriod);
        py_SpatialPooler.def("getBoostStrength", &SpatialPooler::getBoostStrength);
        py_SpatialPooler.def("setBoostStrength", &SpatialPooler::setBoostStrength);
        py_SpatialPooler.def("getIterationNum", &SpatialPooler::getIterationNum);
        py_SpatialPooler.def("setIterationNum", &SpatialPooler::setIterationNum);
        py_SpatialPooler.def("getIterationLearnNum", &SpatialPooler::getIterationLearnNum);
        py_SpatialPooler.def("setIterationLearnNum", &SpatialPooler::setIterationLearnNum);
        py_SpatialPooler.def("getSpVerbosity", &SpatialPooler::getSpVerbosity);
        py_SpatialPooler.def("setSpVerbosity", &SpatialPooler::setSpVerbosity);
        py_SpatialPooler.def("getWrapAround", &SpatialPooler::getWrapAround);
        py_SpatialPooler.def("setWrapAround", &SpatialPooler::setWrapAround);
        py_SpatialPooler.def("getUpdatePeriod", &SpatialPooler::getUpdatePeriod);
        py_SpatialPooler.def("setUpdatePeriod", &SpatialPooler::setUpdatePeriod);
        py_SpatialPooler.def("getSynPermActiveInc", &SpatialPooler::getSynPermActiveInc);
        py_SpatialPooler.def("setSynPermActiveInc", &SpatialPooler::setSynPermActiveInc);
        py_SpatialPooler.def("getSynPermInactiveDec", &SpatialPooler::getSynPermInactiveDec);
        py_SpatialPooler.def("setSynPermInactiveDec", &SpatialPooler::setSynPermInactiveDec);
        py_SpatialPooler.def("getSynPermBelowStimulusInc", &SpatialPooler::getSynPermBelowStimulusInc);
        py_SpatialPooler.def("setSynPermBelowStimulusInc", &SpatialPooler::setSynPermBelowStimulusInc);
        py_SpatialPooler.def("getSynPermConnected", &SpatialPooler::getSynPermConnected);
        py_SpatialPooler.def("setSynPermConnected", &SpatialPooler::setSynPermConnected);
        py_SpatialPooler.def("getSynPermMax", &SpatialPooler::getSynPermMax);
        py_SpatialPooler.def("getMinPctOverlapDutyCycles", &SpatialPooler::getMinPctOverlapDutyCycles);
        py_SpatialPooler.def("setMinPctOverlapDutyCycles", &SpatialPooler::setMinPctOverlapDutyCycles);

        // loadFromString
        py_SpatialPooler.def("loadFromString", [](SpatialPooler& self, const std::string& inString)
        {
            std::istringstream inStream(inString);
            self.load(inStream);
        });

        // writeToString
        py_SpatialPooler.def("writeToString", [](const SpatialPooler& self)
        {
            std::ostringstream os;
            os.flags(ios::scientific);
            os.precision(numeric_limits<double>::digits10 + 1);

            self.save(os);

            return os.str();
        });

        // compute
        py_SpatialPooler.def("compute", [](SpatialPooler& self, py::array& x, bool learn, py::array& y)
        {
            if (py::isinstance<py::array_t<std::uint32_t>>(x) == false)
            {
                throw runtime_error("Incompatible format. Expect uint32");
            }

            if (py::isinstance<py::array_t<std::uint32_t>>(y) == false)
            {
                throw runtime_error("Incompatible format. Expect uint32");
            }

            self.compute(get_it<UInt>(x), learn, get_it<UInt>(y));
        });

        // stripUnlearnedColumns
        py_SpatialPooler.def("stripUnlearnedColumns", [](SpatialPooler& self, py::array_t<UInt>& x)
        {
            self.stripUnlearnedColumns(get_it(x));
        });

        // setBoostFactors
        py_SpatialPooler.def("setBoostFactors", [](SpatialPooler& self, py::array_t<Real>& x)
        {
            self.setBoostFactors(get_it(x));
        });

        // getBoostFactors
        py_SpatialPooler.def("getBoostFactors", [](const SpatialPooler& self, py::array_t<Real>& x)
        {
            self.getBoostFactors(get_it(x));
        });

        // setOverlapDutyCycles
        py_SpatialPooler.def("setOverlapDutyCycles", [](SpatialPooler& self, py::array_t<Real>& x)
        {
            self.setOverlapDutyCycles(get_it(x));
        });

        // getOverlapDutyCycles
        py_SpatialPooler.def("getOverlapDutyCycles", [](const SpatialPooler& self, py::array_t<Real>& x)
        {
            self.getOverlapDutyCycles(get_it(x));
        });

        // setActiveDutyCycles
        py_SpatialPooler.def("setActiveDutyCycles", [](SpatialPooler& self, py::array_t<Real>& x)
        {
            self.setActiveDutyCycles(get_it(x));
        });

        // getActiveDutyCycles
        py_SpatialPooler.def("getActiveDutyCycles", [](const SpatialPooler& self, py::array_t<Real>& x)
        {
            self.getActiveDutyCycles(get_it(x));
        });

        // setMinOverlapDutyCycles
        py_SpatialPooler.def("setMinOverlapDutyCycles", [](SpatialPooler& self, py::array_t<Real>& x)
        {
            self.setMinOverlapDutyCycles(get_it(x));
        });

        // getMinOverlapDutyCycles
        py_SpatialPooler.def("getMinOverlapDutyCycles", [](const SpatialPooler& self, py::array_t<Real>& x)
        {
            self.getMinOverlapDutyCycles(get_it(x));
        });

        // setPotential
        py_SpatialPooler.def("setPotential", [](SpatialPooler& self, UInt column, py::array_t<UInt>& x)
        {
            self.setPotential(column, get_it(x));
        });

        // getPotential
        py_SpatialPooler.def("getPotential", [](const SpatialPooler& self, UInt column, py::array_t<UInt>& x)
        {
            self.getPotential(column, get_it(x));
        });

        // setPermanence
        py_SpatialPooler.def("setPermanence", [](SpatialPooler& self, UInt column, py::array_t<Real>& x)
        {
            self.setPermanence(column, get_it(x));
        });

        // getPermanence
        py_SpatialPooler.def("getPermanence", [](const SpatialPooler& self, UInt column, py::array_t<Real>& x)
        {
            self.getPermanence(column, get_it(x));
        });

        // getConnectedSynapses
        py_SpatialPooler.def("getConnectedSynapses", [](const SpatialPooler& self, UInt column, py::array_t<UInt>& x)
        {
            self.getConnectedSynapses(column, get_it(x));
        });

        // getConnectedCounts
        py_SpatialPooler.def("getConnectedCounts", [](const SpatialPooler& self, py::array_t<UInt>& x)
        {
            self.getConnectedCounts(get_it(x));
        });

        // getOverlaps
        py_SpatialPooler.def("getOverlaps", [](SpatialPooler& self)
        {
            auto overlaps = self.getOverlaps();

            return py::array_t<UInt>( overlaps.size(), overlaps.data());
        });

        // getBoostedOverlaps
        py_SpatialPooler.def("getBoostedOverlaps", [](SpatialPooler& self)
        {
            auto overlaps = self.getBoostedOverlaps();

            return py::array_t<Real>( overlaps.size(), overlaps.data());
        });


        ////////////////////
        // inhibitColumns

        auto inhibitColumns_func = [](SpatialPooler& self, py::array_t<Real>& overlaps)
        {
            std::vector<nupic::Real> overlapsVector(get_it(overlaps), get_end(overlaps));

            std::vector<nupic::UInt> activeColumnsVector;

            self.inhibitColumns_(overlapsVector, activeColumnsVector);

            return py::array_t<UInt>( activeColumnsVector.size(), activeColumnsVector.data());
        };

        py_SpatialPooler.def("_inhibitColumns", inhibitColumns_func);
        py_SpatialPooler.def("inhibitColumns_", inhibitColumns_func);


        //////////////////////
        // getIterationLearnNum
        py_SpatialPooler.def("getIterationLearnNum", &SpatialPooler::getIterationLearnNum);


        // pickle

        py_SpatialPooler.def(py::pickle(
            [](const SpatialPooler& sp)
        {
            std::stringstream ss;

            sp.save(ss);

            return ss.str();
        },
            [](std::string& s)
        {
            std::istringstream ss(s);
            SpatialPooler sp;
            sp.load(ss);

            return sp;
        }));




    }
} // namespace nupic_ext
