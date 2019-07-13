/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2018, Numenta, Inc.
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
 * Author: @chhenning, 2018
 * --------------------------------------------------------------------- */

/** @file
PyBind11 bindings for SpatialPooler class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/types/Sdr.hpp>

#include "bindings/engine/py_utils.hpp"


namespace htm_ext
{
namespace py = pybind11;
using namespace htm;

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
            , Real
            , Real
            , Real
            , Real
            , UInt
            , Real
            , Int
            , UInt
            , bool>()
            , py::call_guard<py::scoped_ostream_redirect,
                             py::scoped_estream_redirect>(),
R"(
Argument inputDimensions A list of integers representing the
        dimensions of the input vector. Format is [height, width,
        depth, ...], where each value represents the size of the
        dimension. For a topology of one dimensions with 100 inputs
        use [100]. For a two dimensional topology of 10x5
        use [10,5].

Argument columnDimensions A list of integers representing the
        dimensions of the columns in the region. Format is [height,
        width, depth, ...], where each value represents the size of
        the dimension. For a topology of one dimensions with 2000
        columns use 2000, or [2000]. For a three dimensional
        topology of 32x64x16 use [32, 64, 16].

Argument potentialRadius This parameter determines the extent of the
        input that each column can potentially be connected to. This
        can be thought of as the input bits that are visible to each
        column, or a 'receptive field' of the field of vision. A large
        enough value will result in global coverage, meaning
        that each column can potentially be connected to every input
        bit. This parameter defines a square (or hyper square) area: a
        column will have a max square potential pool with sides of
        length (2 * potentialRadius + 1).

Argument potentialPct The percent of the inputs, within a column's
        potential radius, that a column can be connected to. If set to
        1, the column will be connected to every input within its
        potential radius. This parameter is used to give each column a
        unique potential pool when a large potentialRadius causes
        overlap between the columns. At initialization time we choose
        ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
        input bits to comprise the column's potential pool.

Argument globalInhibition If true, then during inhibition phase the
        winning columns are selected as the most active columns from the
        region as a whole. Otherwise, the winning columns are selected
        with respect to their local neighborhoods. Global inhibition
        boosts performance significantly but there is no topology at the
        output.

Argument localAreaDensity The desired density of active columns within
        a local inhibition area (the size of which is set by the
        internally calculated inhibitionRadius, which is in turn
        determined from the average size of the connected potential
        pools of all columns). The inhibition logic will insure that at
        most N columns remain ON within a local inhibition area, where
        N = localAreaDensity * (total number of columns in inhibition
        area). 

Argument stimulusThreshold This is a number specifying the minimum
        number of synapses that must be active in order for a column to
        turn ON. The purpose of this is to prevent noisy input from
        activating columns.

Argument synPermInactiveDec The amount by which the permanence of an
        inactive synapse is decremented in each learning step.

Argument synPermActiveInc The amount by which the permanence of an
        active synapse is incremented in each round.

Argument synPermConnected The default connected threshold. Any synapse
        whose permanence value is above the connected threshold is
        a "connected synapse", meaning it can contribute to
        the cell's firing.

Argument minPctOverlapDutyCycle A number between 0 and 1.0, used to set
        a floor on how often a column should have at least
        stimulusThreshold active inputs. Periodically, each column looks
        at the overlap duty cycle of all other column within its
        inhibition radius and sets its own internal minimal acceptable
        duty cycle to: minPctDutyCycleBeforeInh * max(other columns'
        duty cycles). On each iteration, any column whose overlap duty
        cycle falls below this computed value will get all of its
        permanence values boosted up by synPermActiveInc. Raising all
        permanences in response to a sub-par duty cycle before
        inhibition allows a cell to search for new inputs when either
        its previously learned inputs are no longer ever active, or when
        the vast majority of them have been "hijacked" by other columns.

Argument dutyCyclePeriod The period used to calculate duty cycles.
        Higher values make it take longer to respond to changes in
        boost. Shorter values make it potentially more unstable and
        likely to oscillate.

Argument boostStrength A number greater or equal than 0, used to
        control boosting strength. No boosting is applied if it is set to 0.
        The strength of boosting increases as a function of boostStrength.
        Boosting encourages columns to have similar activeDutyCycles as their
        neighbors, which will lead to more efficient use of columns. However,
        too much boosting may also lead to instability of SP outputs.


Argument seed Seed for our random number generator. If seed is < 0
        a randomly generated seed is used. The behavior of the spatial
        pooler is deterministic once the seed is set.

Argument spVerbosity spVerbosity level: 0, 1, 2, or 3

Argument wrapAround boolean value that determines whether or not inputs
        at the beginning and end of an input dimension are considered
        neighbors for the purpose of mapping inputs to columns.
)"
            , py::arg("inputDimensions") = vector<UInt>({ 32, 32 })
            , py::arg("columnDimensions") = vector<UInt>({ 64, 64 })
            , py::arg("potentialRadius") = 16
            , py::arg("potentialPct") = 0.5
            , py::arg("globalInhibition") = false
            , py::arg("localAreaDensity") = 0.02f
            , py::arg("stimulusThreshold") = 0
            , py::arg("synPermInactiveDec") = 0.01
            , py::arg("synPermActiveInc") = 0.1
            , py::arg("synPermConnected") = 0.1
            , py::arg("minPctOverlapDutyCycle") = 0.001
            , py::arg("dutyCyclePeriod") = 1000
            , py::arg("boostStrength") = 0.0
            , py::arg("seed") = 1
            , py::arg("spVerbosity") = 0
            , py::arg("wrapAround") = true
        );

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
        py_SpatialPooler.def("getSynPermMax", &SpatialPooler::getSynPermMax);
        py_SpatialPooler.def("getMinPctOverlapDutyCycles", &SpatialPooler::getMinPctOverlapDutyCycles);
        py_SpatialPooler.def("setMinPctOverlapDutyCycles", &SpatialPooler::setMinPctOverlapDutyCycles);

        // loadFromString
        py_SpatialPooler.def("loadFromString", [](SpatialPooler& self, const py::bytes& inString)
        {
            std::stringstream inStream(inString.cast<std::string>());
            self.load(inStream);
        });

        // writeToString
        py_SpatialPooler.def("writeToString", [](const SpatialPooler& self)
        {
            std::ostringstream os;
            os.flags(ios::scientific);
            os.precision(numeric_limits<double>::digits10 + 1);

            self.save(os);

            return py::bytes( os.str() );
        });

        // compute
        py_SpatialPooler.def("compute", [](SpatialPooler& self, SDR& input, bool learn, SDR& output)
            { self.compute( input, learn, output ); },
R"(
This is the main workhorse method of the SpatialPooler class. This method
takes an input SDR and computes the set of output active columns. If 'learn' is
set to True, this method also performs learning.

Argument input An SDR that comprises the input to the spatial pooler.  The size
        of the SDR must match total number of input bits implied by the
        constructor (also returned by the method getNumInputs).

Argument learn A boolean value indicating whether learning should be
        performed. Learning entails updating the permanence values of
        the synapses, duty cycles, etc. Learning is typically on but
        setting learning to 'off' is useful for analyzing the current
        state of the SP. For example, you might want to feed in various
        inputs and examine the resulting SDR's. Note that if learning
        is off, boosting is turned off and columns that have never won
        will be removed from activeVector.  TODO: we may want to keep
        boosting on even when learning is off.

Argument output An SDR representing the winning columns after
        inhibition. The size of the SDR is equal to the number of
        columns (also returned by the method getNumColumns).
)",
        py::arg("input"),
        py::arg("learn") = true,
        py::arg("output")
        ); 

        // setBoostFactors
        py_SpatialPooler.def("setBoostFactors", [](SpatialPooler& self, py::array& x)
        {
            self.setBoostFactors(get_it<Real>(x));
        });

        // getBoostFactors
        py_SpatialPooler.def("getBoostFactors", [](const SpatialPooler& self, py::array& x)
        {
            self.getBoostFactors(get_it<Real>(x));
        });

        // setOverlapDutyCycles
        py_SpatialPooler.def("setOverlapDutyCycles", [](SpatialPooler& self, py::array& x)
        {
            self.setOverlapDutyCycles(get_it<Real>(x));
        });

        // getOverlapDutyCycles
        py_SpatialPooler.def("getOverlapDutyCycles", [](const SpatialPooler& self, py::array& x)
        {
            self.getOverlapDutyCycles(get_it<Real>(x));
        });

        // setActiveDutyCycles
        py_SpatialPooler.def("setActiveDutyCycles", [](SpatialPooler& self, py::array& x)
        {
            self.setActiveDutyCycles(get_it<Real>(x));
        });

        // getActiveDutyCycles
        py_SpatialPooler.def("getActiveDutyCycles", [](const SpatialPooler& self, py::array& x)
        {
            self.getActiveDutyCycles(get_it<Real>(x));
        });

        // setMinOverlapDutyCycles
        py_SpatialPooler.def("setMinOverlapDutyCycles", [](SpatialPooler& self, py::array& x)
        {
            self.setMinOverlapDutyCycles(get_it<Real>(x));
        });

        // getMinOverlapDutyCycles
        py_SpatialPooler.def("getMinOverlapDutyCycles", [](const SpatialPooler& self, py::array& x)
        {
            self.getMinOverlapDutyCycles(get_it<Real>(x));
        });

        // setPotential
        py_SpatialPooler.def("setPotential", [](SpatialPooler& self, UInt column, py::array& x)
        {
            self.setPotential(column, get_it<UInt>(x));
        });

        // getPotential
        py_SpatialPooler.def("getPotential", [](const SpatialPooler& self, UInt column, py::array& x)
        {
            self.getPotential(column, get_it<UInt>(x));
        });

        // setPermanence
        py_SpatialPooler.def("setPermanence", [](SpatialPooler& self, UInt column, py::array& x)
        {
            self.setPermanence(column, get_it<Real>(x));
        });

        // getPermanence
        py_SpatialPooler.def("getPermanence", [](const SpatialPooler& self, UInt column, py::array& x)
        {
            self.getPermanence(column, get_it<Real>(x));
        });

        // getConnectedSynapses
        py_SpatialPooler.def("getConnectedSynapses", [](const SpatialPooler& self, UInt column, py::array& x)
        {
            self.getConnectedSynapses(column, get_it<UInt>(x));
        });

        // getConnectedCounts
        py_SpatialPooler.def("getConnectedCounts", [](const SpatialPooler& self, py::array& x)
        {
            self.getConnectedCounts(get_it<UInt>(x));
        });

        // getOverlaps
        py_SpatialPooler.def("getOverlaps", [](SpatialPooler& self)
        {
            auto overlaps = self.getOverlaps();

            return py::array_t<SynapseIdx>( overlaps.size(), overlaps.data());
        });

        // getBoostedOverlaps
        py_SpatialPooler.def("getBoostedOverlaps", [](SpatialPooler& self)
        {
            auto overlaps = self.getBoostedOverlaps();

            return py::array_t<Real>( overlaps.size(), overlaps.data());
        });


        ////////////////////
        // inhibitColumns

        auto inhibitColumns_func = [](SpatialPooler& self, py::array& overlaps)
        {
            std::vector<htm::Real> overlapsVector(get_it<Real>(overlaps), get_end<Real>(overlaps));

            std::vector<htm::UInt> activeColumnsVector;

            self.inhibitColumns_(overlapsVector, activeColumnsVector);

            return py::array_t<UInt>( activeColumnsVector.size(), activeColumnsVector.data());
        };

        py_SpatialPooler.def("_inhibitColumns", inhibitColumns_func);
        py_SpatialPooler.def("inhibitColumns_", inhibitColumns_func);


        //////////////////////
        // getIterationLearnNum
        py_SpatialPooler.def("getIterationLearnNum", &SpatialPooler::getIterationLearnNum);


        py_SpatialPooler.def("__str__",
            [](SpatialPooler &self) {
                std::stringstream buf;
                buf << self;
                return buf.str(); });


        // pickle

        py_SpatialPooler.def(py::pickle(
            [](const SpatialPooler& sp)
        {
            std::stringstream ss;

            sp.save(ss);

            return py::bytes( ss.str() );
        },
            [](py::bytes &s)
        {
            std::stringstream ss( s.cast<std::string>() );
            SpatialPooler sp;
            sp.load(ss);

            return sp;
        }));

    }
} // namespace htm_ext
