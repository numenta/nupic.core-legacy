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
#include <pybind11/stl.h>

#include <nupic/utils/SdrMetrics.hpp>
#include <nupic/utils/StringUtils.hpp>  // trim

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_SDR_Metrics(py::module& m)
    {
        py::class_<SDR_MetricsHelper_> py_Helper(m, "SDR_MetricsHelper_");
        py_Helper.def( "addData", &SDR_MetricsHelper_::addData,
R"(Add an SDR datum to this Metric.  This method can only be called if the
Metric was constructed with dimensions and NOT an SDR.

Argument sdr is data source, its dimensions must be the same as this Metric's
dimensions.)");
        py_Helper.def_property_readonly( "period",
            [](const SDR_MetricsHelper_ &self){ return self.period; });
        py_Helper.def_property_readonly( "samples",
            [](const SDR_MetricsHelper_ &self){ return self.samples; });
        py_Helper.def_property_readonly( "dimensions",
            [](const SDR_MetricsHelper_ &self){ return self.dimensions; });

        // =====================================================================
        // SDR SPARSITY
        py::class_<SDR_Sparsity, SDR_MetricsHelper_> py_Sparsity(m, "SDR_Sparsity",
R"(Measures the sparsity of an SDR.  This accumulates measurements using an
exponential moving average, and outputs a summary of results.

Example Usage:
    A = SDR( dimensions )
    B = SDR_Sparsity( A, period = 1000 )
    A.randomize( 0.01 )
    A.randomize( 0.15 )
    A.randomize( 0.05 )
    B.sparsity ->  0.05
    B.min()    ->  0.01
    B.max()    ->  0.15
    B.mean()   -> ~0.07
    B.std()    -> ~0.06
    str(B)     -> Sparsity Min/Mean/Std/Max 0.01 / 0.0700033 / 0.0588751 / 0.15)");
        py_Sparsity.def( py::init<SDR&, UInt>(),
R"(Argument sdr is data source is to track.  Add data to this sparsity metric by
assigning to this SDR.

Argument period is Time scale for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Sparsity.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this sparsity metric by calling method
sparsity.addData( SDR ) with an SDR which has these dimensions.

Argument period is time scale for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Sparsity.def_property_readonly("sparsity",
            [](const SDR_Sparsity &self) { return self.sparsity; });
        py_Sparsity.def( "min",  &SDR_Sparsity::min );
        py_Sparsity.def( "max",  &SDR_Sparsity::max );
        py_Sparsity.def( "mean", &SDR_Sparsity::mean );
        py_Sparsity.def( "std",  &SDR_Sparsity::std );
        py_Sparsity.def("__str__", [](SDR_Sparsity &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });

        // =====================================================================
        // SDR ACTIVATION FREQUENCY
        py::class_<SDR_ActivationFrequency, SDR_MetricsHelper_>
            py_ActivationFrequency(m, "SDR_ActivationFrequency",
R"(Measures the activation frequency of each value in an SDR.  This accumulates
measurements using an exponential moving average, and outputs a summary of
results.

Activation frequencies are Real numbers in the range [0, 1], where zero
indicates never active, and one indicates always active.
Example Usage:
    A = SDR( 2 )
    B = SDR_ActivationFrequency( A, period = 1000 )
    A.dense = [0, 0]
    A.dense = [0, 1]
    A.dense = [1, 1]
    B.activationFrequency -> { 0.33, 0.66 }
    B.min()     -> 1/3
    B.max()     -> 2/3
    B.mean()    -> 1/2
    B.std()     -> ~0.16
    B.entropy() -> ~0.92
    str(B)      -> Activation Frequency Min/Mean/Std/Max 0.333333 / 0.5 / 0.166667 / 0.666667
                   Entropy 0.918296)");
        py_ActivationFrequency.def( py::init<SDR&, UInt>(),
R"(Argument sdr is data source to track.  Add data to this SDR_ActivationFrequency
instance by assigning to this SDR.

Argument period is Time scale for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_ActivationFrequency.def( py::init<vector<UInt>, UInt>(),

R"(Argument dimensions of SDR.  Add data to this SDR_ActivationFrequency
instance by calling method af.addData( SDR ) with an SDR which has
these dimensions.

Argument period is Time scale for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_ActivationFrequency.def_property_readonly("activationFrequency",
            [](const SDR_ActivationFrequency &self) { return self.activationFrequency; });
        py_ActivationFrequency.def( "min",     &SDR_ActivationFrequency::min );
        py_ActivationFrequency.def( "max",     &SDR_ActivationFrequency::max );
        py_ActivationFrequency.def( "mean",    &SDR_ActivationFrequency::mean );
        py_ActivationFrequency.def( "std",     &SDR_ActivationFrequency::std );
        py_ActivationFrequency.def( "entropy", &SDR_ActivationFrequency::entropy,
R"(Binary entropy is a measurement of information.  It measures how well the
SDR utilizes its resources (bits).  A low entropy indicates that many
bits in the SDR are under-utilized and do not transmit as much
information as they could.  A high entropy indicates that the SDR
optimally utilizes its resources.  The most optimal use of SDR resources
is when all bits have an equal activation frequency.  For convenience,
the entropy is scaled by the theoretical maximum into the range [0, 1].

Returns binary entropy of SDR, scaled to range [0, 1].)");
        py_ActivationFrequency.def("__str__", [](SDR_ActivationFrequency &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });

        // =====================================================================
        // SDR OVERLAP
        py::class_<SDR_Overlap, SDR_MetricsHelper_> py_Overlap(m, "SDR_Overlap",
R"(Measures the overlap between successive assignments to an SDR.  This class
accumulates measurements using an exponential moving average, and outputs a
summary of results.

This class normalizes the overlap into the range [0, 1] by dividing by the
number of active values.

Example Usage:
    A = SDR( dimensions = 1000 )
    B = SDR_Overlap( A, period = 1000 )
    A.randomize( 0.20 )
    A.addNoise( 0.95 )  ->  5% overlap
    A.addNoise( 0.55 )  -> 45% overlap
    A.addNoise( 0.72 )  -> 28% overlap
    B.overlap   ->  0.28
    B.min()     ->  0.05
    B.max()     ->  0.45
    B.mean()    ->  0.26
    B.std()     -> ~0.16
    str(B)      -> Overlap Min/Mean/Std/Max 0.05 / 0.260016 / 0.16389 / 0.45)");
        py_Overlap.def( py::init<SDR&, UInt>(),
R"(Argument sdr is data source to track.  Add data to this SDR_Overlap instance
by assigning to this SDR.

Argument period is Time scale for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Overlap.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this SDR_Overlap instance
by calling method overlap.addData( SDR ) with an SDR which has these dimensions.

Argument period is Time scale for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Overlap.def_property_readonly("overlap",
            [](const SDR_Overlap &self) { return self.overlap; });
        py_Overlap.def( "min",     &SDR_Overlap::min );
        py_Overlap.def( "max",     &SDR_Overlap::max );
        py_Overlap.def( "mean",    &SDR_Overlap::mean );
        py_Overlap.def( "std",     &SDR_Overlap::std );
        py_Overlap.def("__str__", [](SDR_Overlap &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });

        // =====================================================================
        // SDR METRICS
        py::class_<SDR_Metrics> py_Metrics(m, "SDR_Metrics",
R"(Measures an SDR.  This applies the following three metrics:
     SDR_Sparsity
     SDR_ActivationFrequency
     SDR_Overlap
This accumulates measurements using an exponential moving average, and
outputs a summary of results.

Example Usage:
    A = SDR( dimensions = 2000 )
    M = SDR_Metrics( A, period = 1000 )
    A.randomize( 0.10 )
    for i in range( 20 ):
        A.addNoise( 0.55 )

    M.sparsity            -> SDR_Sparsity
    M.activationFrequency -> SDR_ActivationFrequency
    M.overlap             -> SDR_Overlap
    str(M) -> SDR( 2000 )
                Sparsity Min/Mean/Std/Max 0.1 / 0.1 / 0 / 0.1
                Activation Frequency Min/Mean/Std/Max 0 / 0.1 / 0.100464 / 0.666667
                Entropy 0.822222
                Overlap Min/Mean/Std/Max 0.45 / 0.45 / 0 / 0.45)");
        py_Metrics.def( py::init<SDR&, UInt>(),
R"(Argument sdr is data source to track.  Add data to this SDR_Metrics instance
by assigning to this SDR.

Argument period is Time scale for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Metrics.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this SDR_Metrics instance
by calling method metrics.addData( SDR ) with an SDR which has these dimensions.

Argument period is Time scale for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Metrics.def( "addData", &SDR_Metrics::addData,
R"(Add an SDR datum to these Metrics.  This method can only be called if
SDR_Metrics was constructed with dimensions and NOT an SDR.

Argument sdr is data source, its dimensions must be the same as this Metric's
dimensions.)", py::arg("sdr"));
        py_Metrics.def_property_readonly("dimensions",
            [](const SDR_Metrics &self) { return self.dimensions; });
        py_Metrics.def_property_readonly("sparsity",
            [](const SDR_Metrics &self) -> const SDR_Sparsity &
                { return self.sparsity; });
        py_Metrics.def_property_readonly("activationFrequency",
            [](const SDR_Metrics &self) -> const SDR_ActivationFrequency &
                { return self.activationFrequency; });
        py_Metrics.def_property_readonly("overlap",
            [](const SDR_Metrics &self) -> const SDR_Overlap &
                { return self.overlap; });
        py_Metrics.def("__str__", [](SDR_Metrics &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });
    }
}
