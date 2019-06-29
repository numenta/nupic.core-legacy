/* ----------------------------------------------------------------------
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
 * ---------------------------------------------------------------------- */

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <htm/utils/SdrMetrics.hpp>

namespace py = pybind11;

using namespace std;
using namespace htm;

namespace htm_ext
{
    void init_SDR_Metrics(py::module& m)
    {
        py::class_<MetricsHelper_> py_Helper(m, "MetricsHelper_");
        py_Helper.def( "addData", &MetricsHelper_::addData,
R"(Add an SDR datum to this Metric.  This method can only be called if the
Metric was constructed with dimensions and NOT an SDR.

Argument sdr is data source, its dimensions must be the same as this Metric's
dimensions.)");
        py_Helper.def_property_readonly( "period",
            [](const MetricsHelper_ &self){ return self.period; },
R"(Time constant for the exponential moving average which incorporate data into
this measurement.  If there are fewer data samples than the period then a regular
average is used instead of an exponential moving average.)");
        py_Helper.def_property_readonly( "samples",
            [](const MetricsHelper_ &self){ return self.samples; },
                "Number of data samples received & incorporated into this measurement.");
        py_Helper.def_property_readonly( "dimensions",
            [](const MetricsHelper_ &self){ return self.dimensions; },
                "Shape of the SDR data source.");

        // =====================================================================
        // SDR SPARSITY
        py::class_<Sparsity, MetricsHelper_> py_Sparsity(m, "Sparsity",
R"(Measures the sparsity of an SDR.  This accumulates measurements using an
exponential moving average, and outputs a summary of results.

Example Usage:
    A = SDR( dimensions )
    B = Sparsity( A, period = 1000 )
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

Argument period is time constant for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Sparsity.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this sparsity metric by calling method
sparsity.addData( SDR ) with an SDR which has these dimensions.

Argument period is time constant for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Sparsity.def_property_readonly("sparsity",
            [](const Sparsity &self) { return self.sparsity; },
                "Current Sparsity, or sparsity of most recently added SDR.");
        py_Sparsity.def( "min",  &Sparsity::min, "Minimum Sparsity");
        py_Sparsity.def( "max",  &Sparsity::max, "Maximum Sparsity");
        py_Sparsity.def( "mean", &Sparsity::mean,"Average of Sparsity");
        py_Sparsity.def( "std",  &Sparsity::std, "Standard Deviation of Sparsity");
        py_Sparsity.def("__str__", [](Sparsity &self){
            stringstream buf;
            buf << self;
            return py::str( buf.str() ).attr("strip")();
        });

        // =====================================================================
        // SDR ACTIVATION FREQUENCY
        py::class_<ActivationFrequency, MetricsHelper_>
            py_ActivationFrequency(m, "ActivationFrequency",
R"(Measures the activation frequency of each value in an SDR.  This accumulates
measurements using an exponential moving average, and outputs a summary of
results.

Activation frequencies are Real numbers in the range [0, 1], where zero
indicates never active, and one indicates always active.
Example Usage:
    A = SDR( 2 )
    B = ActivationFrequency( A, period = 1000 )
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

        py_ActivationFrequency.def( py::init<SDR&, UInt, Real>(),
R"(Argument sdr is data source to track.  Add data to this ActivationFrequency
instance by assigning to this SDR.

Argument period is time constant for exponential moving average.

Argument initialValue is Optional.  Makes this ActivationFrequency instance
think that it is the result of a long running process (even though it was just
created).  This assigns an initial activation frequency to all bits in the SDR,
and causes it to always use the exponential moving average instead of the
regular average which is usually applied to the first "period" many samples.

Note: This argument is useful for using this metric as part of boosting
      algorithms which seek to push the activation frequencies to a target
      value. These algorithms will overreact to the default early behavior of
      this class during the first "period" many samples.
)",
            py::arg("sdr"), py::arg("period"), py::arg("initialValue")=-1);

        py_ActivationFrequency.def( py::init<vector<UInt>, UInt, Real>(),
R"(Argument dimensions of SDR.  Add data to this ActivationFrequency
instance by calling method af.addData( SDR ) with an SDR which has
these dimensions.

Argument period is time constant for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"), py::arg("initialValue")=-1);

        py_ActivationFrequency.def_property_readonly("activationFrequency",
            [](const ActivationFrequency &self) {
                auto capsule = py::capsule(&self, [](void *self) {});
                return py::array(self.activationFrequency.size(), self.activationFrequency.data(), capsule); },
                    "Data Buffer of Activation Frequencies");
        py_ActivationFrequency.def( "min",     &ActivationFrequency::min, "Minimum of Activation Frequencies");
        py_ActivationFrequency.def( "max",     &ActivationFrequency::max, "Maximum of Activation Frequencies");
        py_ActivationFrequency.def( "mean",    &ActivationFrequency::mean,"Average of Activation Frequencies" );
        py_ActivationFrequency.def( "std",     &ActivationFrequency::std, "Standard Deviation of Activation Frequencies");
        py_ActivationFrequency.def( "entropy", &ActivationFrequency::entropy,
R"(Binary entropy is a measurement of information.  It measures how well the
SDR utilizes its resources (bits).  A low entropy indicates that many
bits in the SDR are under-utilized and do not transmit as much
information as they could.  A high entropy indicates that the SDR
optimally utilizes its resources.  The most optimal use of SDR resources
is when all bits have an equal activation frequency.  For convenience,
the entropy is scaled by the theoretical maximum into the range [0, 1].

Returns binary entropy of SDR, scaled to range [0, 1].)");
        py_ActivationFrequency.def("__str__", [](ActivationFrequency &self){
            stringstream buf;
            buf << self;
            return py::str( buf.str() ).attr("strip")();
        });

        // =====================================================================
        // SDR OVERLAP
        py::class_<Overlap, MetricsHelper_> py_Overlap(m, "Overlap",
R"(Measures the overlap between successive assignments to an SDR.  This class
accumulates measurements using an exponential moving average, and outputs a
summary of results.

This class normalizes the overlap into the range [0, 1] by dividing by the
number of active values.

Example Usage:
    A = SDR( dimensions = 1000 )
    B = Overlap( A, period = 1000 )
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
R"(Argument sdr is data source to track.  Add data to this Overlap instance
by assigning to this SDR.

Argument period is time constant for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Overlap.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this Overlap instance
by calling method overlap.addData( SDR ) with an SDR which has these dimensions.

Argument period is time constant for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Overlap.def("reset", &Overlap::reset, "For use with time-series data sets.");
        py_Overlap.def_property_readonly("overlap",
            [](const Overlap &self) { return self.overlap; },
                "Overlap between the two most recently added SDRs.");
        py_Overlap.def( "min",     &Overlap::min, "Minimum Overlap");
        py_Overlap.def( "max",     &Overlap::max, "Maximum Overlap");
        py_Overlap.def( "mean",    &Overlap::mean,"Average Overlap");
        py_Overlap.def( "std",     &Overlap::std, "Standard Deviation of Overlap");
        py_Overlap.def("__str__", [](Overlap &self){
            stringstream buf;
            buf << self;
            return py::str( buf.str() ).attr("strip")();
        });

        // =====================================================================
        // SDR METRICS
        py::class_<Metrics> py_Metrics(m, "Metrics",
R"(Measures an SDR.  This applies the following three metrics:
     Sparsity
     ActivationFrequency
     Overlap
This accumulates measurements using an exponential moving average, and
outputs a summary of results.

Example Usage:
    A = SDR( dimensions = 2000 )
    M = Metrics( A, period = 1000 )
    A.randomize( 0.10 )
    for i in range( 20 ):
        A.addNoise( 0.55 )

    M.sparsity            -> Sparsity class instance
    M.activationFrequency -> ActivationFrequency class instance
    M.overlap             -> Overlap class instance
    str(M) -> SDR( 2000 )
                Sparsity Min/Mean/Std/Max 0.1 / 0.1 / 0 / 0.1
                Activation Frequency Min/Mean/Std/Max 0 / 0.1 / 0.100464 / 0.666667
                Entropy 0.822222
                Overlap Min/Mean/Std/Max 0.45 / 0.45 / 0 / 0.45)");
        py_Metrics.def( py::init<SDR&, UInt>(),
R"(Argument sdr is data source to track.  Add data to this Metrics instance
by assigning to this SDR.

Argument period is time constant for exponential moving average.)",
            py::arg("sdr"), py::arg("period"));
        py_Metrics.def( py::init<vector<UInt>, UInt>(),
R"(Argument dimensions of SDR.  Add data to this Metrics instance
by calling method metrics.addData( SDR ) with an SDR which has these dimensions.

Argument period is time constant for exponential moving average.)",
            py::arg("dimensions"), py::arg("period"));
        py_Metrics.def( "reset", &Metrics::reset, "For use with time-series data sets.");
        py_Metrics.def( "addData", &Metrics::addData,
R"(Add an SDR datum to these Metrics.  This method can only be called if
Metrics was constructed with dimensions and NOT an SDR.

Argument sdr is data source, its dimensions must be the same as this Metric's
dimensions.)", py::arg("sdr"));
        py_Metrics.def_property_readonly("dimensions",
            [](const Metrics &self) { return self.dimensions; });
        py_Metrics.def_property_readonly("sparsity",
            [](const Metrics &self) -> const Sparsity &
                { return self.sparsity; });
        py_Metrics.def_property_readonly("activationFrequency",
            [](const Metrics &self) -> const ActivationFrequency &
                { return self.activationFrequency; });
        py_Metrics.def_property_readonly("overlap",
            [](const Metrics &self) -> const Overlap &
                { return self.overlap; });
        py_Metrics.def("__str__", [](Metrics &self){
            stringstream buf;
            buf << self;
            return py::str( buf.str() ).attr("strip")();
        });
    }
}
