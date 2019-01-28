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

#include <nupic/ntypes/SdrMetrics.hpp>
#include <nupic/utils/StringUtils.hpp>  // trim

namespace py = pybind11;

using namespace nupic;

namespace nupic_ext
{
    void init_SDR_Metrics(py::module& m)
    {
        py::class_<_SDR_MetricsHelper> py_Helper(m, "_SDR_MetricsHelper");
        py_Helper.def( "addData", &_SDR_MetricsHelper::addData );
        py_Helper.def_property_readonly( "period",
            [](const _SDR_MetricsHelper &self){ return self.period; });
        py_Helper.def_property_readonly( "samples",
            [](const _SDR_MetricsHelper &self){ return self.samples; });
        py_Helper.def_property_readonly( "dimensions",
            [](const _SDR_MetricsHelper &self){ return self.dimensions; });

        py::class_<SDR_Sparsity, _SDR_MetricsHelper> py_Sparsity(m, "SDR_Sparsity");
        py_Sparsity.def( py::init<SDR&, UInt>() );
        py_Sparsity.def( py::init<vector<UInt>, UInt>() );
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

        py::class_<SDR_ActivationFrequency, _SDR_MetricsHelper>
            py_ActivationFrequency(m, "SDR_ActivationFrequency");
        py_ActivationFrequency.def( py::init<SDR&, UInt>() );
        py_ActivationFrequency.def( py::init<vector<UInt>, UInt>() );
        py_ActivationFrequency.def_property_readonly("activationFrequency",
            [](const SDR_ActivationFrequency &self) { return self.activationFrequency; });
        py_ActivationFrequency.def( "min",     &SDR_ActivationFrequency::min );
        py_ActivationFrequency.def( "max",     &SDR_ActivationFrequency::max );
        py_ActivationFrequency.def( "mean",    &SDR_ActivationFrequency::mean );
        py_ActivationFrequency.def( "std",     &SDR_ActivationFrequency::std );
        py_ActivationFrequency.def( "entropy", &SDR_ActivationFrequency::entropy );
        py_ActivationFrequency.def("__str__", [](SDR_ActivationFrequency &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });

        py::class_<SDR_Overlap, _SDR_MetricsHelper> py_Overlap(m, "SDR_Overlap");
        py_Overlap.def( py::init<SDR&, UInt>() );
        py_Overlap.def( py::init<vector<UInt>, UInt>() );
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

        py::class_<SDR_Metrics> py_Metrics(m, "SDR_Metrics");
        py_Metrics.def( py::init<SDR&, UInt>() );
        py_Metrics.def( py::init<vector<UInt>, UInt>() );
        py_Metrics.def( "addData", &SDR_Metrics::addData );
        py_Metrics.def_property_readonly("dimensions",
            [](const SDR_Metrics &self) { return self.dimensions; });
        py_Metrics.def_property_readonly("sparsity",
            [](const SDR_Metrics &self) { return self.sparsity; });
        py_Metrics.def_property_readonly("activationFrequency",
            [](const SDR_Metrics &self) { return self.activationFrequency; });
        py_Metrics.def_property_readonly("overlap",
            [](const SDR_Metrics &self) { return self.overlap; });
        py_Metrics.def("__str__", [](SDR_Metrics &self){
            stringstream buf;
            buf << self;
            return StringUtils::trim( buf.str() );
        });
    }
}
