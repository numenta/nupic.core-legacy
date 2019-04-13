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
 * PyBind11 bindings for SDRClassifier class
 */


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/algorithms/SDRClassifier.hpp>


namespace nupic_ext
{
namespace py = pybind11;
using namespace std;
using namespace nupic;
using nupic::sdr::SDR;
using namespace nupic::algorithms::sdr_classifier;

    void init_SDR_Classifier(py::module& m)
    {
        py::class_<Classifier> py_Classifier(m, "Classifier",
R"(TODO: DOCS)");

        py_Classifier.def(py::init<Real>(),
R"(TODO: DOCS)",
            py::arg("alpha") = 0.001);

        py_Classifier.def("infer", &Classifier::infer,
R"(TODO: DOCS)");

        py_Classifier.def("learn", &Classifier::learn,
R"(TODO: DOCS)");

        py_Classifier.def("learn", [](Classifier &self, const SDR &pattern, UInt categoryIdx)
            { self.learn( pattern, {categoryIdx} ); });

        // TODO: Pickle support


        py::class_<Predictor> py_Predictor(m, "Predictor",
R"(TODO: DOCS)");

        py_Predictor.def(py::init<const std::vector<UInt> &, Real>(),
R"(TODO: DOCS)",
            py::arg("steps"),
            py::arg("alpha") = 0.001);

        py_Predictor.def("reset", &Predictor::reset,
R"(TODO: DOCS)");

        py_Predictor.def("infer", &Predictor::infer,
R"(TODO: DOCS)");

        py_Predictor.def("learn", &Predictor::learn,
R"(TODO: DOCS)");

        py_Predictor.def("learn", [](Predictor &self, UInt recordNum, const SDR &pattern, UInt categoryIdx)
            { self.learn( recordNum, pattern, {categoryIdx} ); });

        // TODO: Pickle support
    }
} // namespace nupic_ext
