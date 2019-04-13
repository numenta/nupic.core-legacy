/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2018, Numenta, Inc.
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
 *
 * Author: @chhenning, 2018
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
using namespace nupic::algorithms::sdr_classifier;

    void init_SDR_Classifier(py::module& m)
    {
        py::class_<Classifier> py_Classifier(m, "Classifier",
R"(TODO: DOCS)");

        py_Classifier.def(py::init<Real64>(),
R"(TODO: DOCS)",
            py::arg("alpha") = 0.001);

        // TODO: Read only attribute "alpha"

        py_Classifier.def("infer", &Classifier::infer,
R"(TODO: DOCS)");

        py_Classifier.def("inferCategory", &Classifier::inferCategory,
R"(TODO: DOCS)");

        py_Classifier.def("learn", &Classifier::learn,
R"(TODO: DOCS)");

        // TODO: Pickle support




        // TODO: Predictor



        // py_Classifier.def("compute", [](
        //     Classifier& self,
        //     UInt recordNum,
        //     const std::vector<UInt>& patternNZ,
        //     const std::vector<UInt>& bucketIdx,
        //     const std::vector<Real64>& actValue)
        // {
        //     ClassifierResult result;

        //     self.compute(recordNum, patternNZ, bucketIdx, actValue,
        //                             category, learn, infer, result);

        //     py::dict dict;

        //     for (map<Int, PDF>::const_iterator it = result.begin(); it != result.end(); ++it)
        //     {
        //         std::string key = "actualValues";

        //         if (it->first != -1)
        //         {
        //             key = it->first;
        //         }

        //         py::list value;
        //         for (UInt i = 0; i < it->second.size(); ++i)
        //         {
        //             value.append(it->second.at(i));
        //         }

        //         dict[key.c_str()] = value;
        //     }

        //     return dict;
        // },
        // py::arg("recordNum") = 0u,
        // py::arg("patternNZ") = std::vector<UInt>({}),
        // py::arg("bucketIdx") = std::vector<UInt>({}),
        // py::arg("actValue") = std::vector<Real64>({}),
        // );
    }
} // namespace nupic_ext
