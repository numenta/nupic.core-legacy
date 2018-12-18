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
PyBind11 bindings for SDRClassifier class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>

namespace py = pybind11;
using namespace nupic;
using namespace nupic::algorithms::cla_classifier;
using namespace nupic::algorithms::sdr_classifier;

namespace nupic_ext
{
    void init_SDR_Classifier(py::module& m)
    {
        typedef std::vector<UInt> steps_t;

        py::class_<SDRClassifier> py_SDR_Classifier(m, "SDRClassifier");

        py_SDR_Classifier.def(py::init<>())
            .def(py::init<steps_t, Real64, Real64, UInt>(), py::arg("steps"), py::arg("alpha") = 0.001, py::arg("actValueAlpha") = 0.3, py::arg("verbosity") = 0);

        py_SDR_Classifier.def("convertedCompute", [](SDRClassifier& self, UInt recordNum, const std::vector<UInt>& patternNZ,
            const std::vector<UInt>& bucketIdxList,
            const std::vector<Real64>& actValueList,
            bool category, bool learn, bool infer)
        {
            ClassifierResult result;

            self.compute(recordNum, patternNZ, bucketIdxList, actValueList, category, learn, infer, &result);

            py::dict dict;

            for (map<Int, vector<Real64>*>::const_iterator it = result.begin();
                it != result.end(); ++it)
            {
                std::string key = "actualValues";

                if (it->first != -1)
                {
                    key = it->first;
                }

                py::list value;
                for (UInt i = 0; i < it->second->size(); ++i)
                {
                    value.append(it->second->at(i));
                }

                dict[key.c_str()] = value;
            }

            return dict;
        });

        py_SDR_Classifier.def("loadFromString", [](SDRClassifier& self, const std::string& inString)
        {
            std::istringstream inStream(inString);
            self.load(inStream);
        });

    }
} // namespace nupic_ext
