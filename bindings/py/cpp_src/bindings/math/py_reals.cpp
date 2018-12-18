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
PyBind11 bindings for the Real data type.
*/

#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/types/BasicType.hpp>

namespace py = pybind11;

namespace nupic_ext
{
    std::string GetNTARealType() { return "NTA_Real"; }

    std::string GetNumpyDataType(const std::string& type)
    {
#ifdef NTA_DOUBLE_PRECISION
        if (type == "NTA_Real")
        {
            return py::format_descriptor<double>::format();
        }
        else if (type == "NTA_Real32")
        {
            return py::format_descriptor<float>::format();
        }
        else if (type == "NTA_Real64")
        {
            return py::format_descriptor<double>::format();
        }

        throw std::runtime_error("Unsupported type name");
#else
        if (type == "NTA_Real")
        {
            return py::format_descriptor<float>::format();
        }
        else if (type == "NTA_Real32")
        {
            return py::format_descriptor<float>::format();
        }
        else if (type == "NTA_Real64")
        {
            return py::format_descriptor<double>::format();
        }

        throw std::runtime_error("Unsupported type name");
#endif // NTA_DOUBLE_PRECISION
    }

    void init_reals(py::module& m)
    {
        m.def("GetBasicTypeFromName", [](const std::string& type) { return nupic::BasicType::parse(type); });
        m.def("GetBasicTypeSize", [](const std::string& type) { return nupic::BasicType::getSize(nupic::BasicType::parse(type)); });

        m.def("GetNumpyDataType", &GetNumpyDataType);

        // GetNTARealType
        m.def("GetNTARealType", &GetNTARealType
            , "Gets the name of the NuPIC floating point base type, which is used for most internal calculations. This base type name can be used with GetBasicTypeFromName(),GetBasicTypeSize(), and GetNumpyDataType().");

        // GetNTAReal
        m.def("GetNTAReal", []()
        {
            return GetNumpyDataType(GetNTARealType());
        }, "Gets the numpy dtype of the NuPIC floating point base type, which is used for most internal calculations. The returned value can be used with numpy functions like numpy.array(..., dtype = dtype) and numpy.astype(..., dtype = dtype).");
    }

} // namespace nupic_ext
