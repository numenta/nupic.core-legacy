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
PyBind11 bindings for SparseMatrix class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseMatrix.hpp>

#include "Matrix.hpp"
#include "bindings/engine/py_utils.hpp"
#include <fstream>

namespace py = pybind11;


typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32> > SparseMatrix32_t;

typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real64, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real64> > _SparseMatrix64;

namespace nupic_ext
{
    void init_SM32(py::module& m)
    {
        py::class_<SparseMatrix32_t> sm(m, "SM32");

        // create an alias for SM32
        m.attr("SparseMatrix") = sm;

        ////////////////////
        // Constructors

        sm.def(py::init<>());

        sm.def(py::init<nupic::UInt32, nupic::UInt32>(), py::arg("nrows"), py::arg("ncols"));

        sm.def(py::init([](const std::string& str)
        {
            SparseMatrix32_t s;

            std::stringstream ss(str);
            s.fromCSR(ss);

            return s;
        }));

        sm.def(py::init([](py::array_t<nupic::Real32>& a)
        {
            if (a.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

            SparseMatrix32_t s(a.shape(0), a.shape(1), get_it(a));

            return s;
        }));

        sm.def(py::init([](const SparseMatrix32_t& other)
        {
            return SparseMatrix32_t(other);
        }));


        sm.def(py::init([](const SparseMatrix32_t& other, py::array_t<nupic::UInt32>& data, int rowCol)
        {
            return SparseMatrix32_t(other, get_it(data), get_end(data), rowCol);
        }));

        ////////////////////////////
        // Properties
        sm.def_property_readonly("shape", [](const SparseMatrix32_t& sm)
        {
            return py::make_tuple(sm.nRows(), sm.nCols());
        });


        ////////////////////////////

        // Simple Members
        sm.def("isZero", &SparseMatrix32_t::isZero);
        sm.def("getIsNearlyZeroFunction", &SparseMatrix32_t::getIsNearlyZeroFunction);
        sm.def("isCompact", &SparseMatrix32_t::isCompact);
        sm.def("nRows", &SparseMatrix32_t::nRows);
        sm.def("nCols", &SparseMatrix32_t::nCols);
        sm.def("nCols", &SparseMatrix32_t::nCols);
        sm.def("nNonZeros", &SparseMatrix32_t::nNonZeros);

        sm.def("nBytes", &SparseMatrix32_t::nBytes);
        sm.def("nNonZerosOnRow", &SparseMatrix32_t::nNonZerosOnRow);
        sm.def("nNonZerosOnCol", &SparseMatrix32_t::nNonZerosOnCol);
        sm.def("nNonZeros", &SparseMatrix32_t::nNonZeros);
        sm.def("isRowZero", &SparseMatrix32_t::isRowZero);
        sm.def("isColZero", &SparseMatrix32_t::isColZero);
        sm.def("nNonZeroRows", &SparseMatrix32_t::nNonZeroRows);
        sm.def("nNonZeroCols", &SparseMatrix32_t::nNonZeroCols);
        sm.def("nZeroRows", &SparseMatrix32_t::nZeroRows);
        sm.def("nZeroCols", &SparseMatrix32_t::nZeroCols);
        sm.def("firstNonZeroOnRow", &SparseMatrix32_t::firstNonZeroOnRow);
        sm.def("lastNonZeroOnRow", &SparseMatrix32_t::lastNonZeroOnRow);
        sm.def("rowBandwidth", &SparseMatrix32_t::rowBandwidth);
        sm.def("firstNonZeroOnCol", &SparseMatrix32_t::firstNonZeroOnCol);
        sm.def("lastNonZeroOnCol", &SparseMatrix32_t::lastNonZeroOnCol);
        sm.def("colBandwidth", &SparseMatrix32_t::colBandwidth);
        sm.def("nonZerosInRowRange", &SparseMatrix32_t::nonZerosInRowRange);
        sm.def("nNonZerosInRowRange", &SparseMatrix32_t::nNonZerosInRowRange);
        sm.def("nNonZerosInBox", &SparseMatrix32_t::nNonZerosInBox);
        sm.def("isSymmetric", &SparseMatrix32_t::isSymmetric);
        sm.def("isBinary", &SparseMatrix32_t::isBinary);
        sm.def("equals", &SparseMatrix32_t::equals);
        sm.def("sameRowNonZeroIndices", &SparseMatrix32_t::sameRowNonZeroIndices);
        sm.def("sameNonZeroIndices", &SparseMatrix32_t::sameNonZeroIndices);
        sm.def("compact", &SparseMatrix32_t::compact);
        sm.def("decompact", &SparseMatrix32_t::decompact);
        sm.def("CSRSize", &SparseMatrix32_t::CSRSize);
        sm.def("fromCSR", &SparseMatrix32_t::fromCSR, py::arg("inStreamParam"), py::arg("zero_permissive") = false);
        sm.def("toCSR", &SparseMatrix32_t::toCSR);
        sm.def("fromBinary", &SparseMatrix32_t::fromBinary);
        sm.def("toBinary", &SparseMatrix32_t::toBinary);
        sm.def("resize", &SparseMatrix32_t::resize, py::arg("new_nrows"), py::arg("new_ncols"), py::arg("setToZero") = false);
        sm.def("reshape", &SparseMatrix32_t::reshape);
        sm.def("deleteRow", &SparseMatrix32_t::deleteRow);
        sm.def("deleteCol", &SparseMatrix32_t::deleteCol);
        sm.def("append", &SparseMatrix32_t::append, py::arg("other"), py::arg("zero_permissive") = false);
        sm.def("duplicateRow", &SparseMatrix32_t::duplicateRow);
        sm.def("setZero", &SparseMatrix32_t::setZero, py::arg("row"), py::arg("col"), py::arg("resizeYesNo") = false);
        sm.def("setDiagonalToZero", &SparseMatrix32_t::setDiagonalToZero);
        sm.def("setDiagonalToVal", &SparseMatrix32_t::setDiagonalToVal);
        sm.def("setNonZero", &SparseMatrix32_t::setNonZero, py::arg("i"), py::arg("j"), py::arg("val"), py::arg("resizeYesNo") = false);
        sm.def("set", &SparseMatrix32_t::set, py::arg("i"), py::arg("j"), py::arg("val"), py::arg("resizeYesNo") = false);
        sm.def("setBoxToZero", &SparseMatrix32_t::setBoxToZero);
        sm.def("setBox", &SparseMatrix32_t::setBox);
        sm.def("increment", &SparseMatrix32_t::increment, py::arg("i"), py::arg("j"), py::arg("delta") = 1, py::arg("resizeYesNo") = false);
        sm.def("incrementWNZ", &SparseMatrix32_t::incrementWNZ, py::arg("i"), py::arg("j"), py::arg("delta") = 1, py::arg("resizeYesNo") = false);
        sm.def("get", &SparseMatrix32_t::get);
        sm.def("row_nz_index_begin", &SparseMatrix32_t::row_nz_index_begin);
        sm.def("row_nz_index_end", &SparseMatrix32_t::row_nz_index_end);
        sm.def("row_nz_value_begin", &SparseMatrix32_t::row_nz_value_begin);
        sm.def("row_nz_value_end", &SparseMatrix32_t::row_nz_value_end);
        sm.def("setRowToZero", &SparseMatrix32_t::setRowToZero);
        sm.def("setRowToVal", &SparseMatrix32_t::setRowToVal);
        sm.def("setColToZero", &SparseMatrix32_t::setColToZero);
        sm.def("setColToVal", &SparseMatrix32_t::setColToVal);
        sm.def("setToZero", &SparseMatrix32_t::setToZero);
        sm.def("setFromOuter", &SparseMatrix32_t::setFromOuter, py::arg("x"), py::arg("y"), py::arg("keepMemory") = false);
        sm.def("setFromElementMultiplyWithOuter", &SparseMatrix32_t::setFromElementMultiplyWithOuter);

		sm.def("getRowToDense", &SparseMatrix32_t::getRowToDense); // Assuming expecting a vector as an argument.
        sm.def("copyRow", &SparseMatrix32_t::copyRow);
        sm.def("getColToDense", &SparseMatrix32_t::getColToDense);
        sm.def("setColFromDense", &SparseMatrix32_t::setColFromDense);
        sm.def("shiftRows", &SparseMatrix32_t::shiftRows);
        sm.def("shiftCols", &SparseMatrix32_t::shiftCols);
        sm.def("clipRow", &SparseMatrix32_t::clipRow, py::arg("row"), py::arg("val"), py::arg("above") = true);
        sm.def("clipRowBelowAndAbove", &SparseMatrix32_t::clipRowBelowAndAbove);
        sm.def("clipCol", &SparseMatrix32_t::clipCol, py::arg("col"), py::arg("val"), py::arg("above") = true);
        sm.def("clipColBelowAndAbove", &SparseMatrix32_t::clipColBelowAndAbove);
        sm.def("clip", &SparseMatrix32_t::clip, py::arg("val"), py::arg("above") = true);
        sm.def("clipBelowAndAbove", &SparseMatrix32_t::clipBelowAndAbove);
        sm.def("countWhereEqual", &SparseMatrix32_t::countWhereEqual);
        sm.def("countWhereGreater", &SparseMatrix32_t::countWhereGreater);
        sm.def("countWhereGreaterEqual", &SparseMatrix32_t::countWhereGreaterEqual);
        sm.def("argmax", &SparseMatrix32_t::argmax);
        sm.def("argmin", &SparseMatrix32_t::argmin);
        sm.def("normalizeRow", &SparseMatrix32_t::normalizeRow, py::arg("row"), py::arg("val") = 1.0, py::arg("exact") = false);
        sm.def("normalizeCol", &SparseMatrix32_t::normalizeCol, py::arg(""), py::arg("val") = 1.0, py::arg("exact") = false);
        sm.def("normalizeRows", &SparseMatrix32_t::normalizeRows, py::arg("val") = 1.0, py::arg("exact") = false);
        sm.def("normalizeCols", &SparseMatrix32_t::normalizeCols, py::arg("val") = 1.0, py::arg("exact") = false);
        sm.def("normalize", &SparseMatrix32_t::normalize, py::arg("val") = 1.0, py::arg("exact") = false);
        sm.def("normalize_max", &SparseMatrix32_t::normalize_max, py::arg("val") = 1.0);
        sm.def("rowSum", &SparseMatrix32_t::rowSum);
        sm.def("rowProd", &SparseMatrix32_t::rowProd);
        sm.def("colSum", &SparseMatrix32_t::colSum);
        sm.def("colProd", &SparseMatrix32_t::colProd);
        sm.def("sum", &SparseMatrix32_t::sum);
        sm.def("prod", &SparseMatrix32_t::prod);
        sm.def("lerp", &SparseMatrix32_t::lerp);
        sm.def("addTwoRows", &SparseMatrix32_t::addTwoRows);
        sm.def("addTwoCols", &SparseMatrix32_t::addTwoCols);
        sm.def("map", &SparseMatrix32_t::map);
        sm.def("incrementWithOuterProduct", &SparseMatrix32_t::incrementWithOuterProduct);
        sm.def("incrementOnOuterProductVal", &SparseMatrix32_t::incrementOnOuterProductVal, py::arg("rows"), py::arg("cols"), py::arg("val") = 1.0);
        sm.def("sortRowsAscendingNNZ", &SparseMatrix32_t::sortRowsAscendingNNZ);
        sm.def("replaceNZ", &SparseMatrix32_t::replaceNZ, py::arg("val") = 1.0);
        sm.def("diagNZProd", &SparseMatrix32_t::diagNZProd);
        sm.def("diagSum", &SparseMatrix32_t::diagSum);
        sm.def("diagNZLogSum", &SparseMatrix32_t::diagNZLogSum);
        sm.def("rowNegate", &SparseMatrix32_t::rowNegate);
        sm.def("colNegate", &SparseMatrix32_t::colNegate);
        sm.def("negate", &SparseMatrix32_t::negate);
        sm.def("rowAbs", &SparseMatrix32_t::rowAbs);
        sm.def("colAbs", &SparseMatrix32_t::colAbs);
        sm.def("abs", &SparseMatrix32_t::abs);
        sm.def("elementRowSquare", &SparseMatrix32_t::elementRowSquare);
        sm.def("elementColSquare", &SparseMatrix32_t::elementColSquare);
        sm.def("elementSquare", &SparseMatrix32_t::elementSquare);
        sm.def("elementRowCube", &SparseMatrix32_t::elementRowCube);
        sm.def("elementColCube", &SparseMatrix32_t::elementColCube);
        sm.def("elementCube", &SparseMatrix32_t::elementCube);
        sm.def("elementRowNZInverse", &SparseMatrix32_t::elementRowNZInverse);
        sm.def("elementColNZInverse", &SparseMatrix32_t::elementColNZInverse);
        sm.def("elementNZInverse", &SparseMatrix32_t::elementNZInverse);
        sm.def("elementRowSqrt", &SparseMatrix32_t::elementRowSqrt);
        sm.def("elementColSqrt", &SparseMatrix32_t::elementColSqrt);
        sm.def("elementSqrt", &SparseMatrix32_t::elementSqrt);
        sm.def("elementRowNZLog", &SparseMatrix32_t::elementRowNZLog);
        sm.def("elementColNZLog", &SparseMatrix32_t::elementColNZLog);
        sm.def("elementNZLog", &SparseMatrix32_t::elementNZLog);
        sm.def("elementRowNZExp", &SparseMatrix32_t::elementRowNZExp);
        sm.def("elementColNZExp", &SparseMatrix32_t::elementColNZExp);
        sm.def("elementNZExp", &SparseMatrix32_t::elementNZExp);
        sm.def("divide", &SparseMatrix32_t::divide);
        sm.def("elementRowNZPow", &SparseMatrix32_t::elementRowNZPow);
        sm.def("elementColNZPow", &SparseMatrix32_t::elementColNZPow);
        sm.def("elementNZPow", &SparseMatrix32_t::elementNZPow);
        sm.def("elementRowNZLogk", &SparseMatrix32_t::elementRowNZLogk);
        sm.def("elementColNZLogk", &SparseMatrix32_t::elementColNZLogk);
        sm.def("elementNZLogk", &SparseMatrix32_t::elementNZLogk);
        sm.def("rowAdd", &SparseMatrix32_t::rowAdd);
        sm.def("colAdd", &SparseMatrix32_t::colAdd);
        sm.def("elementNZAdd", &SparseMatrix32_t::elementNZAdd);
        sm.def("rowSubtract", &SparseMatrix32_t::rowSubtract);
        sm.def("colSubtract", &SparseMatrix32_t::colSubtract);
        sm.def("elementNZMultiply", &SparseMatrix32_t::elementNZMultiply);
        sm.def("elementNZDivide", &SparseMatrix32_t::elementNZDivide);

        sm.def("write", [](const SparseMatrix32_t& sm, py::args args) { throw std::runtime_error("Cap'n Proto is not available."); });
        sm.def("read", [](SparseMatrix32_t& sm, py::args args) { throw std::runtime_error("Cap'n Proto is not available."); });
        sm.def("getSchema", [](const SparseMatrix32_t& sm) { throw std::runtime_error("Cap'n Proto schema is not available."); });

        // member functions which are overloaded
		// Note: the overload_cast syntax requires C++14+ So rewriting them with C++11 syntax.
        //sm.def("add", py::overload_cast<const nupic::Real32&>(&SparseMatrix32_t::add));
		sm.def("add", (void (SparseMatrix32_t::*)(const SparseMatrix32_t& other)) &SparseMatrix32_t::add);
		sm.def("add", (void (SparseMatrix32_t::*)(const nupic::Real32 &val))      &SparseMatrix32_t::add);

        //sm.def("nonZeroIndicesIncluded", py::overload_cast<nupic::UInt32, const SparseMatrix32_t&>(&SparseMatrix32_t::nonZeroIndicesIncluded, py::const_));
        //sm.def("nonZeroIndicesIncluded", py::overload_cast<const SparseMatrix32_t&>(&SparseMatrix32_t::nonZeroIndicesIncluded, py::const_));
        sm.def("nonZeroIndicesIncluded", (bool (SparseMatrix32_t::*)(unsigned int, const SparseMatrix32_t &B) const) &SparseMatrix32_t::nonZeroIndicesIncluded);
        sm.def("nonZeroIndicesIncluded", (bool (SparseMatrix32_t::*)(const SparseMatrix32_t &B) const)                &SparseMatrix32_t::nonZeroIndicesIncluded);

        //sm.def("transpose", py::overload_cast<>(&SparseMatrix32_t::transpose));
        //sm.def("transpose", py::overload_cast<SparseMatrix32_t&>(&SparseMatrix32_t::transpose, py::const_));
        sm.def("transpose", (void (SparseMatrix32_t::*)(SparseMatrix32_t &tr) const) &SparseMatrix32_t::transpose);
        sm.def("transpose", (void (SparseMatrix32_t::*)( ))                          &SparseMatrix32_t::transpose);

        sm.def("getTransposed", [](const SparseMatrix32_t& sm)
        {
            SparseMatrix32_t result;
            sm.transpose(result);

            return result;
        });

        //sm.def("addToTranspose", py::overload_cast<>(&SparseMatrix32_t::addToTranspose));
        //sm.def("addToTranspose", py::overload_cast<SparseMatrix32_t&>(&SparseMatrix32_t::addToTranspose, py::const_));
        sm.def("addToTranspose", (void (SparseMatrix32_t::*)(SparseMatrix32_t&) const) &SparseMatrix32_t::addToTranspose);
        sm.def("addToTranspose", (void (SparseMatrix32_t::*)( ))                       &SparseMatrix32_t::addToTranspose);


        // overloaded functions with different return types seems to be problem
        // (void (SparseMatrix32_t::*)(nupic::UInt32, const nupic::Real32&)) &SparseMatrix32_t::thresholdRow
        sm.def("thresholdRow", (void (SparseMatrix32_t::*)(nupic::UInt32, const nupic::Real32&)) &SparseMatrix32_t::thresholdRow, "", py::arg("row"), py::arg("threshold") = nupic::Epsilon);
        sm.def("thresholdRow", [](SparseMatrix32_t& sm, nupic::UInt32 row, const nupic::Real32& threshold
            , py::array_t<nupic::UInt32>& cut_j, py::array_t<nupic::UInt32>& cut_nz)
        {
            return sm.thresholdRow(row, threshold, get_it(cut_j), get_it(cut_nz));
        }, "", py::arg("row"), py::arg("threshold"), py::arg("cut_j"), py::arg("cut_nz"));

        sm.def("thresholdCol", &SparseMatrix32_t::thresholdCol);

        //sm.def("multiply", py::overload_cast<const nupic::Real32&>(&SparseMatrix32_t::multiply));
        //sm.def("multiply", py::overload_cast<const SparseMatrix32_t&, SparseMatrix32_t&>(&SparseMatrix32_t::multiply, py::const_));
        sm.def("multiply", (void (SparseMatrix32_t::*)(const nupic::Real32&) )                           &SparseMatrix32_t::multiply);
        sm.def("multiply", (void (SparseMatrix32_t::*)(const SparseMatrix32_t&, SparseMatrix32_t&) const)&SparseMatrix32_t::multiply);


        ////////////////////////////

        sm.def("__add", [](SparseMatrix32_t& sm, nupic::Real32 val) { sm.add(val); });
        sm.def("__multiply", [](SparseMatrix32_t& sm, nupic::Real32 val) { sm.multiply(val); });
        sm.def("__subtract", [](SparseMatrix32_t& sm, nupic::Real32 val) { sm.subtract(val); });
        sm.def("__divide", [](SparseMatrix32_t& sm, nupic::Real32 val) { sm.divide(val); });

        /////////////////////
        // __add__

        // allows Matrix + 3

        sm.def("__add__", [](const SparseMatrix32_t& sm, int val)
        {
            SparseMatrix32_t result(sm);
            result.add(val);
            return result;
        });

        sm.def("__add__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.add(val);
            return result;
        });

        sm.def("__add__", [](const SparseMatrix32_t& sm, const SparseMatrix32_t& other)
        {
            SparseMatrix32_t result(sm);
            result.add(other);
            return result;
        });

        /////////////////////
        // __radd__

        // allows 3 + Matrix
        sm.def("__radd__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.add(val);

            return result;
        });


        sm.def("__sub__", [](const SparseMatrix32_t& sm, py::args args)
        {
            auto arg = args[0];

            SparseMatrix32_t result(sm);

            if (py::isinstance<py::int_>(arg))
            {
                auto val = static_cast<nupic::Real32>(arg.cast<int>());

                result.subtract(val);
            }
            else if (py::isinstance<py::float_>(arg))
            {
                auto val = arg.cast<float>();
                result.subtract(val);
            }
            else if (py::isinstance<SparseMatrix32_t>(arg))
            {
                auto other = arg.cast<SparseMatrix32_t>();
                result.subtract(other);
            }
            else
            {
                throw std::runtime_error("Not supported.");
            }

            return result;
        });

        sm.def("__rsub__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.subtract(val);

            return result;
        });

        ////////////////////////////
        // __mul__

        sm.def("__mul__", [](const SparseMatrix32_t& sm, int val)
        {
            SparseMatrix32_t result(sm);
            result.multiply(val);
            return result;
        });

        sm.def("__mul__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.multiply(val);
            return result;
        });

        sm.def("__mul__", [](const SparseMatrix32_t& sm, const SparseMatrix32_t& other)
        {
            SparseMatrix32_t result(sm);
            sm.multiply(other, result);
            return result;
        });

        sm.def("__mul__", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32> a)
        {
            if (a.ndim() == 1)
            {
                //@todo not sure this is correct
                py::array_t<nupic::Real32> y(sm.nRows());

                sm.rightVecProd(get_it(a), get_it(y));

                return SparseMatrix32_t(y.shape(0), 0, get_it(y));
            }
            else if (a.ndim() == 2)
            {
                SparseMatrix32_t result(sm);
                SparseMatrix32_t other(a.shape(0), a.shape(1), get_it(a));
                sm.multiply(other, result);
                return result;
            }

            throw std::runtime_error("Incorrect number of dimensions.");
        });


        ////////////////////////////
        // __rmul__

        sm.def("__rmul__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.multiply(val);

            return result;
        });

        ////////////////////////////
        // __truediv__

        sm.def("__truediv__", [](const SparseMatrix32_t& sm, int val)
        {
            SparseMatrix32_t result(sm);
            result.divide(val);
            return result;
        });

        sm.def("__truediv__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.divide(val);
            return result;
        });

        ////////////////////////////
        // __idiv__

        sm.def("__idiv__", [](SparseMatrix32_t& sm, nupic::Real32 val)
        {
            sm.divide(val);
            return sm;
        });

        ////////////////////////////
        // __rdiv__

        sm.def("__rdiv__", [](const SparseMatrix32_t& sm, nupic::Real32 val)
        {
            SparseMatrix32_t result(sm);
            result.divide(val);

            return result;
        });



        sm.def("__str__", [](const SparseMatrix32_t& sm)
        {
            auto out = py::array_t<nupic::Real32>({ sm.nRows(), sm.nCols() });

            sm.toDense(get_it(out));

            return out.attr("__str__")();
        });

        sm.def("__getitem__", [](const SparseMatrix32_t& sm, const std::vector<nupic::UInt32>& indices)
        {
            if (indices.size() != 2)
            {
                throw std::runtime_error("Indices must be two values.");
            }

            return sm.get(indices[0], indices[1]);
        });

        sm.def("__setitem__", [](SparseMatrix32_t& sm, const std::vector<nupic::UInt32>& indices, nupic::Real32 value)
        {
            if (indices.size() != 2)
            {
                throw std::runtime_error("Indices must be two values.");
            }

            sm.set(indices[0], indices[1], value);
        });

        //////////////////////
        //

        sm.def(py::pickle(
            [](const SparseMatrix32_t& sm)
        {
            std::stringstream s;

            sm.toCSR(s);

            return s.str();
        },
            [](std::string& s)
        {
            std::istringstream ss(s);
            SparseMatrix32_t sm;
            sm.fromCSR(ss);

            return sm;
        }));

        //sm.def("__getstate__", [](const SparseMatrix32_t& sm)
        //{
        //    std::stringstream s;

        //    sm.toCSR(s);

        //    return s.str();
        //});

        //sm.def("__setstate__", [](SparseMatrix32_t& sm, std::string& s)
        //{
        //    if (s.empty() == false)
        //    {
        //        std::istringstream ss(s);
        //        sm.fromCSR(ss);
        //        return true;
        //    }

        //    throw std::runtime_error("Failed to read SparseMatrix state from string.");
        //    return false;
        //});

        sm.def("__neg__", [](const SparseMatrix32_t& sm)
        {
            SparseMatrix32_t result(sm);

            result.negate();

            return result;
        });

        sm.def("__abs__", [](const SparseMatrix32_t& sm)
        {
            SparseMatrix32_t result(sm);

            result.abs();

            return result;
        });


        ///////////


        sm.def("copy", [](SparseMatrix32_t& sm, SparseMatrix32_t& other) { sm.copy(other); });

        sm.def("fromDense",
            [](SparseMatrix32_t& sm, py::array_t<nupic::Real>& matrix)
        {
            if (matrix.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

            sm.fromDense(matrix.shape(0), matrix.shape(1), get_it(matrix));
        });

        sm.def("toDense",
            [](const SparseMatrix32_t& sm)
        {
            auto out = py::array_t<nupic::Real32>({ sm.nRows(), sm.nCols() });

            sm.toDense(get_it(out));

            return out;
        });

        sm.def("setRowFromDense", [](SparseMatrix32_t& sm, nupic::UInt row, py::array_t<nupic::Real32> r)
        {
            sm.setRowFromDense_itr(row, get_it(r));
        });

        sm.def("setRowFromSparse", [](SparseMatrix32_t& sm, nupic::UInt row, py::array_t<nupic::UInt32>& ind, py::array_t<nupic::Real32>& nz)
        {
            sm.setRowFromSparse(row, get_it(ind), get_end(ind), get_it(nz));
        });

        sm.def("addRow", [](SparseMatrix32_t& self, py::array_t<nupic::Real32>& row)
        {
            self.addRow(get_it(row));
        });

        sm.def("addRowNZ", [](SparseMatrix32_t& self, py::array_t<nupic::UInt32>& ind, py::array_t<nupic::Real32>& nz, bool zero_permissive)
        {
            self.addRow(get_it(ind), get_end(ind), get_it(nz), zero_permissive);
        }, "", py::arg("ind"), py::arg("nz"), py::arg("zero_permissive") = false);


        sm.def("rowSums", [](const SparseMatrix32_t& self)
        {
            py::array_t<nupic::Real32> m(self.nRows());

            self.rowSums(get_it(m));

            return m;
        });

        sm.def("binaryLoadFromFile",
            [](SparseMatrix32_t& sm, const std::string filename)
        {
            std::ifstream load_file(filename.c_str());
            sm.fromBinary(load_file);
            load_file.close();
        });


        sm.def("binarySaveToFile",
            [](SparseMatrix32_t& sm, const std::string filename)
        {
            std::ofstream save_file(filename.c_str());
            sm.toBinary(save_file);
            save_file.close();
        });


        //void addCol(PyObject *col)
        sm.def("addCol", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& x)
        {
            sm.addCol(get_it(x));
        });

        //void addColNZ(PyObject *ind, PyObject *nz)
        sm.def("addColNZ", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& ind, py::array_t<nupic::Real32>& nz)
        {
            sm.addCol(get_it(ind), get_end(ind), get_it(nz));
        });

        //void deleteRows(PyObject *rowIndices)
        sm.def("deleteRows", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rowIndices)
        {
            sm.deleteRows(get_it(rowIndices), get_end(rowIndices));
        });

        //void deleteCols(PyObject *colIndices)
        sm.def("deleteCols", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& colIndices)
        {
            sm.deleteCols(get_it(colIndices), get_end(colIndices));
        });

        // getRow
        sm.def("getRow", [](SparseMatrix32_t& sm, nupic::UInt32 row)
        {
            const auto ncols = sm.nCols();
            py::array_t<nupic::Real32> dense_row(ncols);
            sm.getRowToDense_itr(row, get_it(dense_row));

            return dense_row;
        });


        //PyObject* getCol(nupic::UInt32 col) const
        sm.def("getCol", [](const SparseMatrix32_t& sm, nupic::UInt32 col)
        {
            py::array_t<nupic::Real32> dense_col(sm.nRows());
            sm.getColToDense_itr(col, get_it(dense_col));

            return dense_col;
        });

        //PyObject* getDiagonal() const
        sm.def("getDiagonal", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> diag(sm.nRows());
            sm.getDiagonalToDense(get_it(diag));

            return diag;
        });

        //PyObject* rowNonZeros(nupic::UInt32 row) const
        sm.def("rowNonZeros", [](const SparseMatrix32_t& sm, nupic::UInt32 row)
        {
            const auto n = sm.nNonZerosOnRow(row);
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(sm.nRows());
            sm.getRowToSparse(row, get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* rowNonZeroIndices(nupic::UInt32 row) const
        sm.def("rowNonZeroIndices", [](const SparseMatrix32_t& sm, nupic::UInt32 row)
        {
            const auto n = sm.nNonZerosOnRow(row);
            py::array_t<nupic::UInt32> ind(n);

            sm.getRowIndicesToSparse(row, get_it(ind));

            return ind;
        });

        //PyObject* colNonZeros(nupic::UInt32 col) const
        sm.def("colNonZeros", [](const SparseMatrix32_t& sm, nupic::UInt32 col)
        {
            const auto n = sm.nNonZerosOnCol(col);
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(n);

            sm.getColToSparse(col, get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* nonZeroRows() const
        sm.def("nonZeroRows", [](const SparseMatrix32_t& sm)
        {
            const auto nNonZeroRows = sm.nNonZeroRows();
            py::array_t<nupic::UInt32> nzRows(nNonZeroRows);
            sm.nonZeroRows(get_it(nzRows));

            return nzRows;
        });

        //PyObject* zeroRows() const
        sm.def("zeroRows", [](const SparseMatrix32_t& sm)
        {
            const auto nZeroRows = sm.nZeroRows();
            py::array_t<nupic::UInt32> zRows(nZeroRows);
            sm.zeroRows(get_it(zRows));

            return zRows;
        });

        //PyObject* nonZeroCols() const
        sm.def("nonZeroCols", [](const SparseMatrix32_t& sm)
        {
            const auto nNonZeroCols = sm.nNonZeroCols();
            py::array_t<nupic::UInt32> nzCols(nNonZeroCols);

            sm.nonZeroCols(get_it(nzCols));

            return nzCols;
        });

        //PyObject* zeroCols() const
        sm.def("zeroCols", [](const SparseMatrix32_t& sm)
        {
            const auto nZeroCols = sm.nZeroCols();
            py::array_t<nupic::UInt32> zCols(nZeroCols);

            sm.zeroCols(get_it(zCols));

            return zCols;
        });

        //PyObject* zeroRowAndCol() const
        sm.def("zeroRowAndCol", [](const SparseMatrix32_t& sm)
        {
            std::vector<nupic::UInt32> zrc;
            nupic::UInt32 c = sm.zeroRowAndCol(std::back_inserter(zrc));

            py::array_t<nupic::UInt32> toReturn(c);
            std::copy(zrc.begin(), zrc.end(), get_it(toReturn));

            return toReturn;
        });

        //void setElements(PyObject* py_i, PyObject* py_j, PyObject* py_v)
        sm.def("setElements", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j, py::array_t<nupic::Real32>& v)
        {
            sm.setElements(get_it(i), get_end(i), get_it(j), get_it(v));
        });

        //PyObject* getElements(PyObject* py_i, PyObject* py_j) const
        sm.def("getElements", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j)
        {
            py::array_t<nupic::Real32> v(i.size());

            sm.getElements(get_it(i), get_end(i), get_it(j), get_it(v));

            return v;
        });

        //void setOuter(PyObject* py_i, PyObject* py_j, PyObject* py_v)
        sm.def("setOuter", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j, py::array_t<nupic::Real32>& v)
        {
            Numpy_Matrix<nupic::Real32> m(v.request());

            // setOuter cannot deal with py::array_t
            sm.setOuter(get_it(i), get_end(i), get_it(j), get_end(j), m);

        }, "Sets on the outer product of the passed ranges.");

        //SparseMatrix32 getOuter(PyObject* py_i, PyObject* py_j) const
        sm.def("getOuter", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j)
        {
            SparseMatrix32_t v(i.size(), j.size());

            sm.getOuter(get_it(i), get_end(i), get_it(j), get_end(j), v);

            return v;
        }, "Get on the outer products of the passed ranges.");

        //PyObject* getAllNonZeros(bool three_lists = false) const
        sm.def("getAllNonZeros", [](const SparseMatrix32_t& sm, bool three_lists)
        {
            const auto nnz = sm.nNonZeros();

            py::array_t<nupic::UInt32> rows(nnz), cols(nnz);
            py::array_t<nupic::Real32> vals(nnz);

            sm.getAllNonZeros(get_it(rows), get_it(cols), get_it(vals));

            if (!three_lists)
            {
                // Return one list of triples
                py::tuple toReturn(nnz);

                for (nupic::UInt32 i = 0; i != nnz; ++i)
                {
                    toReturn[i] = py::make_tuple(rows.data(i), cols.data(i), vals.data(i));
                }

                return toReturn;
            }
            else
            {
                py::tuple toReturn(3);

                // Return three lists
                toReturn[0] = rows;
                toReturn[1] = cols;
                toReturn[2] = vals;

                return toReturn;
            }
        }, "Returns the positions and values of all the non-zeros stored in this matrix. The result can be either three lists ((i), (j), (v)) or one list of triples (i,j,v)."
        , py::arg("three_lists") = false);

        //void setAllNonZeros(nupic::UInt32 nrows, nupic::UInt32 ncols, PyObject* py_i, PyObject* py_j, PyObject* py_v, bool sorted = true)
        sm.def("setAllNonZeros", [](SparseMatrix32_t& sm, nupic::UInt32 nrows, nupic::UInt32 ncols
            , py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j, py::array_t<nupic::Real32>& v
            , bool sorted)
        {
            sm.setAllNonZeros(nrows, ncols,
                get_it(i), get_end(i),
                get_it(j), get_end(j),
                get_it(v), get_end(v),
                sorted);
        }, "", py::arg("nrows"), py::arg("ncols"), py::arg("i"), py::arg("j"), py::arg("v"), py::arg("sorted") = true);

        //PyObject* getNonZerosInBox(nupic::UInt32 row_begin, nupic::UInt32 row_end, nupic::UInt32 col_begin, nupic::UInt32 col_end) const
        sm.def("getNonZerosInBox", [](const SparseMatrix32_t& sm, nupic::UInt32 row_begin, nupic::UInt32 row_end
            , nupic::UInt32 col_begin, nupic::UInt32 col_end)
        {
            std::vector<nupic::UInt32> rows, cols;
            std::vector<nupic::Real32> vals;

            sm.getNonZerosInBox(row_begin, row_end, col_begin, col_end,
                std::back_inserter(rows),
                std::back_inserter(cols),
                std::back_inserter(vals));

            py::tuple toReturn(rows.size());

            for (nupic::UInt32 i = 0; i != rows.size(); ++i)
            {
                toReturn[i] = py::make_tuple(rows[i], cols[i], vals[i]);
            }

            return toReturn;
        });

        //PyObject* tolist() const
        sm.def("tolist", [](const SparseMatrix32_t& sm)
        {
            const auto nnz = sm.nNonZeros();
            std::vector<nupic::UInt32> rows(nnz), cols(nnz);
            py::array_t<nupic::Real32> vals(nnz);
            sm.getAllNonZeros(rows.begin(), cols.begin(), get_it(vals));

            py::tuple toReturn(rows.size());

            py::tuple ind_list(nnz);

            for (nupic::UInt32 i = 0; i != nnz; ++i)
            {
                ind_list[i] = py::make_tuple(rows[i], cols[i]);
            }

            return py::make_tuple(ind_list, vals);
        });

        ///////////////////////////
        // setSlice

        //void setSlice(nupic::UInt32 i_begin, nupic::UInt32 j_begin, const SparseMatrix32& other)
        sm.def("setSlice", [](SparseMatrix32_t& sm, nupic::UInt32 i_begin, nupic::UInt32 j_begin, const SparseMatrix32_t& other)
        {
            sm.setSlice(i_begin, j_begin, other);
        });

        //void setSlice(nupic::UInt32 i_begin, nupic::UInt32 j_begin, PyObject* py_other)
        sm.def("setSlice", [](SparseMatrix32_t& sm, nupic::UInt32 i_begin, nupic::UInt32 j_begin, py::array_t<nupic::Real32>& other)
        {
            Numpy_Matrix<nupic::Real32> m(other.request());

            sm.setSlice(i_begin, j_begin, m);
        });

        ///////////////////////////
        // getSlice

        //SparseMatrix32 getSlice(nupic::UInt32 i_begin, nupic::UInt32 i_end, nupic::UInt32 j_begin, nupic::UInt32 j_end) const
        sm.def("getSlice", [](const SparseMatrix32_t& sm, nupic::UInt32 i_begin, nupic::UInt32 i_end, nupic::UInt32 j_begin, nupic::UInt32 j_end)
        {
            SparseMatrix32_t other(i_end - i_begin, j_end - j_begin);

            sm.getSlice(i_begin, i_end, j_begin, j_end, other);

            return other;
        });

        //SparseMatrix32 getSlice2(nupic::UInt32 i_begin, nupic::UInt32 i_end, nupic::UInt32 j_begin, nupic::UInt32 j_end) const
        sm.def("getSlice2", [](const SparseMatrix32_t& sm, nupic::UInt32 i_begin, nupic::UInt32 i_end, nupic::UInt32 j_begin, nupic::UInt32 j_end)
        {
            SparseMatrix32_t other(i_end - i_begin, j_end - j_begin);

            sm.getSlice2(i_begin, i_end, j_begin, j_end, other);

            return other;
        });

        //inline void setRowsToZero(PyObject* py_rows)
        sm.def("setRowsToZero", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows)
        {
            sm.setRowsToZero(get_it(rows), get_end(rows));
        });

        //inline void setColsToZero(PyObject* py_cols)
        sm.def("setColsToZero", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& cols)
        {
            sm.setColsToZero(get_it(cols), get_end(cols));
        });

        //inline void setDiagonal(PyObject* py_v)
        sm.def("setDiagonal", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& v)
        {
            sm.setDiagonal(get_it(v));
        });

        //void incrementOnOuterWNZ(PyObject* py_i, PyObject* py_j, nupic::Real32 delta = 1)
        sm.def("incrementOnOuterWNZ", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j, nupic::Real32 delta)
        {
            sm.incrementOnOuterWNZ(get_it(i), get_end(i), get_it(j), get_end(j), delta);
        }, "", py::arg("i"), py::arg("j"), py::arg("delta") = 1);

        //void incrementOnOuterWNZWThreshold(PyObject* py_i, PyObject* py_j, nupic::Real32 threshold, nupic::Real32 delta = 1)
        sm.def("incrementOnOuterWNZWThreshold", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j
            , nupic::Real32 threshold, nupic::Real32 delta)
        {
            sm.incrementOnOuterWNZWThreshold(get_it(i), get_end(i), get_it(j), get_end(j), threshold, delta);
        }, "", py::arg("i"), py::arg("j"), py::arg("threshold"), py::arg("delta") = 1);

        //void _incrementNonZerosOnOuter(PyObject* py_rows, PyObject* py_cols, nupic::Real32 delta)
        sm.def("incrementNonZerosOnOuter", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols, nupic::Real32 delta)
        {
            sm.incrementNonZerosOnOuter(get_it(rows), get_end(rows), get_it(cols), get_end(cols), delta);
        });

        //void _incrementNonZerosOnRowsExcludingCols(PyObject* py_rows, PyObject* py_cols, nupic::Real32 delta)
        sm.def("incrementNonZerosOnRowsExcludingCols", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols, nupic::Real32 delta)
        {
            sm.incrementNonZerosOnRowsExcludingCols(get_it(rows), get_end(rows), get_it(cols), get_end(cols), delta);
        });

        //void _setZerosOnOuter(PyObject* py_rows, PyObject* py_cols, nupic::Real32 value)
        sm.def("setZerosOnOuter", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols, nupic::Real32 value)
        {
            sm.setZerosOnOuter(get_it(rows), get_end(rows), get_it(cols), get_end(cols), value);
        });

        ///////////////////////////////////
        // setRandomZerosOnOuter

        sm.def("setRandomZerosOnOuter", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols
            , nupic::Int32 numNewNonZeros, nupic::Real32 value, nupic::Random& rng)
        {
            // _setRandomZerosOnOuter_singleCount
            sm.setRandomZerosOnOuter(get_it(rows), get_end(rows), get_it(cols), get_end(cols), numNewNonZeros, value, rng);

        });

        sm.def("setRandomZerosOnOuter", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols
            , py::array_t<nupic::Int32>& newNonZeroCounts, nupic::Real32 value, nupic::Random& rng)
        {
            //_setRandomZerosOnOuter_multipleCounts
            sm.setRandomZerosOnOuter(get_it(rows), get_end(rows), get_it(cols), get_end(cols), get_it(newNonZeroCounts), get_end(newNonZeroCounts), value, rng);
        });


        ///////////////////////////////////

        //void _increaseRowNonZeroCountsOnOuterTo(PyObject* py_rows, PyObject* py_cols, nupic::Int32 numDesiredNonZeros, nupic::Real32 initialValue, nupic::Random& rng)
        sm.def("increaseRowNonZeroCountsOnOuterTo", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols
            , nupic::Int32 numDesiredNonZeros, nupic::Real32 initialValue, nupic::Random& rng)
        {
            sm.increaseRowNonZeroCountsOnOuterTo(get_it(rows), get_end(rows), get_it(cols), get_end(cols),
                numDesiredNonZeros, initialValue,
                rng);
        });

        //void _clipRowsBelowAndAbove(PyObject* py_rows, nupic::Real32 a, nupic::Real32 b)
        sm.def("clipRowsBelowAndAbove", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, nupic::Real32 a, nupic::Real32 b)
        {
            sm.clipRowsBelowAndAbove(get_it(rows), get_end(rows), a, b);
        });

        sm.def("nNonZerosPerRow", [](const SparseMatrix32_t& sm, py::args args)
        {
            if (args.size() == 0)
            {
                // _nNonZerosPerRow_allRows

                py::array_t<nupic::UInt32> nnzpr(sm.nRows());

                sm.nNonZerosPerRow(get_it(nnzpr));

                return nnzpr;
            }
            else if (args.size() == 1)
            {
                // _nNonZerosPerRow

                auto arg = args[0];
                std::string as_string = py::str(arg.get_type());

                if (py::isinstance<py::array_t<nupic::UInt32>>(arg))
                {
                    auto rows = arg.cast<py::array_t<nupic::UInt32>>();

                    py::array_t<nupic::UInt32> nnzpr(rows.size());

                    sm.nNonZerosPerRow(get_it(rows), get_end(rows), get_it(nnzpr));

                    return nnzpr;
                }
                if (py::isinstance<py::list>(arg))
                {
                    try
                    {
                        auto rows = arg.cast<std::vector<nupic::UInt32>>();

                        py::array_t<nupic::UInt32> nnzpr(rows.size());

                        sm.nNonZerosPerRow(rows.begin(), rows.end(), get_it(nnzpr));

                        return nnzpr;
                    }
                    catch (...)
                    {
                        throw std::runtime_error("Cannot apply list.");
                    }
                }

                else
                {
                    throw std::runtime_error("Incorrect parameter.");
                }
            }

            throw std::runtime_error("Too many arguments.");
        });


        sm.def("nNonZerosPerCol", [](const SparseMatrix32_t& self)
        {
            py::array_t<nupic::UInt32> x(self.nCols());

            self.nNonZerosPerCol(get_it(x));

            return x;
        });


        //PyObject* _nNonZerosPerRowOnCols(PyObject* py_rows, PyObject* py_cols)
        sm.def("nNonZerosPerRowOnCols", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& rows, py::array_t<nupic::UInt32>& cols)
        {
            py::array_t<nupic::UInt32> out(rows.size());

            sm.nNonZerosPerRowOnCols(get_it(rows), get_end(rows), get_it(cols), get_end(cols), get_it(out));

            return out;
        });

        //PyObject* nNonZerosPerCol() const
        sm.def("nNonZerosPerCol", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::UInt32> nnzpc(sm.nCols());

            sm.nNonZerosPerCol(get_it(nnzpc));

            return nnzpc;
        });

        //PyObject* rowBandwidths() const
        sm.def("rowBandwidths", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::UInt32> nnzpc(sm.nRows());

            sm.rowBandwidths(get_it(nnzpc));

            return nnzpc;
        });

        //PyObject* colBandwidths() const
        sm.def("colBandwidths", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::UInt32> nnzpc(sm.nCols());

            sm.colBandwidths(get_it(nnzpc));

            return nnzpc;
        });

        //SparseMatrix32 nNonZerosPerBox(PyObject* box_i, PyObject* box_j) const
        sm.def("nNonZerosPerBox", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& box_i, py::array_t<nupic::UInt32>& box_j)
        {
            SparseMatrix32_t result(box_i.size(), box_j.size());

            sm.nNonZerosPerBox(get_it(box_i), get_end(box_i), get_it(box_j), get_end(box_j), result);

            return result;
        });

        //PyObject* max() const
        sm.def("max", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 max_row, max_col;
            nupic::Real32 max_val;
            sm.max(max_row, max_col, max_val);

            return py::make_tuple(max_row, max_col, max_val);
        });

        //PyObject* min() const
        sm.def("min", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 min_row, min_col;
            nupic::Real32 min_val;
            sm.min(min_row, min_col, min_val);

            return py::make_tuple(min_row, min_col, min_val);
        });

        //PyObject* rowMin(nupic::UInt32 row_index) const
        sm.def("rowMin", [](const SparseMatrix32_t& sm, nupic::UInt32 row_index)
        {
            nupic::UInt32 idx;
            nupic::Real32 min_val;
            sm.rowMin(row_index, idx, min_val);

            return py::make_tuple(idx, min_val);
        });

        //PyObject* rowMax(nupic::UInt32 row_index) const
        sm.def("rowMax", [](const SparseMatrix32_t& sm, nupic::UInt32 row_index)
        {
            nupic::UInt32 idx;
            nupic::Real32 max_val;
            sm.rowMax(row_index, idx, max_val);
            return py::make_tuple(idx, max_val);
        });

        //PyObject* colMin(nupic::UInt32 col_index) const
        sm.def("colMin", [](const SparseMatrix32_t& sm, nupic::UInt32 col_index)
        {
            nupic::UInt32 idx;
            nupic::Real32 min_val;
            sm.colMin(col_index, idx, min_val);
            return py::make_tuple(idx, min_val);
        });

        //PyObject* colMax(nupic::UInt32 row_index) const
        sm.def("colMax", [](const SparseMatrix32_t& sm, nupic::UInt32 row_index)
        {
            nupic::UInt32 idx;
            nupic::Real32 max_val;
            sm.colMax(row_index, idx, max_val);
            return py::make_tuple(idx, max_val);
        });

        //PyObject* rowMax() const
        sm.def("rowMax", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 n = sm.nRows();
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(n);

            sm.rowMax(get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* rowMin() const
        sm.def("rowMin", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 n = sm.nRows();
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(n);
            sm.rowMin(get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* colMax() const
        sm.def("colMax", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 n = sm.nCols();
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(n);
            sm.colMax(get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* colMin() const
        sm.def("colMin", [](const SparseMatrix32_t& sm)
        {
            nupic::UInt32 n = sm.nCols();
            py::array_t<nupic::UInt32> ind(n);
            py::array_t<nupic::Real32> val(n);

            sm.colMin(get_it(ind), get_it(val));

            return py::make_tuple(ind, val);
        });

        //PyObject* boxMin(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col) const
        sm.def("boxMin", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col)
        {
            nupic::UInt32 min_row, min_col;
            nupic::Real32 min_val;

            sm.boxMin(begin_row, end_row, begin_col, end_col, min_row, min_col, min_val);

            return py::make_tuple(min_row, min_col, min_val);
        });

        //PyObject* boxMax(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col) const
        sm.def("boxMax", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col)
        {
            nupic::UInt32 max_row, max_col;
            nupic::Real32 max_val;

            sm.boxMax(begin_row, end_row, begin_col, end_col, max_row, max_col, max_val);

            return py::make_tuple(max_row, max_col, max_val);
        });

        //PyObject* whereEqual(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value) const
        sm.def("whereEqual", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value)
        {
            std::vector<nupic::UInt32> rows, cols;
            sm.whereEqual(begin_row, end_row, begin_col, end_col, value,
                std::back_inserter(rows), std::back_inserter(cols));

            py::tuple toReturn(rows.size());

            for (size_t i = 0; i != rows.size(); ++i)
            {
                toReturn[i] = py::make_tuple(rows[i], cols[i]);
            }

            return toReturn;
        });

        //PyObject* whereGreater(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value) const
        sm.def("whereGreater", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value)
        {
            std::vector<nupic::UInt32> rows, cols;
            sm.whereGreater(begin_row, end_row, begin_col, end_col, value,
                std::back_inserter(rows), std::back_inserter(cols));

            py::array_t<nupic::UInt32> toReturn({ static_cast<int>(rows.size()), 2 });
            auto r = toReturn.mutable_unchecked<2>();

            for (size_t i = 0; i != rows.size(); ++i)
            {
                r(i, 0) = rows[i];
                r(i, 1) = cols[i];
            }

            return toReturn;
        });

        ////PyObject* whereGreaterEqual(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value) const
        sm.def("whereGreaterEqual", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value)
        {
            std::vector<nupic::UInt32> rows, cols;
            sm.whereGreaterEqual(begin_row, end_row, begin_col, end_col, value,
                std::back_inserter(rows), std::back_inserter(cols));

            py::array_t<nupic::UInt32> toReturn({ static_cast<int>(rows.size()), 2 });
            auto r = toReturn.mutable_unchecked<2>();

            for (size_t i = 0; i != rows.size(); ++i) {
                r(i, 0) = rows[i];
                r(i, 1) = cols[i];
            }
            return toReturn;
        });

        //nupic::UInt32 countWhereGreaterOrEqual(nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value) const
        sm.def("countWhereGreaterOrEqual", [](const SparseMatrix32_t& sm, nupic::UInt32 begin_row, nupic::UInt32 end_row, nupic::UInt32 begin_col, nupic::UInt32 end_col, const nupic::Real32& value)
        {
            std::vector<nupic::UInt32> rows, cols;
            return sm.countWhereGreaterEqual(begin_row, end_row, begin_col, end_col, value);
        });

        //void permuteRows(PyObject* py_permutation)
        sm.def("permuteRows", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& permutation)
        {
            sm.permuteRows(get_it(permutation));
        });

        //void permuteCols(PyObject* py_permutation)
        sm.def("permuteCols", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& permutation)
        {
            sm.permuteCols(get_it(permutation));
        });

        //PyObject* rowSums() const
        sm.def("rowSums", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nRows());

            sm.rowSums(get_it(m));

            return m;
        });

        //PyObject* colSums() const
        sm.def("colSums", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nCols());

            sm.colSums(get_it(m));

            return m;
        });

        //PyObject* addRows(PyObject* whichRows) const
        sm.def("addRows", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& whichRows)
        {
            py::array_t<nupic::Real32> res(sm.nCols());

            sm.addRows(get_it(whichRows), get_end(whichRows), get_it(res), get_end(res));

            return res;
        });

        //PyObject* addListOfRows(PyObject* py_whichRows) const
        sm.def("addListOfRows", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& whichRows)
        {
            py::array_t<nupic::Real32> res(sm.nCols());

            sm.addListOfRows(get_it(whichRows), get_end(whichRows), get_it(res), get_end(res));

            return res;
        });

        //PyObject* rowProds() const
        sm.def("rowProds", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nRows());

            sm.rowProds(get_it(m));

            return m;
        });

        //PyObject* colProds() const
        sm.def("colProds", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nCols());

            sm.colProds(get_it(m));

            return m;
        });

        //PyObject* logRowSums() const
        sm.def("logRowSums", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nRows());

            sm.logRowSums(get_it(m), get_end(m));

            return m;
        });

        //PyObject* logColSums() const
        sm.def("logColSums", [](const SparseMatrix32_t& sm)
        {
            py::array_t<nupic::Real32> m(sm.nCols());

            sm.logColSums(get_it(m), get_end(m));

            return m;
        });

        //void scaleRows(PyObject* py_s)
        sm.def("scaleRows", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& s)
        {
            sm.scaleRows(get_it(s));
        });

        //void scaleCols(PyObject* py_s)
        sm.def("scaleCols", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& s)
        {
            sm.scaleCols(get_it(s));
        });

        //void normalizeBlockByRows(PyObject* py_inds, nupic::Real32 val = -1.0, nupic::Real32 eps_n = 1e-6)
        sm.def("normalizeBlockByRows", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& inds, nupic::Real32 val, nupic::Real32 eps_n)
        {
            sm.normalizeBlockByRows(get_it(inds), get_end(inds), val, eps_n);
        }, "", py::arg("inds"), py::arg("val") = -1.0, py::arg("eps_n") = 1e-6);

        //void normalizeBlockByRows_binary(PyObject* py_inds, nupic::Real32 val = -1.0, nupic::Real32 eps_n = 1e-6)
        sm.def("normalizeBlockByRows_binary", [](SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& inds, nupic::Real32 val, nupic::Real32 eps_n)
        {
            sm.normalizeBlockByRows_binary(get_it(inds), get_end(inds), val, eps_n);
        }, "", py::arg("inds"), py::arg("val") = -1.0, py::arg("eps_n") = 1e-6);

        //void axby(nupic::UInt32 row, nupic::Real32 a, nupic::Real32 b, PyObject *xIn)
        sm.def("axby", [](SparseMatrix32_t& sm, nupic::UInt32 row, nupic::Real32 a, nupic::Real32 b, py::array_t<nupic::Real32>& xIn)
        {
            sm.axby(row, a, b, get_it(xIn));
        });

        //void axby(nupic::Real32 a, nupic::Real32 b, PyObject *xIn)
        sm.def("axby", [](SparseMatrix32_t& sm, nupic::Real32 a, nupic::Real32 b, py::array_t<nupic::Real32>& xIn)
        {
            sm.axby(a, b, get_it(xIn));
        });

        //nupic::Real32 rightVecProd(nupic::UInt32 row, PyObject *xIn) const
        sm.def("rightVecProd", [](const SparseMatrix32_t& sm, nupic::UInt32 row, py::array_t<nupic::Real32>& xIn)
        {
            return sm.rightVecProd(row, get_it(xIn));
        }, "Computes the dot product of the given row with the given vector.");

        //inline PyObject* rightVecProd(PyObject *xIn) const
        sm.def("rightVecProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            sm.rightVecProd(get_it(xIn), get_it(y));

            return y;
        }, "Regular matrix vector multiplication, with allocation of the result.");

        //inline void rightVecProd_fast(PyObject *xIn, PyObject *yOut) const
        sm.def("rightVecProd_fast", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn, py::array_t<nupic::Real32>& yOut)
        {
            sm.rightVecProd(get_it(xIn), get_it(yOut));
        }, "Regular matrix vector multiplication, with allocation of the result. Fast because doesn't go through NumpyVectorT and doesn't allocate memory.");

        //PyObject* rightVecProd(PyObject* pyRows, PyObject *xIn) const
        sm.def("rightVecProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& Rows, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(Rows.size());
            sm.rightVecProd(get_it(Rows), get_end(Rows), get_it(xIn), get_it(y));
            return y;
        }, "Matrix vector product on the right side, only for some rows.");

        //SparseMatrix32 blockRightVecProd(nupic::UInt32 block_size, PyObject* xIn) const
        sm.def("blockRightVecProd", [](const SparseMatrix32_t& sm, nupic::UInt32 block_size, py::array_t<nupic::Real32>& xIn)
        {
            SparseMatrix32_t result;

            sm.blockRightVecProd(block_size, get_it(xIn), result);

            return result;
        });

        //nupic::Real32 leftVecProd(nupic::UInt32 col, PyObject *xIn) const
        sm.def("leftVecProd", [](const SparseMatrix32_t& sm, nupic::UInt32 col, py::array_t<nupic::Real32>& xIn)
        {
            return sm.leftVecProd(col, get_it(xIn));
        }, "Dot product of column col and vector xIn.");

        //PyObject* leftVecProd(PyObject *xIn) const
        sm.def("leftVecProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nCols());

            sm.leftVecProd(get_it(xIn), get_it(y));

            return y;
        }, "Vector matrix product on the left, i.e. dot product of xIn and each column of the matrix.");

        //PyObject* leftVecProd(PyObject* pyCols, PyObject *xIn) const
        sm.def("leftVecProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& Cols, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(Cols.size());

            sm.leftVecProd(get_it(Cols), get_end(Cols), get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* leftVecProd_binary(PyObject* pyCols, PyObject *xIn) const
        sm.def("leftVecProd_binary", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& Cols, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(Cols.size());

            sm.leftVecProd_binary(get_it(Cols), get_end(Cols), get_it(xIn), get_it(y));

            return y;
        }, "Binary search for the columns.");

        //PyObject* rightDenseMatProd(PyObject* mIn) const
        sm.def("rightDenseMatProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(sm.nRows(), m.nCols());

            sm.rightDenseMatProd(m, r);

            return r.get_py_array();
        });

        //PyObject* rightDenseMatProdAtNZ(PyObject* mIn) const
        sm.def("rightDenseMatProdAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(sm.nRows(), m.nCols());

            sm.rightDenseMatProdAtNZ(m, r);

            return r.get_py_array();
        });

        //PyObject* denseMatExtract(PyObject* mIn) const
        sm.def("denseMatExtract", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(sm.nRows(), m.nCols());

            sm.denseMatExtract(m, r);

            return r.get_py_array();
        });

        //PyObject* leftDenseMatProd(PyObject* mIn) const
        sm.def("leftDenseMatProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nCols());

            for (int i = 0; i != m.nRows(); ++i)
                sm.leftVecProd(m.get_row(i), r.get_row(i));

            return r.get_py_array();
        });

        //void elementRowAdd(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementRowAdd", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementRowAdd(i, get_it(xIn));
        });

        //void elementRowSubtract(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementRowSubtract", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementRowSubtract(i, get_it(xIn));
        });

        //void elementRowMultiply(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementRowMultiply", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementRowMultiply(i, get_it(xIn));
        });

        //void elementRowDivide(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementRowDivide", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementRowDivide(i, get_it(xIn));
        });

        //void elementColAdd(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementColAdd", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementColAdd(i, get_it(xIn));
        });

        //void elementColSubtract(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementColSubtract", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementColSubtract(i, get_it(xIn));
        });

        //void elementColMultiply(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementColMultiply", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementColMultiply(i, get_it(xIn));
        });

        //void elementColDivide(nupic::UInt32 i, PyObject* xIn)
        sm.def("elementColDivide", [](SparseMatrix32_t& sm, nupic::UInt32 i, py::array_t<nupic::Real32>& xIn)
        {
            sm.elementColDivide(i, get_it(xIn));
        });

        ////////////////////////
        // elementMultiply

        sm.def("elementMultiply", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            sm.elementMultiply(get_it(mIn));
        });

        sm.def("elementMultiply", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn, SparseMatrix32_t& result)
        {
            sm.elementMultiply(get_it(mIn), result);
        });

        sm.def("elementMultiply", [](SparseMatrix32_t& sm, const SparseMatrix32_t& m)
        {
            sm.elementMultiply(m);
        });

        sm.def("elementMultiply", [](const SparseMatrix32_t& sm, const SparseMatrix32_t& m, SparseMatrix32_t& result)
        {
            sm.elementMultiply(m, result);
        });


        //////--------------------------------------------------------------------------------
        ////// AtNZ operations, i.e. considering the sparse matrix as a 0/1 sparse matrix.
        //////--------------------------------------------------------------------------------

        //PyObject* rightVecProdAtNZ(PyObject* xIn) const
        sm.def("rightVecProdAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            sm.rightVecProdAtNZ(get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* leftVecProdAtNZ(PyObject* xIn) const
        sm.def("leftVecProdAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nCols());

            sm.leftVecProdAtNZ(get_it(xIn), get_it(y));

            return y;
        });

        ///////////////////////////////
        // rightVecSumAtNZ

        sm.def("rightVecSumAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray)
        {
            py::array_t<nupic::Real32> out(sm.nRows());

            sm.rightVecSumAtNZ(get_it(denseArray), get_it(out));

            return out;
        });

        sm.def("rightVecSumAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray, py::array_t<nupic::Real32>& out)
        {
            if (out.size() < sm.nRows())
            {
                throw std::runtime_error("Array is to small.");
            }

            sm.rightVecSumAtNZ(get_it(denseArray), get_it(out));

            return out;
        }, "", py::arg("denseArray"), py::arg("out"));


        ///////////////////////////////////
        // rightVecSumAtNZSparse

        sm.def("rightVecSumAtNZSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray)
        {
            py::array_t<nupic::Int32> out(sm.nRows());

            sm.rightVecSumAtNZSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out));

            return out;
        });

        sm.def("rightVecSumAtNZSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray, py::array_t<nupic::Int32>& out)
        {
            NTA_ASSERT(out.size() >= sm.nRows());

            sm.rightVecSumAtNZSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out));

            return out;
        }, "", py::arg("sparseBinaryArray"), py::arg("out"));


        ////////////////////////////
        // rightVecSumAtNZGtThreshold

        sm.def("rightVecSumAtNZGtThreshold", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray, nupic::Real32 threshold)
        {
            py::array_t<nupic::Real32> out( sm.nRows() );

            sm.rightVecSumAtNZGtThreshold(get_it(denseArray), get_it(out), threshold);

            return out;
        }, "Deprecated. Use rightVecSumAtNZGtThreshold with an 'out' specified.");

        sm.def("rightVecSumAtNZGtThreshold", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray, nupic::Real32 threshold
            , py::array_t<nupic::Real32>& out)
        {
            NTA_ASSERT(out.size() >= sm.nRows());

            sm.rightVecSumAtNZGtThreshold(get_it(denseArray), get_it(out), threshold);
        });


        ////////////////////////////
        // rightVecSumAtNZGtThresholdSparse

        sm.def("rightVecSumAtNZGtThresholdSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray, nupic::Real32 threshold
            , py::array_t<nupic::Int32>& out)
        {
            NTA_ASSERT(out.size() >= sm.nRows());

            sm.rightVecSumAtNZGtThresholdSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out), threshold);

            return out;
        });

        sm.def("rightVecSumAtNZGtThresholdSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray, nupic::Real32 threshold)
        {
            py::array_t<nupic::Int32> out( sm.nRows());

            sm.rightVecSumAtNZGtThresholdSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out), threshold);

            return out;
        });

        ////////////////////////////
        // rightVecSumAtNZGteThreshold

        sm.def("rightVecSumAtNZGteThreshold", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray, nupic::Real32 threshold
            , py::array_t<nupic::Real32>& out)
        {
            NTA_ASSERT(out.size() >= sm.nRows());

            sm.rightVecSumAtNZGteThreshold(get_it(denseArray), get_it(out), threshold);

            return out;
        });

        sm.def("rightVecSumAtNZGteThreshold", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& denseArray, nupic::Real32 threshold)
        {
            py::array_t<nupic::Real32> out( sm.nRows() );

            sm.rightVecSumAtNZGteThreshold(get_it(denseArray), get_it(out), threshold);

            return out;
        });

        /////////////////////////////
        // rightVecSumAtNZGteThresholdSparse

        sm.def("rightVecSumAtNZGteThresholdSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray, nupic::Real32 threshold
            , py::array_t<nupic::Int32>& out)
        {
            NTA_ASSERT(out.size() >= sm.nRows());

            sm.rightVecSumAtNZGteThresholdSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out), threshold);

            return out;
        });

        sm.def("rightVecSumAtNZGteThresholdSparse", [](const SparseMatrix32_t& sm, py::array_t<nupic::UInt32>& sparseBinaryArray, nupic::Real32 threshold)
        {
            py::array_t<nupic::Int32> out( sm.nRows() );

            sm.rightVecSumAtNZGteThresholdSparse(get_it(sparseBinaryArray), get_end(sparseBinaryArray), get_it(out), threshold);

            return out;
        });


        ////////////////////////////////////
        //


        //inline PyObject* leftVecSumAtNZ(PyObject* xIn) const
        sm.def("leftVecSumAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nCols());

            sm.leftVecSumAtNZ(get_it(xIn), get_it(y));

            return y;
        }, "Regular matrix vector multiplication on the left side, assuming that the values of the non-zeros are all 1, so that we can save actually computing the multiplications. Allocates the result.");

        //inline void leftVecSumAtNZ_fast(PyObject *xIn, PyObject *yOut) const
        sm.def("leftVecSumAtNZ_fast", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn, py::array_t<nupic::Real32>& yOut)
        {
            sm.leftVecSumAtNZ(get_it(xIn), get_it(yOut));
        }, "Regular matrix vector multiplication on the left, without allocation of the result, assuming that the values of the non-zeros are always 1 in the sparse matrix, so that we can save computing multiplications explicitly. Also fast because doesn't go through NumpyVectorT and doesn't allocate memory.");

        //PyObject* rightDenseMatProdAtNZ(PyObject* mIn) const
        sm.def("rightDenseMatProdAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nRows());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.rightVecProdAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* leftDenseMatProdAtNZ(PyObject* mIn) const
        sm.def("leftDenseMatProdAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nCols());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.leftVecProdAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* rightDenseMatSumAtNZ(PyObject* mIn) const
        sm.def("rightDenseMatSumAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nRows());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.rightVecSumAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* leftDenseMatSumAtNZ(PyObject* mIn) const
        sm.def("leftDenseMatSumAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nCols());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.leftVecSumAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* rightDenseMatMaxAtNZ(PyObject* mIn) const
        sm.def("rightDenseMatMaxAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nRows());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.rightVecMaxAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* leftDenseMatMaxAtNZ(PyObject* mIn) const
        sm.def("leftDenseMatMaxAtNZ", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& mIn)
        {
            Numpy_Matrix<nupic::Real32> m(mIn.request());
            Numpy_Matrix<nupic::Real32> r(m.nRows(), sm.nCols());

            for (int i = 0; i != m.nRows(); ++i)
            {
                sm.leftVecMaxAtNZ(m.get_row(i), r.get_row(i));
            }

            return r.get_py_array();
        });

        //PyObject* vecArgMaxAtNZ(PyObject *xIn)
        sm.def("vecArgMaxAtNZ", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::UInt32> y(sm.nRows());

            sm.vecArgMaxAtNZ(get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* vecMaxAtNZ(PyObject *xIn)
        sm.def("vecMaxAtNZ", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            sm.vecMaxAtNZ(get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* rowVecProd(PyObject* xIn, nupic::Real32 lb = nupic::Epsilon) const
        sm.def("rowVecProd", [](const SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn, nupic::Real32 lb)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            sm.rowVecProd(get_it(xIn), get_it(y));

            return y;
        }, "", py::arg("xIn"), py::arg("lb") = nupic::Epsilon);

        //PyObject* vecMaxProd(PyObject *xIn)
        sm.def("vecMaxProd", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            sm.vecMaxProd(get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* vecArgMaxProd(PyObject *xIn)
        sm.def("vecArgMaxProd", [](SparseMatrix32_t& sm, py::array_t<nupic::Real32>& xIn)
        {
            py::array_t<nupic::UInt32> y(sm.nRows());

            sm.vecArgMaxProd(get_it(xIn), get_it(y));

            return y;
        });

        //PyObject* getNonZerosSorted(nupic::Int n = -1, bool ascending_values = true) const
        sm.def("getNonZerosSorted", [](const SparseMatrix32_t& sm, nupic::Int n, bool ascending_values)
        {
            typedef nupic::ijv<nupic::UInt32, nupic::Real32> IJV;

            nupic::UInt32 nnz = sm.nNonZeros();
            nupic::UInt32 N = n == -1 ? nnz : n;
            std::vector<IJV> ijvs(N);
            if (ascending_values)
            {
                sm.getNonZerosSorted(ijvs.begin(), N, IJV::greater_value());
            }
            else
            {
                sm.getNonZerosSorted(ijvs.begin(), N, IJV::less_value());
            }

            py::tuple toReturn(N);

            for (nupic::UInt32 i = 0; i != N; ++i)
            {
                toReturn[i] = py::make_tuple(ijvs[i].i(), ijvs[i].j(), ijvs[i].v());
            }

            return toReturn;
        }, "", py::arg("n") = -1, py::arg("ascending_values") = true);

        //PyObject* threshold(nupic::Real32 threshold, bool getCuts = false)
        sm.def("threshold", [](SparseMatrix32_t& sm, nupic::Real32 threshold, bool getCuts)
        {
            if (!getCuts)
            {
                sm.threshold(threshold);
                return py::tuple();
            }

            std::vector<nupic::UInt32> cut_i, cut_j;
            std::vector<nupic::Real32> cut_nz;
            nupic::UInt32 c = 0;
            c = sm.threshold(threshold,
                std::back_inserter(cut_i),
                std::back_inserter(cut_j),
                std::back_inserter(cut_nz));

            py::tuple toReturn(c);

            for (nupic::UInt32 i = 0; i != c; ++i)
            {
                toReturn[i] = py::make_tuple(cut_i[i], cut_j[i], cut_nz[i]);
            }

            return toReturn;
        }, "", py::arg("threshold"), py::arg("getCuts") = false);

        //PyObject* toPyString() const
        sm.def("toPyString", [](const SparseMatrix32_t& sm)
        {
            std::stringstream s;

            sm.toCSR(s);

            return s.str();
        });

        //bool fromPyString(PyObject *s)
        sm.def("fromPyString", [](SparseMatrix32_t& sm, const std::string& s)
        {
            if (s.empty() == false)
            {
                std::istringstream ss(s);
                sm.fromCSR(ss);
                return true;
            }

            throw std::runtime_error("Failed to read SparseMatrix state from string.");
            return false;
        });


        sm.def("__eq__", [](const SparseMatrix32_t& sm, const SparseMatrix32_t& other)
        {
            return sm == other;
        });

        sm.def("__ne__", [](const SparseMatrix32_t& sm, const SparseMatrix32_t& other)
        {
            return sm != other;
        });

        //////////////////////

        sm.def("initializeWithFixedNNZR", [](SparseMatrix32_t& self, nupic::UInt32 nnzr, nupic::Real32 v, nupic::UInt32 mode, nupic::UInt32 seed)
        {
            self.initializeWithFixedNNZR(nnzr, v, mode, seed);
        }, "Initialize a sparse matrix with a fixed number of non-zeros on each row."
            , py::arg("nnzr"), py::arg("v") = 1.0, py::arg("mode") = 0, py::arg("seed") = 42);
    }

} // namespace nupic_ext
