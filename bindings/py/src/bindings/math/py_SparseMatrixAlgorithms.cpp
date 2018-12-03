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
PyBind11 bindings for SparseMatrixAlgorithms classes
*/


#include <fstream>

// the use of 'register' keyword is removed in C++17
// Python2.7 uses 'register' in unicodeobject.h
#ifdef _WIN32
#pragma warning( disable : 5033)  // MSVC
#else
#pragma GCC diagnostic ignored "-Wregister"  // for GCC and CLang
#endif

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseBinaryMatrix.hpp>
#include <nupic/math/SparseMatrixAlgorithms.hpp>

#include "bindings/engine/py_utils.hpp"

namespace py = pybind11;

namespace nupic_ext
{
    void init_SparseMatrixAlgorithms(py::module& m)
    {
        //!!! Not all are SparseMatrixAlgorithms !!!
        // see sparse_matrix.i


        typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32>> SparseMatrix32_t;
        typedef nupic::SparseBinaryMatrix<nupic::UInt32, nupic::UInt32> SparseBinaryMatrix32_t;


        //PyObject* kthroot_product(const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > > & sm, nupic::UInt32 segment_size, PyObject* xIn, nupic::Real32 threshold)
        m.def("kthroot_product", [](const SparseMatrix32_t& sm, nupic::UInt32 segment_size, py::array_t<nupic::Real32>& xIn, nupic::Real32 threshold)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            nupic::SparseMatrixAlgorithms::kthroot_product(sm, segment_size, get_it(xIn), get_it(y), threshold);

            return y;
        });


        //PyObject* sparseRightVecProd(const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& a, nupic::UInt32 m, nupic::UInt32 n, PyObject* x)
        m.def("sparseRightVecProd", [](const SparseMatrix32_t& sm, nupic::UInt32 m, nupic::UInt32 n, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            nupic::SparseMatrixAlgorithms::sparseRightVecProd(sm, m, n, get_it(x), get_it(y));

            return y;
        });


        //inline bool isZero_01(PyObject* py_x)
        m.def("isZero_01", [](py::array_t<nupic::Real32>& x)
        {
            return nupic::isZero_01(get_it(x), get_end(x));
        }, "A function that decide if a binary 0/1 vector is all zeros, or not.");


        //inline nupic::Real32 dense_vector_sum(PyObject* py_x)
        m.def("dense_vector_sum", [](py::array_t<nupic::Real32>& x)
        {
            return nupic::sum(get_it(x), get_end(x));

        }, "A function that sums the elements in a dense range, faster than numpy and C++.");


        ////--------------------------------------------------------------------------------
        //// Functions on 2D dense arrays of 0/1
        ////--------------------------------------------------------------------------------

        //inline PyObject* nonZeroRowsIndicator_01(nupic::UInt32 nrows, nupic::UInt32 ncols, PyObject* py_x)
        m.def("nonZeroRowsIndicator_01", [](nupic::UInt32 nrows, nupic::UInt32 ncols, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::UInt32> ind(nrows);

            nupic::nonZeroRowsIndicator_01(nrows, ncols, get_it(x), get_end(x), get_it(ind), get_end(ind));

            return ind;
        });

        //inline PyObject* nonZeroColsIndicator_01(nupic::UInt32 nrows, nupic::UInt32 ncols, PyObject* py_x)
        m.def("nonZeroColsIndicator_01", [](nupic::UInt32 nrows, nupic::UInt32 ncols, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::UInt32> ind(ncols);

            nupic::nonZeroColsIndicator_01(nrows, ncols, get_it(x), get_end(x), get_it(ind), get_end(ind));

            return ind;
        });


        //inline nupic::UInt32 nNonZeroRows_01(nupic::UInt32 nrows, nupic::UInt32 ncols, PyObject* py_x)
        m.def("nNonZeroRows_01", [](nupic::UInt32 nrows, nupic::UInt32 ncols, py::array_t<nupic::Real32>& x)
        {
            return nupic::nNonZeroRows_01(nrows, ncols, get_it(x), get_end(x));
        });


        //inline nupic::UInt32 nNonZeroCols_01(nupic::UInt32 nrows, nupic::UInt32 ncols, PyObject* py_x)
        m.def("nNonZeroCols_01", [](nupic::UInt32 nrows, nupic::UInt32 ncols, py::array_t<nupic::Real32>& x)
        {
            return nupic::nNonZeroCols_01(nrows, ncols, get_it(x), get_end(x));
        });


        //PyObject* matrix_entropy(const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& sm, nupic::Real32 s = 1.0)
        m.def("matrix_entropy", [](const SparseMatrix32_t& sm, nupic::Real32 s)
        {
            py::array_t<nupic::Real32> e_rows(sm.nRows());
            py::array_t<nupic::Real32> e_cols(sm.nRows());

            nupic::SparseMatrixAlgorithms::matrix_entropy(sm,
                get_it(e_rows), get_end(e_rows),
                get_it(e_cols), get_end(e_cols),
                s);

            return py::make_tuple(e_rows, e_cols);
        }, "", py::arg("sm"), py::arg("s") = 1.0);


        //PyObject* smoothVecMaxProd(const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& sm, nupic::Real32 k, PyObject *xIn)
        m.def("smoothVecMaxProd", [](const SparseMatrix32_t& sm, nupic::Real32 k, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            nupic::SparseMatrixAlgorithms::smoothVecMaxProd(sm, k, get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        });


        //PyObject* smoothVecArgMaxProd(const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& sm, nupic::Real32 k, PyObject *xIn)
        m.def("smoothVecArgMaxProd", [](const SparseMatrix32_t& sm, nupic::Real32 k, py::array_t<nupic::Real32>& x)
        {
            py::array_t<nupic::Real32> y(sm.nRows());

            nupic::SparseMatrixAlgorithms::smoothVecArgMaxProd(sm, k, get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        });


        //--------------------------------------------------------------------------------
        // LBP
        //--------------------------------------------------------------------------------

        //void LBP_piPrime(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& mat, nupic::Real32 min_floor = 0)
        m.def("LBP_piPrime", [](SparseMatrix32_t& mat, nupic::Real32 min_floor)
        {
            nupic::SparseMatrixAlgorithms::LBP_piPrime(mat, min_floor);
        }, "", py::arg("mat"), py::arg("min_floor") = 1.0);


        //void SM_subtractNoAlloc(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& B
        // , double min_floor = 0)
        m.def("SM_subtractNoAlloc", [](SparseMatrix32_t& A, const SparseMatrix32_t& B, nupic::Real32 min_floor)
        {
            nupic::SparseMatrixAlgorithms::subtractNoAlloc(A, B, min_floor);
        }, "", py::arg("A"), py::arg("B"), py::arg("min_floor") = 1.0);


        //void SM_assignNoAlloc(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& B)
        m.def("SM_assignNoAlloc", [](SparseMatrix32_t& A, const SparseMatrix32_t& B)
        {
            nupic::SparseMatrixAlgorithms::assignNoAlloc(A, B);
        });

        //void SM_assignNoAllocFromBinary(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , const nupic::SparseBinaryMatrix<nupic::UInt32, nupic::UInt32>& B)
        m.def("SM_assignNoAllocFromBinary", [](SparseMatrix32_t& A, const SparseBinaryMatrix32_t& B)
        {
            nupic::SparseMatrixAlgorithms::assignNoAllocFromBinary(A, B);
        });


        //void SM_addConstantOnNonZeros(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , const nupic::SparseBinaryMatrix<nupic::UInt32, nupic::UInt32>& B, double cval)
        m.def("SM_addConstantOnNonZeros", [](SparseMatrix32_t& A, const SparseBinaryMatrix32_t& B, double cval)
        {
            nupic::SparseMatrixAlgorithms::addConstantOnNonZeros(A, B, static_cast<float>(cval));
        });


        //void SM_logSumNoAlloc(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , const nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& B
        // , double min_floor = 0)
        m.def("SM_logSumNoAlloc", [](SparseMatrix32_t& A, const SparseMatrix32_t& B, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::logSumNoAlloc(A, B, static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("B"), py::arg("min_floor") = 0);


        //void SM_logAddValNoAlloc(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , double val, double min_floor = 0)
        m.def("SM_logAddValNoAlloc", [](SparseMatrix32_t& A, double val, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::logAddValNoAlloc(A, static_cast<float>(val), static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("val"), py::arg("min_floor") = 0);


        //void SM_logDiffNoAlloc(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& B, double min_floor = 0)
        m.def("SM_logDiffNoAlloc", [](SparseMatrix32_t& A, const SparseMatrix32_t& B, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::logDiffNoAlloc(A, B, static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("B"), py::arg("min_floor") = 0);


        //void SM_addToNZOnly(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , double v, double min_floor = 0)
        m.def("SM_addToNZOnly", [](SparseMatrix32_t& A, double v, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::addToNZOnly(A, v, static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("v"), py::arg("min_floor") = 0);


        //void SM_addToNZDownCols(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , PyObject* py_x, double min_floor = 0)
        m.def("SM_addToNZDownCols", [](SparseMatrix32_t& A, py::array_t<nupic::Real32>& x, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::addToNZDownCols(A, get_it(x), get_end(x), static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("x"), py::arg("min_floor") = 0);


        //void SM_addToNZAcrossRows(nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32 > >& A
        // , PyObject* py_x, double min_floor = 0)
        m.def("SM_addToNZAcrossRows", [](SparseMatrix32_t& A, py::array_t<nupic::Real32>& x, double min_floor)
        {
            nupic::SparseMatrixAlgorithms::addToNZAcrossRows(A, get_it(x), get_end(x), static_cast<float>(min_floor));
        }, "", py::arg("A"), py::arg("x"), py::arg("min_floor") = 0);


        //// For unit testing

        //nupic::LogSumApprox
        py::class_<nupic::LogSumApprox> py_LogSumApprox(m, "LogSumApprox");

        py_LogSumApprox.def(py::init<int, float, float, bool>()
            , py::arg("n_") = 5000000, py::arg("min_a_") = -28, py::arg("max_a_") = 28, py::arg("trace_") = false);

        //inline nupic::Real32 logSum(nupic::Real32 x, nupic::Real32 y) const
        py_LogSumApprox.def("logSum", [](const nupic::LogSumApprox& self, nupic::Real32 x, nupic::Real32 y)
        {
            return self.sum_of_logs(x, y);
        });


        //inline nupic::Real32 fastLogSum(nupic::Real32 x, nupic::Real32 y) const
        m.def("fastLogSum", [](const nupic::LogSumApprox& self, nupic::Real32 x, nupic::Real32 y)
        {
            return self.fast_sum_of_logs(x, y);
        });

        // nupic::LogDiffApprox
        py::class_<nupic::LogDiffApprox> py_LogDiffApprox(m, "LogDiffApprox");

        py_LogDiffApprox.def(py::init<int, float, float, bool>()
            , py::arg("n_") = 5000000, py::arg("min_a_") = 1e-10, py::arg("max_a_") = 28, py::arg("trace_") = false);


        //inline nupic::Real32 logDiff(nupic::Real32 x, nupic::Real32 y) const
        py_LogDiffApprox.def("logDiff", [](const nupic::LogDiffApprox& self, nupic::Real32 x, nupic::Real32 y)
        {
            return self.diff_of_logs(x, y);
        });


        //inline nupic::Real32 fastLogDiff(nupic::Real32 x, nupic::Real32 y) const
        py_LogDiffApprox.def("fastLogDiff", [](const nupic::LogDiffApprox& self, nupic::Real32 x, nupic::Real32 y)
        {
            return self.fast_diff_of_logs(x, y);
        });


    }
} // namespace nupic_ext
