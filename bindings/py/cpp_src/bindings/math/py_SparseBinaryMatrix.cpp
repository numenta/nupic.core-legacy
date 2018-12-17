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
PyBind11 bindings for SparseBinaryMatrix class
*/


#include <bindings/suppress_register.hpp>  //include before pybind11.h
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nupic/math/Math.hpp>
#include <nupic/math/DenseMatrix.hpp>
#include <nupic/math/Domain.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseBinaryMatrix.hpp>

#include "Matrix.hpp"
#include "bindings/engine/py_utils.hpp"
#include <fstream>

namespace py = pybind11;

typedef nupic::Domain<nupic::UInt32> _Domain32;
typedef nupic::Domain2D<nupic::UInt32> _Domain2D32;
typedef nupic::DistanceToZero<nupic::Real32> _DistanceToZero32;

typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real32, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real32>> SparseMatrix32_t;
typedef nupic::SparseMatrix<nupic::UInt32, nupic::Real64, nupic::Int32, nupic::Real64, nupic::DistanceToZero<nupic::Real64>> _SparseMatrix64;


typedef nupic::SparseBinaryMatrix<nupic::UInt32, nupic::UInt32> SM_01_32_32_t;


namespace nupic_ext
{
    void init_SM_01_32_32(py::module& m)
    {
        py::class_<SM_01_32_32_t> sbm(m, "SM_01_32_32");

        // create an alias for SM_01_32_32
        m.attr("SparseBinaryMatrix") = sbm;

        sbm.def(py::init<>())
           .def(py::init<nupic::UInt32>(), py::arg("ncols"))
           .def(py::init<nupic::UInt32, nupic::UInt32>(), py::arg("nrows"), py::arg("ncols"))
           ;

        // constructor from SparseMatrix
        sbm.def(py::init(
            [](const SparseMatrix32_t& sm)
        {
            auto nnz = sm.nNonZeros();
            std::vector<nupic::UInt> rows(nnz);
            std::vector<nupic::UInt> cols(nnz);
            std::vector<nupic::Real> vals(nnz);

            sm.getAllNonZeros(rows.begin(), cols.begin(), vals.begin());

            SM_01_32_32_t s(1);
            s.setAllNonZeros(sm.nRows(), sm.nCols(), rows.begin(), rows.end(), cols.begin(), cols.end());

            return s;
        }));


        // constructor from numpy array
        sbm.def(py::init(
            [](py::array_t<nupic::UInt32>& a)
            {
                if (a.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

                SM_01_32_32_t s(1);
                s.fromDense(a.shape(0), a.shape(1), get_it(a), get_end(a));

                return s;
            }));

        sbm.def(py::init(
            [](const std::string& str)
        {
            std::istringstream istr(str);

            SM_01_32_32_t s;
            s.fromCSR(istr);

            return s;
        }));


        // copy constructor
        sbm.def(py::init([](const SM_01_32_32_t& o) { return new SM_01_32_32_t(o); }));

        // getter members
        sbm.def("nRows", &SM_01_32_32_t::nRows);

		// Overload conflict in SparseBinaryMatrix.hpp
		//  149  inline nz_index_type nCols() const // this one is public  ( nz_index_type is UInt32)
		// 1792  inline void nCols(size_type ncols) // this one is private
		// We only want to expose the public function.
		// pybind11 thinks these are overloaded functions and wants to see both.
		// So, changing the name of the private function to setnCols(ncols)
        //sbm.def("nCols", py::overload_cast<>(&SM_01_32_32_t::nCols, py::const_)); // C++14+
        //sbm.def("nCols", (void (SM_01_32_32_t::*)(nupic::UInt32)) &SM_01_32_32_t::nCols); // private
		sbm.def("nCols", &SM_01_32_32_t::nCols);

        sbm.def("capacity", &SM_01_32_32_t::capacity);

        sbm.def("getVersion", &SM_01_32_32_t::getVersion, py::arg("binary") = false);

        sbm.def("nBytes", &SM_01_32_32_t::nBytes);

        sbm.def("nNonZeros", &SM_01_32_32_t::nNonZeros);
        sbm.def("nNonZerosOnRow", &SM_01_32_32_t::nNonZerosOnRow);

        sbm.def("nNonZerosPerRow", [](const SM_01_32_32_t& sbm)
            {
                typedef py::array_t<nupic::UInt32> out_t;

                if (sbm.nRows() == 0) { return out_t(); }

                auto out = out_t(sbm.nRows());
                sbm.nNonZerosPerRow(get_it(out), get_end(out));

                return out;
            });

        sbm.def("nNonZerosPerCol", [](const SM_01_32_32_t& sbm)
            {
                typedef py::array_t<nupic::UInt32> out_t;

                if (sbm.nCols() == 0) { return out_t(); }

                auto out = out_t(sbm.nCols());
                sbm.nNonZerosPerCol(get_it(out), get_end(out));

                return out;
            });


        // members

        //////////////////
        // appendDenseRow
        //
        // Input: vector of unsigned ints
        // Output: None
        sbm.def("appendDenseRow",
            [](SM_01_32_32_t& sbm, const std::vector<nupic::UInt32>& a)
        {
            sbm.appendDenseRow(a.begin(), a.end());
        });

        sbm.def("appendDenseRow",
            [](SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& a)
        {
            sbm.appendDenseRow(get_it(a), get_end(a));
        });


        sbm.def("appendSparseRow", [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a)
        {
            if (a.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            sbm.appendSparseRow(get_it(a), get_end(a));
        });

        sbm.def("appendSparseCol", [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a)
        {
            if (a.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            sbm.appendSparseCol(get_it(a), get_end(a));
        });


        sbm.def("appendEmptyCols", &SM_01_32_32_t::appendEmptyCols);



        sbm.def("binaryLoadFromFile",
            [](SM_01_32_32_t& sbm, const std::string filename)
        {
            std::ifstream load_file(filename.c_str());
            sbm.fromBinary(load_file);
            load_file.close();
        });


        sbm.def("binarySaveToFile",
            [](const SM_01_32_32_t& sbm, const std::string filename)
        {
            std::ofstream save_file(filename.c_str());
            sbm.toBinary(save_file);
            save_file.close();
        });


        sbm.def("copy", &SM_01_32_32_t::copy);

        sbm.def("clear", &SM_01_32_32_t::clear);

        sbm.def("compact", &SM_01_32_32_t::compact);

        sbm.def("CSRLoadFromFile",
            [](SM_01_32_32_t& sbm, const std::string filename)
        {
            std::ifstream load_file(filename.c_str());
            sbm.fromCSR(load_file);
            load_file.close();
        });


        sbm.def("CSRSaveToFile",
            [](const SM_01_32_32_t& sbm, const std::string filename)
        {
            std::ofstream save_file(filename.c_str());
            sbm.toCSR(save_file);
            save_file.close();
        });

        sbm.def("CSRSize", &SM_01_32_32_t::CSRSize);


        //////////////////
        // findRowDense
		// findRowSparse
		//////////////////
        sbm.def("findRowDense",
            [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a)
        {
            auto it = (nupic::UInt32*) a.request().ptr;
            auto end = it + a.size();

            return sbm.findRowDense(it, end);
        });

        sbm.def("findRowSparse",
            [](SM_01_32_32_t& sbm, const std::vector<nupic::UInt32>& a)
        {
            return sbm.findRowSparse(a.begin(), a.end());
        });


        //////////////////
        // fromDense
        //////////////////
        sbm.def("fromDense",
            [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a)
        {
            sbm.fromDense(a.shape(0), a.shape(1), get_it(a), get_end(a));
        });

        //////////////////////
        // firstRowCloserThan
        sbm.def("firstRowCloserThan",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a, nupic::UInt32 distance) -> nupic::UInt32
        {
            if (a.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            return sbm.firstRowCloserThan(get_it(a), get_end(a), distance);
        });


        //////////////////
        // fromCSR
        //
        sbm.def("fromCSR",
            [](SM_01_32_32_t& sbm, std::string str)
        {
            if (str.empty() == false)
            {
                std::istringstream is(str);

                sbm.fromCSR(is);
            }
            else
            {
                throw std::runtime_error("Failed to read SparseBinaryMatrix state from string.");
            }

        });

        /////////////////
        // fromSparseVector
        //
        sbm.def("fromSparseVector",
            [](SM_01_32_32_t& sbm, nupic::UInt32 nrows, nupic::UInt32 ncols, std::vector<nupic::UInt32> indices, nupic::UInt32 offset)
        {
            sbm.fromSparseVector(nrows, ncols, indices.begin(), indices.end(), offset);

        }, "", py::arg("nrows"), py::arg("ncols"), py::arg("indices"), py::arg("offset") = 0);


        sbm.def("get", &SM_01_32_32_t::get);

        sbm.def("getAllNonZeros", [](const SM_01_32_32_t& sbm, bool two_lists)
        {
            const nupic::UInt32 nnz = sbm.nNonZeros();

            py::array_t<nupic::UInt32> rows(nnz);
            py::array_t<nupic::UInt32> cols(nnz);

            //auto rows_it = (nupic::UInt32*) rows.request().ptr;
            //auto cols_it = (nupic::UInt32*) cols.request().ptr;

            sbm.getAllNonZeros(get_it(rows), get_it(cols));

            py::tuple t;

            if (two_lists == false)
            {
                t = py::tuple(nnz);

                for (nupic::UInt32 i = 0; i != nnz; ++i)
                {
                    nupic::UInt32 r = *rows.data(i);
                    nupic::UInt32 c = *cols.data(i);

                    t[i] = py::make_tuple(r, c);
                }
            }
            else
            {
                t = py::make_tuple(rows, cols);
            }

            return t;
        }, "Returns the indices (i,j) of all the non-zeros, in lexicographic order"
			, py::arg("two_lists") = false);


        sbm.def("getCol",
            [](const SM_01_32_32_t& sbm, nupic::UInt32 col)
        {
            py::array_t<nupic::UInt32> dense_col(sbm.nRows());

			sbm.getColToDense(col, get_it(dense_col), get_end(dense_col));

            return dense_col;
        }, "Returns a dense column as numpy array.");

        //////////////////////
        // edges
        /////////////////////
        sbm.def("edges", &SM_01_32_32_t::edges);


        //////////////////////
        // HammingDistance
        /////////////////////
        sbm.def("minHammingDistance", [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& a)
        {
            if (a.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            auto r = sbm.minHammingDistance(get_it(a), get_end(a));

            py::tuple t = py::make_tuple(r.first, r.second);

            return t;
        });


        //////////////////////
        // inside
        /////////////////////
        sbm.def("inside", &SM_01_32_32_t::inside);

        //////////////////////
        // logicalXXX
        /////////////////////
        sbm.def("logicalNot", &SM_01_32_32_t::logicalNot);
        sbm.def("logicalOr", &SM_01_32_32_t::logicalOr);
        sbm.def("logicalAnd", &SM_01_32_32_t::logicalAnd);

        ////////////////////
        // nNonZerosPerBox
        ////////////////////
        sbm.def("nNonZerosPerBox", [](SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& box_i, py::array_t<nupic::UInt32>& box_j)
        {
            if (box_i.ndim() != 1) { throw std::runtime_error("Number of dimensions must be two."); }
            if (box_j.ndim() != 1) { throw std::runtime_error("Number of dimensions must be two."); }

            SparseMatrix32_t result(box_i.size(), box_j.size());

            sbm.nNonZerosPerBox(get_it(box_i), get_end(box_i), get_it(box_j), get_end(box_j), result);

            return result;
        });


        sbm.def("randomInitialize", &SM_01_32_32_t::randomInitialize);

        //////////////////////
        // overlap
		// maxAllowedOverlap
        /////////////////////
        sbm.def("overlap",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::UInt32>& x)
        {
            py::array_t<nupic::UInt32> y(sbm.nRows());

            sbm.overlap(get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        });

        sbm.def("maxAllowedOverlap",
            [](const SM_01_32_32_t& sbm, nupic::Real32 maxDistance, py::array_t<nupic::UInt32>& py_x)
        {
            auto x_it = (nupic::UInt32*) py_x.request().ptr;
            auto x_end = x_it + py_x.size();

            return sbm.maxAllowedOverlap(maxDistance, x_it, x_end);
        });



		//////////////////////
        // replaceSparseRow
        //////////////////////
        sbm.def("replaceSparseRow",
            [](SM_01_32_32_t& sbm, const nupic::UInt32 row, const std::vector<nupic::UInt32>& a)
        {
            sbm.replaceSparseRow(row, a.begin(), a.end());
        });

        sbm.def("resize", &SM_01_32_32_t::resize);

        ////////////////
        // rowFromDense
        ////////////////
        sbm.def("rowFromDense",
            [](SM_01_32_32_t& sbm, const nupic::UInt32 row, py::array_t<nupic::Real32>& a)
        {
            if (a.ndim() != 1) { throw std::runtime_error("Number of dimensions must be two."); }

			sbm.rowFromDense(row, get_it(a), get_end(a));
        });


        ////////////////
        // rowToDense
        ////////////////
        sbm.def("rowToDense",
            [](const SM_01_32_32_t& sbm, const nupic::UInt32 row)
        {
            py::array_t<nupic::Real32> a(sbm.nCols());

            sbm.rowToDense(row, get_it(a), get_end(a));

            return a;
        });


		////////////////
		// set
		// setAllNonZeros
		////////////////
        sbm.def("set", [](SM_01_32_32_t& sbm, nupic::UInt32 row, nupic::UInt32 col, nupic::Real32 val)
        {
            sbm.set(row, col, val);
        });

        sbm.def("set", [](SM_01_32_32_t& sbm, nupic::UInt32 row, const std::vector<nupic::UInt32>& indices, nupic::Real32 val)
        {
            sbm.set(row, indices.begin(), indices.end(), val);
        });

        sbm.def("setAllNonZeros",
            [](SM_01_32_32_t& sbm, nupic::UInt32 nrows,
                nupic::UInt32 ncols, py::array_t<nupic::UInt32>& i, py::array_t<nupic::UInt32>& j, bool sorted)
        {
            if (i.ndim() != 1 || j.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            sbm.setAllNonZeros(nrows, ncols, get_it(i), get_end(i), get_it(j), get_end(j), sorted);

        }, "Clear this instance and create a new one that has non-zeros only at the positions passed in.",
            py::arg("nrows"), py::arg("ncols"), py::arg("py_i"), py::arg("py_j"), py::arg("sorted") = true);

        sbm.def("setForAllRows", [](SM_01_32_32_t& sbm, const std::vector<nupic::UInt32>& indices, nupic::Real32 val)
        {
            sbm.setForAllRows(indices.begin(), indices.end(), val);
        });


        sbm.def("setRangeToZero", &SM_01_32_32_t::setRangeToZero);
        sbm.def("setRangeToOne", &SM_01_32_32_t::setRangeToOne);

        //////////////////////
        // sums
        /////////////////////
        sbm.def("rowSums", [](const SM_01_32_32_t& sbm)
        {
            py::array_t<nupic::UInt32> a(sbm.nRows());

            sbm.rowSums(get_it(a), get_end(a));

            return a;
        });

        sbm.def("colSums", [](const SM_01_32_32_t& sbm)
        {
            py::array_t<nupic::UInt32> a(sbm.nCols());

            sbm.colSums(get_it(a), get_end(a));

            return a;
        });

        //////////////////////
        // vecMaxProd
        /////////////////////
        sbm.def("vecMaxProd",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(sbm.nRows());

            sbm.vecMaxProd(get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        });

        sbm.def("setSlice", [](SM_01_32_32_t& sbm, nupic::UInt32 i_begin, nupic::UInt32 j_begin, SM_01_32_32_t& other )
        {
            sbm.setSlice(i_begin, j_begin, other);
        }, "Set a slice at dst_first_row, dst_first_col, whose shape and contents are src.");

        sbm.def("setSlice", [](SM_01_32_32_t& sbm, nupic::UInt32 i_begin, nupic::UInt32 j_begin, py::array_t<nupic::Real32>& other)
        {
            if (other.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

            nupic_ext::Numpy_Matrix<> nm(other.request());
            sbm.setSlice(i_begin, j_begin, nm);
        }, "Set a slice at dst_first_row, dst_first_col, whose shape and contents are src.");


        sbm.def("toDense",
            [](const SM_01_32_32_t& sbm) -> py::array_t<nupic::UInt32>
            {
			    py::array_t<nupic::UInt32> a({ sbm.nRows(), sbm.nCols() });

                sbm.toDense(get_it(a), get_end(a));

				return a;
            });

        sbm.def("toCSR", [](const SM_01_32_32_t& sbm)
            {
                std::ostringstream s;
                sbm.toCSR(s);

                return s.str();
            });

        /////////////////
        // toSparseVector
        //
        sbm.def("toSparseVector",
            [](const SM_01_32_32_t& sbm, nupic::UInt32 offset)
        {
            py::array_t<nupic::UInt32> a(sbm.nNonZeros());

            sbm.toSparseVector(get_it(a), get_end(a), offset);

            return a;

        }, "", py::arg("offset") = 0);


        /////////////////////
        sbm.def("getRow", [](const SM_01_32_32_t& sbm, nupic::UInt32 row)
        {
            py::array_t<nupic::UInt32> x( sbm.nCols());
            sbm.getRow(row, get_it(x), get_end(x));

            return x;
        });



		///////////////////////////
		// transpose
		//////////////////////
		sbm.def("transpose", &SM_01_32_32_t::transpose);


        ///////////////////////////
        // rightVecSumAtNZ
        // rightVecSumAtNZ_fast
        // leftVecSumAtNZ
        // leftVecSumAtNZ_fast
        //
        // rightVecArgMaxAtNZ
        // leftVecMaxAtNZ
        //
        // leftDenseMatSumAtNZ
        // leftDenseMatMaxAtNZ
        ///////////////////////////
        sbm.def("rightVecSumAtNZ",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x) -> py::array_t<nupic::Real32>
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(sbm.nRows());

            sbm.rightVecSumAtNZ(get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        }, "Matrix vector multiplication on the right side, optimized to skip the actual multiplications, because all the non-zeros have the same value: 1.");

        sbm.def("rightVecSumAtNZ_fast",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x, py::array_t<nupic::Real32>& y)
        {
            if (x.ndim() != 1 || y.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            sbm.rightVecSumAtNZ(get_it(x), get_end(x), get_it(y), get_end(y));
        }, "Same as rightVecSumAtNZ, but doesn't allocate its result, assumes that the vector of the correct size (number of columns) is passed in for the result.");


        sbm.def("leftVecSumAtNZ",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x) -> py::array_t<nupic::Real32>
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(sbm.nCols());

            sbm.leftVecSumAtNZ(get_it(x), get_end(x), get_it(y), get_end(y));

            return y;
        }, "Matrix vector multiplication on the left side, optimized to skip the actual multiplications, because all the non-zeros have the same value: 1.");

        sbm.def("leftVecSumAtNZ_fast",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x, py::array_t<nupic::Real32>& y)
        {
            if (x.ndim() != 1 || y.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            sbm.leftVecSumAtNZ(get_it(x), get_end(x), get_it(y), get_end(y));
        }, "Same as leftVecSumAtNZ, but doesn't allocate its result, assumes that the vector of the correct size (number of rows) is passed in for the result.");


        sbm.def("rightVecArgMaxAtNZ",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& x)
        {
            if (x.ndim() != 1) { throw std::runtime_error("Number of dimensions must be one."); }

            py::array_t<nupic::Real32> y(sbm.nRows());

            sbm.rightVecArgMaxAtNZ(get_it(x), get_it(y));

            return y;
        });

        sbm.def("leftDenseMatSumAtNZ",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& m)
        {
            if (m.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

            py::array_t<nupic::Real32> r({ (nupic::UInt32) m.shape(0), sbm.nCols() });

            auto m_buf_info = m.request();
            auto r_buf_info = r.request();

            auto m_data = (char*)m_buf_info.ptr;
            auto r_data = (char*)r_buf_info.ptr;

            for (int i = 0; i < m.shape(0); ++i)
            {
                auto m_row_it = (nupic::Real32*) (m_data + i * m_buf_info.strides[0]);
                auto m_row_end = m_row_it + m.shape(1);

                auto r_row_it = (nupic::Real32*) (r_data + i * r_buf_info.strides[0]);
                auto r_row_end = r_row_it + r.shape(1);

                sbm.leftVecSumAtNZ(m_row_it, m_row_end, r_row_it, r_row_end);
            }

            return r;
        });


        sbm.def("leftDenseMatMaxAtNZ",
            [](const SM_01_32_32_t& sbm, py::array_t<nupic::Real32>& m)
        {
            if (m.ndim() != 2) { throw std::runtime_error("Number of dimensions must be two."); }

            py::array_t<nupic::Real32> r({ (nupic::UInt32) m.shape(0), sbm.nCols() });

            auto m_buf_info = m.request();
            auto r_buf_info = r.request();

            auto m_data = (char*)m_buf_info.ptr;
            auto r_data = (char*)r_buf_info.ptr;

            for (int i = 0; i != m.shape(0); ++i)
            {
                auto m_row_it = (nupic::Real32*) (m_data + i * m_buf_info.strides[0]);
                auto r_row_it = (nupic::Real32*) (r_data + i * r_buf_info.strides[0]);

                sbm.leftVecMaxAtNZ(m_row_it, r_row_it);
            }

            return r;
        });

        ///////////////////////////
        // zeroRowsIndicator
        // nonZeroRowsIndicator
        ///////////////////////////

        sbm.def("zeroRowsIndicator",
            [](const SM_01_32_32_t& sbm)
        {
            py::array_t<nupic::UInt32> res(sbm.nRows());

            nupic::UInt32 count = sbm.zeroRowsIndicator(get_it(res), get_end(res));

            return py::make_tuple(count, res);
        });

        sbm.def("nonZeroRowsIndicator",
            [](const SM_01_32_32_t& sbm)
        {
            py::array_t<nupic::UInt32> res(sbm.nRows());

            nupic::UInt32 count = sbm.nonZeroRowsIndicator(get_it(res), get_end(res));

            return py::make_tuple(count, res);
        });


        //////////////////
        // python slots
        //////////////////
        sbm.def("__eq__", [](const SM_01_32_32_t& a, const SM_01_32_32_t& b) { return a.equals(b); });

        sbm.def("__eq__", [](const SM_01_32_32_t& self, py::array_t<nupic::UInt32> a)
            {
                SM_01_32_32_t from_a(1);
				from_a.fromDense(a.shape(0), a.shape(1), get_it(a), get_end(a));

                return self.equals(from_a);
            });



        sbm.def("__ne__", [](const SM_01_32_32_t& a, const SM_01_32_32_t& b) { return !a.equals(b); });

        sbm.def("__ne__", [](const SM_01_32_32_t& self, py::array_t<nupic::UInt32> a)
        {
			SM_01_32_32_t from_a(1);
			from_a.fromDense(a.shape(0), a.shape(1), get_it(a), get_end(a));

			return !self.equals(from_a);
        });

        // pickle
        // https://github.com/pybind/pybind11/issues/1061
        sbm.def(py::pickle(
            [](const SM_01_32_32_t& a) -> std::string
        {
            // __getstate__
            std::ostringstream s;
            a.toCSR(s);

            return s.str();
        },
            [](const std::string& str) -> SM_01_32_32_t
        {
            // __setstate__
            if (str.empty())
            {
                throw std::runtime_error("Empty state");
            }

            std::istringstream is(str);

            SM_01_32_32_t SMB;
            SMB.fromCSR(is);

            return SMB;
        }
        ));


        // non members
        sbm.def("getRowSparse", [](const SM_01_32_32_t& self, nupic::UInt32 row)
            {
				py::array_t<nupic::UInt32> out(self.nNonZerosOnRow(row));
                auto it = get_it(out);

                auto Sparse_Row = self.getSparseRow(row);

                for (nupic::UInt32 i = 0; i != Sparse_Row.size(); ++i)
                {
                    *it = Sparse_Row[i];
                    it++;
                }

                return out;
            });
    }

} // namespace nupic_ext
