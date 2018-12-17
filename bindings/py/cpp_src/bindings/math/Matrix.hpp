/* ---------------------------------------------------------------------
* Numenta Platform for Intelligent Computing (NuPIC)
* Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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
* ---------------------------------------------------------------------
*/

// NOTE: regards to unchecked and mutable_unchecked.
//       See pybind11 issue #1400   https://github.com/pybind/pybind11/issues/1400

#ifndef NUPIC_EXT_BINDINGS_MATRIX_HPP
#define NUPIC_EXT_BINDINGS_MATRIX_HPP

#include <pybind11/numpy.h>

namespace nupic_ext {

    // Simple wrapper to mirror NumpyMatrix members
    template<typename T = float>
    class Numpy_Matrix
    {
    public:

        typedef T size_type;

        Numpy_Matrix(const pybind11::buffer_info& bi)
            : _bi(bi.ptr, bi.itemsize, bi.format, bi.ndim, bi.shape, bi.strides)
            , _alloc(false)
        {}

        Numpy_Matrix(const std::uint32_t nRows, const std::uint32_t nCols)
            : _matrix({ nRows, nCols })
            , _alloc(true)
        {}

        int nRows() const
        {
            if (_alloc)
            {
                return _matrix.shape(0);
            }
            else
            {
                return _bi.shape[0];
            }
        }
        int nCols() const
        {
            if (_alloc)
            {
                return _matrix.shape(1);
            }
            else
            {
                return _bi.shape[1];
            }
        }

        T get(int r, int c) const
        {
            if (_alloc)
            {
                return _matrix.template unchecked<2>()(r,c);
            }
            else
            {
                auto p = (char*)_bi.ptr + (_bi.strides[0] * r) + (_bi.strides[1] * c);
                auto element_ptr = (T*)p;

                return *element_ptr;
            }
        }

        T* get_row(int row)
        {
            if (_alloc)
            {
                auto m_bi = _matrix.request();
                auto p = (char*)m_bi.ptr + (m_bi.strides[0] * row);
                return (T*)p;
            }
            else
            {
                auto p = (char*)_bi.ptr + (_bi.strides[0] * row);
                return (T*)p;

            }
        }

        void set(int r, int c, T v)
        {
            if (_alloc)
            {
                auto accessor = _matrix.template mutable_unchecked<2>();
                accessor(r, c) = v;
            }
            else
            {
                throw std::runtime_error("Not implemented");
            }
        }

        pybind11::array_t<T> get_py_array() const
        {
            if (_alloc)
            {
                return _matrix;
            }
            else
            {
                throw std::runtime_error("Not implemented");
            }
        }

    private:

        pybind11::buffer_info _bi;
        pybind11::array_t<T> _matrix;

        bool _alloc;

    };
} // nupic_ext


#endif // NUPIC_EXT_BINDINGS_MATRIX_HPP




