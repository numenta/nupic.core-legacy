/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
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

%module(package="bindings", directors="1") algorithms
%include <nupic/bindings/exception.i>
%import <nupic/bindings/math.i>

%pythoncode %{
import os

try:
  # NOTE need to import capnp first to activate the magic necessary for
  # SpatialPoolerProto_capnp, etc.
  import capnp
except ImportError:
  capnp = None
else:
  from nupic.proto.Cells4_capnp import Cells4Proto
  from nupic.proto.ClaClassifier_capnp import ClaClassifierProto
  from nupic.proto.ConnectionsProto_capnp import ConnectionsProto
  from nupic.proto.SdrClassifier_capnp import SdrClassifierProto
  from nupic.proto.SpatialPoolerProto_capnp import SpatialPoolerProto
  from nupic.proto.SvmProto_capnp import (SvmDenseProto, Svm01Proto)
  from nupic.proto.TemporalMemoryProto_capnp import TemporalMemoryProto


_ALGORITHMS = _algorithms
%}

%{
/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
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

#include <Python.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

#include <nupic/math/Types.hpp>
#include <nupic/math/Convolution.hpp>
#include <nupic/math/Rotation.hpp>
#include <nupic/math/Erosion.hpp>
#include <nupic/algorithms/GaborNode.hpp>
#include <nupic/algorithms/ImageSensorLite.hpp>
#include <nupic/algorithms/Scanning.hpp>

#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseBinaryMatrix.hpp>
#include <nupic/algorithms/Svm.hpp>
#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>

#include <nupic/algorithms/Cell.hpp>
#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/ClassifierResult.hpp>
#include <nupic/algorithms/Connections.hpp>
#include <nupic/algorithms/SDRClassifier.hpp>
#include <nupic/algorithms/InSynapse.hpp>
#include <nupic/algorithms/OutSynapse.hpp>
#include <nupic/algorithms/SegmentUpdate.hpp>

#include <nupic/proto/Cells4.capnp.h>
#include <nupic/proto/ConnectionsProto.capnp.h>
#include <nupic/proto/SpatialPoolerProto.capnp.h>
#include <nupic/proto/TemporalMemoryProto.capnp.h>
%}

//
// Numpy API
//
%{
#include <nupic/py_support/NumpyArrayObject.hpp>
%}
%init %{
  nupic::initializeNumpy();
%}

%{
#include <nupic/py_support/NumpyVector.hpp>
#include <nupic/py_support/PyCapnp.hpp>
#include <nupic/py_support/PythonStream.hpp>
#include <nupic/py_support/PyHelpers.hpp>

// Hack to fix SWIGPY_SLICE_ARG not found bug
#if PY_VERSION_HEX >= 0x03020000
# define SWIGPY_SLICE_ARG(obj) ((PyObject*) (obj))
#else
# define SWIGPY_SLICE_ARG(obj) ((PySliceObject*) (obj))
#endif

/// %template(_InSynapse) nupic::algorithms::Cells3::InSynapse<nupic::UInt32, nupic::Real32>;
/// %template(Segment3_32) nupic::algorithms::Cells3::Segment<nupic::UInt32, nupic::Real32>;
/// %template(Cell3_32) nupic::algorithms::Cells3::Cell<nupic::UInt32, nupic::Real32>;
/// %template(Cells3_32) nupic::algorithms::Cells3::Cells3<nupic::UInt32, nupic::Real32>;
using namespace nupic::algorithms::connections;
using namespace nupic::algorithms::temporal_memory;
using namespace nupic::algorithms::Cells4;
using namespace nupic::algorithms::cla_classifier;
using namespace nupic::algorithms::sdr_classifier;
using namespace nupic;

#define CHECKSIZE(var) \
  NTA_ASSERT(PyArray_DESCR(var)->elsize == 4) << " elsize:" << PyArray_DESCR(var)->elsize

%}


// %pythoncode %{
//   import numpy
//   from bindings import math
// %}

%pythoncode %{
  uintDType = "uint32"
%}

%naturalvar;


// This dummy inline function exists only to force the linker
// to keep the gaborCompute() function in the resulting
// shared object.
%inline {

void forceRetentionOfGaborComputeWithinLibrary(void) {
 gaborCompute( NULL,                // const NUMPY_ARRAY * psGaborBank
               NULL,                // const NUMPY_ARRAY * psInput
               NULL,                // const NUMPY_ARRAY * psAlpha
               NULL,                // const NUMPY_ARRAY * psBBox
               NULL,                // const NUMPY_ARRAY * psImageBox
               NULL,                // const NUMPY_ARRAY * psOutput
               0.0f,                // float fGainConstant
               (EDGE_MODE)0,        // EDGE_MODE eEdgeMode
               0.0f,                // float fOffImageFillValue
               (PHASE_MODE)0,       // PHASE_MODE ePhaseMode
               (NORMALIZE_METHOD)0, // NORMALIZE_METHOD eNormalizeMethod
               (NORMALIZE_MODE)0,   // NORMALIZE_MODE eNormalizeMode
               (PHASENORM_MODE)0,   // PHASENORM_MODE ePhaseNormMode
               (POSTPROC_METHOD)0,  // POSTPROC_METHOD ePostProcMethod
               0.0f,                // float fPostProcSlope
               0.0f,                // float fPostProcMidpoint
               0.0f,                // float fPostProcMin
               0.0f,                // float fPostProcMax
               NULL,                // const NUMPY_ARRAY * psBufferIn
               NULL,                // const NUMPY_ARRAY * psBufferOut
               NULL,                // const NUMPY_ARRAY * psPostProcLUT
               0.0f                 // float fPostProcScalar
  );
  // Initialization of log system from python disabled for now.
  // See comments in gaborNode.cpp
  // initFromPython(0);
}

}

// These dummy inline functions exist only to force the linker
// to keep the ImageSensorLite functions in the resulting
// shared object.
%inline {
void forceRetentionOfImageSensorLiteLibrary(void) {
  extractAuxInfo( NULL,          // const char * pCtlBufAddr
                  NULL,          // BBOX * psBox
                  NULL,          // int * pnAddress
                  NULL,          // int * pnPartitionID
                  NULL,          // int * pnCategoryID
                  NULL,          // int * pnVideoID
                  NULL           // int * pnAlphaAddress
  );
}
}


//--------------------------------------------------------------------------------
// SVM
//--------------------------------------------------------------------------------
%include <nupic/algorithms/Svm.hpp>

%ignore nupic::algorithms::svm::operator=;

%extend nupic::algorithms::svm::svm_problem
{
  inline void get_samples(PyObject* samplesIn)
  {
    PyArrayObject* samples_py = (PyArrayObject*)samplesIn;
    for (int i = 0; i < self->size(); ++i) {
      float* row_it = (float*)((char *)PyArray_DATA(samples_py)+i*PyArray_STRIDES(samples_py)[0]);
      *row_it++ = self->y_[i];
      for (int j = 0; j < self->n_dims(); ++j, ++row_it)
    *row_it = self->x_[i][j];
    }
  }
}

%extend nupic::algorithms::svm::svm_problem01
{
  inline void get_samples(PyObject* samplesIn)
  {
    PyArrayObject* samples_py = (PyArrayObject*)samplesIn;
    for (int i = 0; i < self->size(); ++i) {
      float* row_it = (float*)((char *)PyArray_DATA(samples_py)+i*PyArray_STRIDES(samples_py)[0]);
      *row_it++ = self->y_[i];
      std::fill(row_it, row_it + self->n_dims(), (float) 0);
      for (int j = 0; j < self->nnz(i); ++j)
    *(row_it + self->x_[i][j]) = 1;
    }
  }
}

%extend nupic::algorithms::svm::svm_model
{
  inline void get_support_vectors(PyObject* svIn)
  {
    PyArrayObject* sv_py = (PyArrayObject*)svIn;
    for (int i = 0; i < self->size(); ++i) {
      float* row_it = (float*)((char *)PyArray_DATA(sv_py)+i*PyArray_STRIDES(sv_py)[0]);
      for (int j = 0; j < self->n_dims(); ++j, ++row_it)
    *row_it = self->sv[i][j];
    }
  }

  inline void get_support_vector_coefficients(PyObject* svCoeffIn)
  {
    PyArrayObject* sv_coeff_py = (PyArrayObject*)svCoeffIn;
    for (size_t i = 0; i < self->sv_coef.size(); ++i) {
      float* row_it = (float*)((char *)PyArray_DATA(sv_coeff_py)+i*PyArray_STRIDES(sv_coeff_py)[0]);
      for (int j = 0; j < self->size(); ++j, ++row_it)
    *row_it = self->sv_coef[i][j];
    }
  }

  inline PyObject* get_hyperplanes()
  {
    if (self->n_class() == 1)
      Py_RETURN_NONE;

    size_t m = self->w.size(), n = self->w[0].size();
    int dims[] = { int(m), int(n) };
    nupic::NumpyMatrix out(dims);
    for (size_t i = 0; i != m; ++i)
      for (size_t j = 0; j != n; ++j)
    *(out.addressOf(0,0) + i*n + j) = self->w[i][j];
    return out.forPython();
  }
}

%extend nupic::algorithms::svm::svm_dense
{
  PyObject* __getstate__()
  {
    SharedPythonOStream py_s(self->persistent_size());
    std::ostream& s = py_s.getStream();
    self->save(s);
    return py_s.close();
  }

  %pythoncode %{
    def __init__(self, *args, **kwargs):
      """
      __init__(self, kernel=0, n_dims=0, threshold=.9, cache_size=100, shrinking=1,
        probability=False, seed=-1) -> svm_dense

      nupic::algorithms::svm::svm_dense::svm_dense(int kernel=0, int n_dims=0,
      float threshold=.9, int cache_size=100, int shrinking=1, bool
      probability=false)
      """
      # Convert numpy ints to regular ints for Python 2.6
      for k in ('kernel', 'n_dims', 'cache_size', 'shrinking'):
          if k in kwargs:
            kwargs[k] = int(kwargs[k])

      this = _ALGORITHMS.new_svm_dense(*args, **kwargs)
      try: self.this.append(this)
      except: self.this = this

    def __setstate__(self, inString):
      self.this = _ALGORITHMS.new_svm_dense()
      self.thisown = 1
      self.loadFromString(inString)
    
    def convertedRead(self, proto):
      """Initialize the svm_dense instance from the given SvmDenseProto
      reader.

      :param proto: SvmDenseProto message reader containing data from a
                    previously serialized svm_dense instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the svm_dense instance using capnp.

      :param: Destination SvmDenseProto message builder
      """
      reader = SvmDenseProto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy
    
    @classmethod
    def getSchema(cls):
      """ Get Cap'n Proto schema. 
      :return: Cap'n Proto schema
      """
      return SvmDenseProto  
  %}
  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }
  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  inline void add_sample(float y_val, PyObject* x_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    self->add_sample(y_val, (float*)PyArray_DATA(x));
  }

  inline float predict(PyObject* x_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    return self->predict((float*)PyArray_DATA(x));
  }

  inline float predict_probability(PyObject* x_vector, PyObject* proba_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    PyArrayObject* proba = (PyArrayObject*) proba_vector;
    return self->predict_probability((float*)PyArray_DATA(x), (float*)PyArray_DATA(proba));
  }

  inline void save(const std::string& filename)
  {
    std::ofstream save_file(filename.c_str());
    self->save(save_file);
    save_file.close();
  }

  inline void load(const std::string& filename)
  {
    std::ifstream load_file(filename.c_str());
    self->load(load_file);
    load_file.close();
  }

  inline float cross_validate(int n_fold, float gamma, float C, float eps)
  {
    float accuracy;
    Py_BEGIN_ALLOW_THREADS;
    accuracy = self->cross_validation(n_fold, gamma, C, eps);
    Py_END_ALLOW_THREADS;
    return accuracy;
  }

  inline void trainReleaseGIL(float gamma, float C, float eps)
  {
    Py_BEGIN_ALLOW_THREADS;
    self->train(gamma, C, eps);
    Py_END_ALLOW_THREADS;
  }
};

%extend nupic::algorithms::svm::svm_01
{
  PyObject* __getstate__()
  {
    SharedPythonOStream py_s(self->persistent_size());
    std::ostream& s = py_s.getStream();
    self->save(s);
    return py_s.close();
  }

  %pythoncode %{
    def __setstate__(self, inString):
      self.this = _ALGORITHMS.new_svm_01()
      self.thisown = 1
      self.loadFromString(inString)
    
    def convertedRead(self, proto):
      """Initialize the svm_dense instance from the given Svm01Proto
      reader.

      :param proto: Svm01Proto message reader containing data from a
                    previously serialized svm_dense instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the svm_01 instance using capnp.

      :param: Destination Svm01Proto message builder
      """
      reader = Svm01Proto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy
    
    @classmethod
    def getSchema(cls):
      """ Get Cap'n Proto schema. 
      :return: Cap'n Proto schema
      """
      return Svm01Proto        
  %}

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }
  
  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  inline void add_sample(float y_val, PyObject* x_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    self->add_sample(y_val, (float*)PyArray_DATA(x));
  }

  inline float predict(PyObject* x_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    return self->predict((float*)PyArray_DATA(x));
  }

  inline float predict_probability(PyObject* x_vector, PyObject* proba_vector)
  {
    PyArrayObject* x = (PyArrayObject*) x_vector;
    PyArrayObject* proba = (PyArrayObject*) proba_vector;
    return self->predict_probability((float*)PyArray_DATA(x), (float*)PyArray_DATA(proba));
  }

  inline float cross_validate(int n_fold, float gamma, float C, float eps)
  {
    float accuracy;
    Py_BEGIN_ALLOW_THREADS;
    accuracy = self->cross_validation(n_fold, gamma, C, eps);
    Py_END_ALLOW_THREADS;
    return accuracy;
  }

  inline void trainReleaseGIL(float gamma, float C, float eps)
  {
    Py_BEGIN_ALLOW_THREADS;
    self->train(gamma, C, eps);
    Py_END_ALLOW_THREADS;
  }

  inline void save(const std::string& filename)
  {
    std::ofstream save_file(filename.c_str());
    self->save(save_file);
    save_file.close();
  }

  inline void load(const std::string& filename)
  {
    std::ifstream load_file(filename.c_str());
    self->load(load_file);
    load_file.close();
  }
};

//--------------------------------------------------------------------------------
// CONVOLUTION
//--------------------------------------------------------------------------------
%include <nupic/math/Convolution.hpp>

%template(Float32SeparableConvolution2D) SeparableConvolution2D<float>;

%extend SeparableConvolution2D<float>
{
  inline void init(nupic::UInt32 nrows, nupic::UInt32 ncols,
           nupic::UInt32 f1_size, nupic::UInt32 f2_size,
           PyObject* pyF1, PyObject* pyF2)
  {
    PyArrayObject *f1 = (PyArrayObject*) pyF1;
    PyArrayObject *f2 = (PyArrayObject*) pyF2;

    self->init(nrows, ncols, f1_size, f2_size, (float*)(PyArray_DATA(f1)), (float*)(PyArray_DATA(f2)));
  }

  inline void compute(PyObject* pyData, PyObject* pyConvolved, bool rotated45 =false)
  {
    PyArrayObject* data = (PyArrayObject*)pyData;
    PyArrayObject* convolved = (PyArrayObject*)pyConvolved;

    self->compute((float*)(PyArray_DATA(data)), (float*)(PyArray_DATA(convolved)), rotated45);
  }

  inline void getBuffer(PyObject* pyBuffer) const
  {
    PyArrayObject *buffer = (PyArrayObject*)pyBuffer;

    const size_t size = self->nrows_ * self->ncols_;
    std::copy(self->buffer_, self->buffer_ + size, (float*)(PyArray_DATA(buffer)));
  }
};

//--------------------------------------------------------------------------------
// ROTATION
//--------------------------------------------------------------------------------
%include <nupic/math/Rotation.hpp>

%template(Float32Rotation45) Rotation45<float>;

%extend Rotation45<float>
{
  inline void rotate(PyObject* pyOriginal, PyObject* pyRotated,
             nupic::UInt32 nrows, nupic::UInt32 ncols, nupic::UInt32 z)
  {
    PyArrayObject* original = (PyArrayObject*)pyOriginal;
    PyArrayObject* rotated = (PyArrayObject*)pyRotated;

    self->rotate((float*)(PyArray_DATA(original)), (float*)(PyArray_DATA(rotated)),
      nrows, ncols, z);
  }

  inline void unrotate(PyObject* pyUnrotated, PyObject* pyRotated,
               nupic::UInt32 nrows, nupic::UInt32 ncols, nupic::UInt32 z)
  {
    PyArrayObject* unrotated = (PyArrayObject*)pyUnrotated;
    PyArrayObject* rotated = (PyArrayObject*)pyRotated;

    self->unrotate((float*)(PyArray_DATA(unrotated)), (float*)(PyArray_DATA(rotated)),
      nrows, ncols, z);
  }
};

//--------------------------------------------------------------------------------
// EROSION
//--------------------------------------------------------------------------------
%include <nupic/math/Erosion.hpp>

%template(Float32Erosion) Erosion<float>;

%extend Erosion<float>
{
  inline void init(nupic::UInt32 nrows, nupic::UInt32 ncols)
  {
    self->init(nrows, ncols);
  }

  inline void compute(PyObject* pyData, PyObject* pyEroded,
                      nupic::UInt32 iterations, bool dilate=false)
  {
    PyArrayObject* data = (PyArrayObject*)pyData;
    PyArrayObject* eroded = (PyArrayObject*)pyEroded;

    self->compute((float*)(PyArray_DATA(data)), (float*)(PyArray_DATA(eroded)),
                  iterations, dilate);
  }

  inline void getBuffer(PyObject* pyBuffer) const
  {
    PyArrayObject *buffer = (PyArrayObject*)pyBuffer;

    const size_t size = self->nrows_ * self->ncols_;
    std::copy(self->buffer_, self->buffer_ + size, (float*)(PyArray_DATA(buffer)));
  }
};

//--------------------------------------------------------------------------------
// SCANNING
//--------------------------------------------------------------------------------

%include <nupic/algorithms/Scanning.hpp>
%inline {
  void computeAlpha(nupic::UInt32 xstep, nupic::UInt32 ystep,
                    nupic::UInt32 widthS, nupic::UInt32 heightS,
                    nupic::UInt32 imageWidth, nupic::UInt32 imageHeight,
                    nupic::UInt32 xcount, nupic::UInt32 ycount,
                    nupic::UInt32 weightWidth, float sharpness,
                    PyObject* pyData, PyObject* pyValues,
                    PyObject* pyCounts, PyObject* pyWeights)
  {
    PyArrayObject *data = (PyArrayObject*) pyData;
    PyArrayObject *values = (PyArrayObject*) pyValues;
    PyArrayObject *counts = (PyArrayObject*) pyCounts;
    PyArrayObject *weights = (PyArrayObject*) pyWeights;
    computeAlpha(xstep, ystep, widthS, heightS, imageWidth, imageHeight,
                 xcount, ycount, weightWidth, sharpness,
                 (float*)(PyArray_DATA(data)), (float*)(PyArray_DATA(values)),
                 (float*)(PyArray_DATA(counts)), (float*)(PyArray_DATA(weights)));
  }
}


//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
// DENDRITIC TREE - started Jan 2010
//--------------------------------------------------------------------------------
%template(Byte_Vector) std::vector<nupic::Byte>;

%include <nupic/math/Types.hpp>
 ///%include <nupic/algorithms/Cells.hpp>

 ///%template(Segment_32) nupic::algorithms::Segment<nupic::UInt32, nupic::Real32>;
 ///%template(Branch_32) nupic::algorithms::Branch<nupic::UInt32, nupic::Real32>;
 ///%template(Cell_32) nupic::algorithms::Cell<nupic::UInt32, nupic::Real32>;
 ///%template(SegVector_32) std::vector<nupic::algorithms::Segment<nupic::UInt32, nupic::Real32>*>;
 ///%template(BranchVector_32) std::vector<nupic::algorithms::Branch<nupic::UInt32, nupic::Real32>*>;
 ///%template(Cells_32) nupic::algorithms::Cells<nupic::UInt32, nupic::Real32>;
 ///%template(Int_Seg_32) std::pair<nupic::UInt32, nupic::algorithms::Segment<nupic::UInt32,nupic::Real32>*>;

// Already seen by swig on linux32 where size_t is the same size as unsigned int
#if !(defined(NTA_ARCH_32) && defined(NTA_OS_LINUX))
%template(Size_T_Vector) std::vector<size_t>;
#endif

//--------------------------------------------------------------------------------
// Some functions, faster than numpy.
//--------------------------------------------------------------------------------
%inline {

  inline nupic::UInt32 non_zeros_ui8(PyObject* py_x, PyObject* py_y)
  {
    PyArrayObject* x = (PyArrayObject*)py_x;
    PyArrayObject* y = (PyArrayObject*)py_y;
    nupic::UInt32 nnz = 0;
    unsigned char* x_data = (unsigned char*) PyArray_DATA(x);
    nupic::UInt32* y_res = (nupic::UInt32*) PyArray_DATA(y);
    for (int i = 0; i != PyArray_DIMS(x)[0]; ++i)
      if (x_data[i] != 0)
        y_res[nnz++] = i;
    return nnz;
  }

  inline nupic::UInt32 non_zeros_i32(PyObject* py_x, PyObject* py_y)
  {
    PyArrayObject* x = (PyArrayObject*)py_x;
    PyArrayObject* y = (PyArrayObject*)py_y;
    nupic::UInt32 nnz = 0;
    nupic::UInt32* x_data = (nupic::UInt32*) PyArray_DATA(x);
    nupic::UInt32* y_res = (nupic::UInt32*) PyArray_DATA(y);
    for (int i = 0; i != PyArray_DIMS(x)[0]; ++i)
      if (x_data[i] != 0)
        y_res[nnz++] = i;
    return nnz;
  }

  inline nupic::UInt32 non_zeros_f32(PyObject* py_x, PyObject* py_y)
  {
    PyArrayObject* x = (PyArrayObject*)py_x;
    CHECKSIZE(x);
    PyArrayObject* y = (PyArrayObject*)py_y;
    CHECKSIZE(y);
    nupic::UInt32 nnz = 0;
    nupic::Real32* x_data = (nupic::Real32*) PyArray_DATA(x);
    nupic::UInt32* y_res = (nupic::UInt32*) PyArray_DATA(y);
    for (int i = 0; i != PyArray_DIMS(x)[0]; ++i)
      if (x_data[i] != 0)
        y_res[nnz++] = i;
    return nnz;
  }

  inline void rightVecProdAtIndices(PyObject* py_ind, PyObject* py_x, PyObject* py_y)
  {
    PyArrayObject* ind = (PyArrayObject*)py_ind;
    PyArrayObject* x = (PyArrayObject*)py_x;
    PyArrayObject* y = (PyArrayObject*)py_y;
    nupic::UInt32 nc = PyArray_DIMS(ind)[0];
    nupic::UInt32 ni = PyArray_DIMS(ind)[1];
    nupic::UInt32* ind_data = (nupic::UInt32*) PyArray_DATA(ind);
    nupic::Real32* x_data = (nupic::Real32*) PyArray_DATA(x);
    nupic::Real32* y_res = (nupic::Real32*) PyArray_DATA(y);

    for (nupic::UInt32 c = 0; c != nc; ++c, ind_data += ni) {
      nupic::Real32 val = 0;
      for (nupic::UInt32 i = 0; i != ni; ++i)
        val += x_data[ind_data[i]];
      *y_res++ = val;
    }
  }
}

//--------------------------------------------------------------------------------
// Optimizations for FDR
%inline {

  //--------------------------------------------------------------------------------
  inline nupic::UInt32 getSegmentActivityLevel(PyObject* py_seg, PyObject* py_state,
                                             bool connectedSynapsesOnly,
                                             nupic::Real32 connectedPerm)
  {
    PyArrayObject* _state = (PyArrayObject*) py_state;
    nupic::Byte* state = (nupic::Byte*) PyArray_DATA(_state);
    nupic::UInt32 stride0 = PyArray_STRIDES(_state)[0];

    nupic::py::List seg;
    seg.assign(py_seg);
    Py_ssize_t n = seg.getCount();
    nupic::UInt32 activity = 0;

    if (connectedSynapsesOnly)
      for (Py_ssize_t i = 0; i < n; ++i) {
        nupic::py::List syn;
        syn.assign(seg.fastGetItem(i));
        nupic::Real32 p = (nupic::Real32) PyFloat_AsDouble(syn.fastGetItem(2));
        if (p >= connectedPerm) {
          nupic::UInt32 c = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(0));
          nupic::UInt32 j = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(1));
          activity += state[c * stride0 + j];
        }
      }
    else
      for (Py_ssize_t i = 0; i < n; ++i) {
        nupic::py::List syn;
        syn.assign(seg.fastGetItem(i));
        nupic::UInt32 c = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(0));
        nupic::UInt32 j = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(1));
        activity += state[c * stride0 + j];
      }

    return activity;
  }

  //--------------------------------------------------------------------------------
  inline bool isSegmentActive(PyObject* py_seg, PyObject* py_state,
                              nupic::Real32 connectedPerm,
                              nupic::UInt32 activationThreshold)
  {
    PyArrayObject* _state = (PyArrayObject*) py_state;
    nupic::Byte* state = (nupic::Byte*) PyArray_DATA(_state);
    nupic::UInt32 stride0 = PyArray_STRIDES(_state)[0];

    nupic::py::List seg;
    seg.assign(py_seg);
    Py_ssize_t n = seg.getCount();
    nupic::UInt32 activity = 0;

    if (n < (Py_ssize_t) activationThreshold)
      return false;

    for (Py_ssize_t i = 0; i < n; ++i) {
      nupic::py::List syn;
      syn.assign(seg.fastGetItem(i));
      nupic::Real32 p = (nupic::Real32) PyFloat_AsDouble(syn.fastGetItem(2));
      if (p >= connectedPerm) {
        nupic::UInt32 c = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(0));
        nupic::UInt32 j = (nupic::UInt32) PyLong_AsLong(syn.fastGetItem(1));
        activity += state[c * stride0 + j];
        if (activity >= activationThreshold)
          return true;
      }
    }

    return false;
  }
}


//--------------------------------------------------------------------------------
// NEW ALGORITHMS (Cells4)
%include <nupic/algorithms/Segment.hpp>
%include <nupic/algorithms/SegmentUpdate.hpp>
%include <nupic/algorithms/OutSynapse.hpp>
%include <nupic/algorithms/InSynapse.hpp>
%include <nupic/algorithms/Cell.hpp>



//--------------------------------------------------------------------------------
%extend nupic::algorithms::Cells4::Segment<nupic::UInt32, nupic::Real32>
{
  %pythoncode %{
    def __init__(self, *args):
      self.this = _ALGORITHMS.new_Segment3_32()
  %}

  inline bool isActive(PyObject* py_activities,
                       nupic::Real32 permConnected,
                       nupic::UInt32 activationThreshold) const
  {
    PyArrayObject* act = (PyArrayObject*) py_activities;
    return self->isActive((nupic::UInt32*) PyArray_DATA(act),
                          permConnected,
                          activationThreshold);
  }
};

%pythoncode %{

  def Segment3(*args, **keywords):
     return Segment3_32(*args)
%}



//--------------------------------------------------------------------------------
/*
%inline {

  inline void scalarEncoding(nupic::UInt32 minval, nupic::UInt32 nInternal,
                             nupic::Real32 range, nupic::UInt32 padding, nupic::UInt32 n,
                             nupic::Real32 input, PyObject* py_output)
  {
    PyArrayObject* p_output = (PyArrayObject*) py_output;
    nupic::Real32 output = PyArray_DATA(p_output);
    int centerbin = padding + int((input - minval) * nInternal / range);

  }

 }
*/


//--------------------------------------------------------------------------------
// EVEN NEWER ALGORITHMS (Cells4)
%include <nupic/algorithms/Cells4.hpp>


//--------------------------------------------------------------------------------
%extend nupic::algorithms::Cells4::Cells4
{
  %pythoncode %{

    def __init__(self, *args, **kwargs):
      self.this = _ALGORITHMS.new_Cells4(*args, **kwargs)

    def __setstate__(self, inString):
      self.this = _ALGORITHMS.new_Cells4()
      self.loadFromString(inString)

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
      return instance

    def write(self, pyBuilder):
      """Serialize the Cells4 instance using capnp.

      :param: Destination Cells4Proto message builder
      """
      reader = Cells4Proto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy

  %}

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }

  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  PyObject* __getstate__()
  {
    SharedPythonOStream py_s(self->persistentSize());
    std::ostream& s = py_s.getStream();
    self->save(s);
    return py_s.close();
  }

  inline void setStatePointers(PyObject* py_infActiveStateT,
                               PyObject* py_infActiveStateT1,
                               PyObject* py_infPredictedStateT,
                               PyObject* py_infPredictedStateT1,
                               PyObject* py_colConfidenceT,
                               PyObject* py_colConfidenceT1,
                               PyObject* py_cellConfidenceT,
                               PyObject* py_cellConfidenceT1)
  {
    PyArrayObject* infActiveStateT = (PyArrayObject*) py_infActiveStateT;
    PyArrayObject* infActiveStateT1 = (PyArrayObject*) py_infActiveStateT1;
    PyArrayObject* infPredictedStateT = (PyArrayObject*) py_infPredictedStateT;
    PyArrayObject* infPredictedStateT1 = (PyArrayObject*) py_infPredictedStateT1;
    PyArrayObject* colConfidenceT = (PyArrayObject*) py_colConfidenceT;
    PyArrayObject* colConfidenceT1 = (PyArrayObject*) py_colConfidenceT1;
    PyArrayObject* cellConfidenceT = (PyArrayObject*) py_cellConfidenceT;
    PyArrayObject* cellConfidenceT1 = (PyArrayObject*) py_cellConfidenceT1;

    self->setStatePointers((nupic::Byte*) PyArray_DATA(infActiveStateT),
                           (nupic::Byte*) PyArray_DATA(infActiveStateT1),
                           (nupic::Byte*) PyArray_DATA(infPredictedStateT),
                           (nupic::Byte*) PyArray_DATA(infPredictedStateT1),
                           (nupic::Real*) PyArray_DATA(colConfidenceT),
                           (nupic::Real*) PyArray_DATA(colConfidenceT1),
                           (nupic::Real*) PyArray_DATA(cellConfidenceT),
                           (nupic::Real*) PyArray_DATA(cellConfidenceT1));
  }

  inline PyObject* getStates() const
  {
    nupic::UInt32 nCells = self->nCells();
    nupic::UInt32 nColumns = self->nColumns();

    nupic::Byte* cpp_activeT, *cpp_activeT1;
    nupic::Byte* cpp_predT, *cpp_predT1;
    nupic::Real* cpp_colConfidenceT, *cpp_colConfidenceT1;
    nupic::Real* cpp_confidenceT, *cpp_confidenceT1;

    self->getStatePointers(cpp_activeT, cpp_activeT1,
                           cpp_predT, cpp_predT1,
                           cpp_colConfidenceT, cpp_colConfidenceT1,
                           cpp_confidenceT, cpp_confidenceT1);

    nupic::NumpyVectorT<nupic::Byte> activeT(nCells, cpp_activeT);
    nupic::NumpyVectorT<nupic::Byte> activeT1(nCells, cpp_activeT1);
    nupic::NumpyVectorT<nupic::Byte> predT(nCells, cpp_predT);
    nupic::NumpyVectorT<nupic::Byte> predT1(nCells, cpp_predT1);
    nupic::NumpyVectorT<nupic::Real> colConfidenceT(nColumns, cpp_colConfidenceT);
    nupic::NumpyVectorT<nupic::Real> colConfidenceT1(nColumns, cpp_colConfidenceT1);
    nupic::NumpyVectorT<nupic::Real> confidenceT(nCells, cpp_confidenceT);
    nupic::NumpyVectorT<nupic::Real> confidenceT1(nCells, cpp_confidenceT1);

    PyObject *result = PyTuple_New(8);
    PyTuple_SET_ITEM(result, 0, activeT.forPython());
    PyTuple_SET_ITEM(result, 1, activeT1.forPython());
    PyTuple_SET_ITEM(result, 2, predT.forPython());
    PyTuple_SET_ITEM(result, 3, predT1.forPython());
    PyTuple_SET_ITEM(result, 4, colConfidenceT.forPython());
    PyTuple_SET_ITEM(result, 5, colConfidenceT1.forPython());
    PyTuple_SET_ITEM(result, 6, confidenceT.forPython());
    PyTuple_SET_ITEM(result, 7, confidenceT1.forPython());

    return result;
  }

  inline PyObject* getLearnStates() const
  {
    nupic::UInt32 nCells = self->nCells();

    nupic::Byte* cpp_activeT, *cpp_activeT1;
    nupic::Byte* cpp_predT, *cpp_predT1;

    self->getLearnStatePointers(cpp_activeT, cpp_activeT1,
                           cpp_predT, cpp_predT1);

    nupic::NumpyVectorT<nupic::Byte> activeT(nCells, cpp_activeT);
    nupic::NumpyVectorT<nupic::Byte> activeT1(nCells, cpp_activeT1);
    nupic::NumpyVectorT<nupic::Byte> predT(nCells, cpp_predT);
    nupic::NumpyVectorT<nupic::Byte> predT1(nCells, cpp_predT1);

    PyObject *result = PyTuple_New(4);
    PyTuple_SET_ITEM(result, 0, activeT.forPython());
    PyTuple_SET_ITEM(result, 1, activeT1.forPython());
    PyTuple_SET_ITEM(result, 2, predT.forPython());
    PyTuple_SET_ITEM(result, 3, predT1.forPython());

    return result;
  }

  /*
  inline std::pair<nupic::UInt32, nupic::UInt32>
    getBestMatchingCell(nupic::UInt32 colIdx, PyObject* py_state)
    {
      PyArrayObject* st = (PyArrayObject*) py_state;
      return self->getBestMatchingCell(colIdx, (nupic::UInt32*) PyArray_DATA(st));
    }
  */

  /*
  inline void computeUpdate(nupic::UInt32 colIdx, nupic::UInt32 cellIdxInCol,
                            nupic::UInt32 segIdx, PyObject* py_state,
                            PyObject* py_learnState,
                            bool sequenceSegmentFlag = false,
                            bool newSynapsesFlag = false)
  {
    PyArrayObject* st = (PyArrayObject*) py_state;
    PyArrayObject* lst = (PyArrayObject*) py_learnState;
    self->computeUpdate(colIdx, cellIdxInCol, segIdx, (nupic::UInt32*) PyArray_DATA(st),
                        (nupic::UInt32*) PyArray_DATA(lst),
                        sequenceSegmentFlag, newSynapsesFlag);
  }
  */

  inline PyObject* compute(PyObject* py_x, bool doInference, bool doLearning)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    nupic::NumpyVectorT<nupic::Real> y(self->nCells());
    self->compute((nupic::Real*) PyArray_DATA(x), y.begin(), doInference, doLearning);
    return y.forPython();
  }
}

// In these SWIG wrapper methods, don't use the `const` qualifier. There has to
// be some difference in the method signature so that C++ function overloading
// can happen. Expose the internal `const` methods with a different name.
%rename(getOverlapsTuple) nupic::algorithms::spatial_pooler::SpatialPooler::getOverlaps() const;
%rename(getBoostedOverlapsTuple) nupic::algorithms::spatial_pooler::SpatialPooler::getBoostedOverlaps() const;

%include <nupic/algorithms/SpatialPooler.hpp>

%extend nupic::algorithms::spatial_pooler::SpatialPooler
{
  %pythoncode %{
    def __init__(self,
                 inputDimensions=[32,32],
                 columnDimensions=[64,64],
                 potentialRadius=16,
                 potentialPct=0.5,
                 globalInhibition=False,
                 localAreaDensity=-1.0,
                 numActiveColumnsPerInhArea=10.0,
                 stimulusThreshold=0,
                 synPermInactiveDec=0.01,
                 synPermActiveInc=0.1,
                 synPermConnected=0.10,
                 minPctOverlapDutyCycle=0.001,
                 dutyCyclePeriod=1000,
                 boostStrength=0.0,
                 seed=-1,
                 spVerbosity=0,
                 wrapAround=True):
      self.this = _ALGORITHMS.new_SpatialPooler()
      _ALGORITHMS.SpatialPooler_initialize(
        self, inputDimensions, columnDimensions, potentialRadius, potentialPct,
        globalInhibition, localAreaDensity, numActiveColumnsPerInhArea,
        stimulusThreshold, synPermInactiveDec, synPermActiveInc, synPermConnected,
        minPctOverlapDutyCycle, dutyCyclePeriod,
        boostStrength, seed, spVerbosity, wrapAround)

    def __getstate__(self):
      # Save the local attributes but override the C++ spatial pooler with the
      # string representation.
      d = dict(self.__dict__)
      d["this"] = self.getCState()
      return d

    def __setstate__(self, state):
      # Create an empty C++ spatial pooler and populate it from the serialized
      # string.
      self.this = _ALGORITHMS.new_SpatialPooler()
      if isinstance(state, str):
        self.loadFromString(state)
        self.valueToCategory = {}
      else:
        self.loadFromString(state["this"])
        # Use the rest of the state to set local Python attributes.
        del state["this"]
        self.__dict__.update(state)

    def _updateBookeepingVars(self, learn):
      self.updateBookeepingVars_(learn)

    def _calculateOverlap(self, inputVector):
      return self.calculateOverlap_(inputVector)

    def _inhibitColumns(self, overlaps):
      return self.inhibitColumns_(overlaps)

    def _updatePermanencesForColumn(self, perm, column, raisePerm=True):
      self.updatePermanencesForColumn_(perm, column, raisePerm)

    def _updateDutyCycles(self, overlaps, activeArray):
      self.updateDutyCycles_(overlaps, activeArray)

    def _bumpUpWeakColumns(self):
      self.bumpUpWeakColumns_();

    def _updateBoostFactors(self):
      self.updateBoostFactors_();

    def _isUpdateRound(self):
      return self.isUpdateRound_();

    def _updateInhibitionRadius(self):
      self.updateInhibitionRadius_();

    def _updateMinDutyCycles(self):
      self.updateMinDutyCycles_();

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the SpatialPooler instance using capnp.

      :param: Destination SpatialPoolerProto message builder
      """
      reader = SpatialPoolerProto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy


    def convertedRead(self, proto):
      """Initialize the SpatialPooler instance from the given SpatialPoolerProto
      reader.

      :param proto: SpatialPoolerProto message reader containing data from a
                    previously serialized SpatialPooler instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
  %}

  inline void compute(PyObject *py_x, bool learn, PyObject *py_y)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    PyArrayObject* y = (PyArrayObject*) py_y;
    self->compute((nupic::UInt*) PyArray_DATA(x), (bool)learn, (nupic::UInt*) PyArray_DATA(y));
  }

  inline void stripUnlearnedColumns(PyObject *py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->stripUnlearnedColumns((nupic::UInt*) PyArray_DATA(x));
  }

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }

  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }


  PyObject* getCState()
  {
    SharedPythonOStream py_s(self->persistentSize());
    std::ostream& s = py_s.getStream();
    // TODO: Consider writing floats as binary instead.
    s.flags(ios::scientific);
    s.precision(numeric_limits<double>::digits10 + 1);
    self->save(s);
    return py_s.close();
  }

  inline void setBoostFactors(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setBoostFactors((nupic::Real*) PyArray_DATA(x));
  }

  inline void getBoostFactors(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getBoostFactors((nupic::Real*) PyArray_DATA(x));
  }

  inline void setOverlapDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setOverlapDutyCycles((nupic::Real*) PyArray_DATA(x));
  }

  inline void getOverlapDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getOverlapDutyCycles((nupic::Real*) PyArray_DATA(x));
  }

  inline void setActiveDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setActiveDutyCycles((nupic::Real*) PyArray_DATA(x));
  }

  inline void getActiveDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getActiveDutyCycles((nupic::Real*) PyArray_DATA(x));
  }


  inline void setMinOverlapDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setMinOverlapDutyCycles((nupic::Real*) PyArray_DATA(x));
  }

  inline void getMinOverlapDutyCycles(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getMinOverlapDutyCycles((nupic::Real*) PyArray_DATA(x));
  }

  inline void setPotential(UInt column, PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setPotential(column, (nupic::UInt*) PyArray_DATA(x));
  }

  inline void getPotential(UInt column, PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getPotential(column, (nupic::UInt*) PyArray_DATA(x));
  }

  inline void setPermanence(UInt column, PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->setPermanence(column, (nupic::Real*) PyArray_DATA(x));
  }

  inline void getPermanence(UInt column, PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getPermanence(column, (nupic::Real*) PyArray_DATA(x));
  }

  inline void getConnectedSynapses(UInt column, PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getConnectedSynapses(column, (nupic::UInt*) PyArray_DATA(x));
  }

  inline void getConnectedCounts(PyObject* py_x)
  {
    PyArrayObject* x = (PyArrayObject*) py_x;
    self->getConnectedCounts((nupic::UInt*) PyArray_DATA(x));
  }

  PyObject* getOverlaps()
  {
    const std::vector<nupic::UInt>& overlaps = self->getOverlaps();
    nupic::NumpyVectorT<nupic::UInt> npOverlaps(overlaps.size(), overlaps.data());
    return npOverlaps.forPython();
  }

  PyObject* getBoostedOverlaps()
  {
    const std::vector<nupic::Real>& overlaps = self->getBoostedOverlaps();
    nupic::NumpyVectorT<nupic::Real> npOverlaps(overlaps.size(), overlaps.data());
    return npOverlaps.forPython();
  }

  inline PyObject* calculateOverlap_(PyObject* py_inputVector)
  {
    PyArrayObject* inputVector = (PyArrayObject*) py_inputVector;
    std::vector<nupic::UInt> overlapVector;

    self->calculateOverlap_((nupic::UInt*) PyArray_DATA(inputVector),
                            overlapVector);

    nupic::NumpyVectorT<nupic::UInt> overlap(overlapVector.size(),
                                             &overlapVector[0]);
    return overlap.forPython();
  }

  inline PyObject* inhibitColumns_(PyObject *py_overlaps)
  {
    PyArrayObject* overlaps = (PyArrayObject*) py_overlaps;
    nupic::UInt overlapsLen = (nupic::UInt)PyArray_DIMS(overlaps)[0];
    nupic::Real* overlapsData = (nupic::Real*)PyArray_DATA(overlaps);
    std::vector<nupic::Real> overlapsVector;
    overlapsVector.assign(overlapsData, overlapsData + overlapsLen);

    std::vector<nupic::UInt> activeColumnsVector;

    self->inhibitColumns_(overlapsVector, activeColumnsVector);

    nupic::NumpyVectorT<nupic::UInt> activeColumns(activeColumnsVector.size(),
                                                   &activeColumnsVector[0]);
    return activeColumns.forPython();
  }

  inline void updatePermanencesForColumn_(PyObject *py_perm, UInt column,
                                          bool raisePerm)
  {
    PyArrayObject* perm = (PyArrayObject*) py_perm;
    nupic::UInt permLen = (nupic::UInt)PyArray_DIMS(perm)[0];
    nupic::Real* permData = (nupic::Real*)PyArray_DATA(perm);
    std::vector<nupic::Real> permVector;
    permVector.assign(permData, permData + permLen);

    self->updatePermanencesForColumn_(permVector, column, raisePerm);
  }

  inline void updateDutyCycles_(PyObject* py_overlaps, PyObject* py_activeArray)
  {
    PyArrayObject* overlaps = (PyArrayObject*) py_overlaps;
    nupic::UInt overlapsLen = (nupic::UInt)PyArray_DIMS(overlaps)[0];
    nupic::UInt* overlapsData = (nupic::UInt*)PyArray_DATA(overlaps);
    std::vector<nupic::UInt> overlapsVector;
    overlapsVector.assign(overlapsData, overlapsData + overlapsLen);

    PyArrayObject* activeArray = (PyArrayObject*) py_activeArray;

    self->updateDutyCycles_(overlapsVector,
                           (nupic::UInt*) PyArray_DATA(activeArray));
  }

}


%include <nupic/algorithms/SDRClassifier.hpp>

%pythoncode %{
  import numpy
%}

%extend nupic::algorithms::sdr_classifier::SDRClassifier
{
  %pythoncode %{
    VERSION = 1

    def __init__(self, steps=(1,), alpha=0.001, actValueAlpha=0.3, verbosity=0):
      self.this = _ALGORITHMS.new_SDRClassifier(
          steps, alpha, actValueAlpha, verbosity)
      self.valueToCategory = {}
      self.version = SDRClassifier.VERSION

    def compute(self, recordNum, patternNZ, classification, learn, infer):
      isNone = False
      noneSentinel = 3.14159

      if type(classification["actValue"]) in (int, float):
        actValueList = [classification["actValue"]]
        bucketIdxList = [classification["bucketIdx"]]
        originalValueList = [classification["actValue"]]
        category = False
      elif classification["actValue"] is None:
        # Use the sentinel value so we know if it gets used in actualValues
        # returned.
        actValueList = [noneSentinel]
        originalValueList = [noneSentinel]
        # Turn learning off this step.
        learn = False
        category = False
        # This does not get used when learning is disabled anyway.
        bucketIdxList = [0]
        isNone = True
      elif type(classification["actValue"]) is list:
         actValueList = classification["actValue"]
         bucketIdxList = classification["bucketIdx"]
         originalValueList = classification["actValue"]
         category = False
      else:
        actValueList = [int(classification["bucketIdx"])]
        originalValueList = [classification["actValue"]]
        bucketIdxList = [classification["bucketIdx"]]
        category = True

      result = self.convertedCompute(
          recordNum, patternNZ, bucketIdxList,
          actValueList, category, learn, infer)

      if isNone:
        for i, v in enumerate(result["actualValues"]):
          if v - noneSentinel < 0.00001:
            result["actualValues"][i] = None
      arrayResult = dict((k, numpy.array(v)) if k != "actualValues" else (k, v)
                         for k, v in result.iteritems())

      if self.valueToCategory or isinstance(classification["actValue"], basestring):
        # Convert the bucketIdx back to the original value.
        for i in xrange(len(arrayResult["actualValues"])):
          if arrayResult["actualValues"][i] is not None:
            arrayResult["actualValues"][i] = self.valueToCategory.get(int(
                arrayResult["actualValues"][i]), classification["actValue"])

        for i in range(len(actValueList)):
          self.valueToCategory[actValueList[i]] = originalValueList[i]

      return arrayResult

    def __getstate__(self):
      # Save the local attributes but override the C++ classifier with the
      # string representation.
      d = dict(self.__dict__)
      d["this"] = self.getCState()
      return d

    def __setstate__(self, state):
      # Create an empty C++ classifier and populate it from the serialized
      # string.
      self.this = _ALGORITHMS.new_SDRClassifier()
      if isinstance(state, str):
        self.loadFromString(state)
        self.valueToCategory = {}
      else:
        assert state["version"] == self.VERSION
        self.loadFromString(state["this"])
        # Use the rest of the state to set local Python attributes.
        del state["this"]
        self.__dict__.update(state)

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the SDRClassifier instance using capnp.

      :param: Destination SdrClassifierProto message builder
      """
      reader = SdrClassifierProto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy

    @classmethod
    def getSchema(cls):
      """ Get Cap'n Proto schema. 
      :return: Cap'n Proto schema
      """
      return SdrClassifierProto

    def convertedRead(self, proto):
      """Initialize the SDRClassifier instance from the given SdrClassifierProto
      reader.

      :param proto: SdrClassifierProto message reader containing data from a
                    previously serialized SDRClassifier instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2
  %}

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  PyObject* getCState()
  {
    SharedPythonOStream py_s(self->persistentSize());
    std::ostream& s = py_s.getStream();
    // TODO: Consider writing floats as binary instead.
    s.flags(ios::scientific);
    s.precision(numeric_limits<double>::digits10 + 1);
    self->save(s);
    return py_s.close();
  }

  PyObject* convertedCompute(UInt recordNum, const vector<UInt>& patternNZ,
                             const vector<UInt>& bucketIdxList, 
                             const vector<Real64>& actValueList, bool category,
                             bool learn, bool infer)
  {
    ClassifierResult result;
    self->compute(recordNum, patternNZ, bucketIdxList, actValueList, category,
                  learn, infer, &result);
    PyObject* d = PyDict_New();
    for (map<Int, vector<Real64>*>::const_iterator it = result.begin();
         it != result.end(); ++it)
    {
      PyObject* key;
      if (it->first == -1)
      {
        key = PyString_FromString("actualValues");
      } else {
        key = PyInt_FromLong(it->first);
      }

      PyObject* value = PyList_New(it->second->size());
      for (UInt i = 0; i < it->second->size(); ++i)
      {
        PyObject* pyActValue = PyFloat_FromDouble(it->second->at(i));
        PyList_SetItem(value, i, pyActValue);
      }

      PyDict_SetItem(d, key, value);
      Py_DECREF(value);
    }
    return d;
  }

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }

  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

}

//--------------------------------------------------------------------------------
// Data structures (Connections)
%rename(ConnectionsSynapse) nupic::algorithms::connections::Synapse;
%template(ConnectionsSynapseVector) vector<nupic::algorithms::connections::Synapse>;
%feature("director") nupic::algorithms::connections::ConnectionsEventHandler;
%include <nupic/algorithms/Connections.hpp>


//--------------------------------------------------------------------------------
%extend nupic::algorithms::connections::Connections
{
  %pythoncode %{

    def __init__(self,
                 numCells,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255):
      self.this = _ALGORITHMS.new_Connections(numCells,
                                              maxSegmentsPerCell,
                                              maxSynapsesPerSegment)


    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the Connections instance using capnp.

      :param: Destination ConnectionsProto message builder
      """
      reader = ConnectionsProto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy


    def convertedRead(self, proto):
      """Initialize the Connections instance from the given ConnectionsProto
      reader.

      :param proto: ConnectionsProto message reader containing data from a
                    previously serialized Connections instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2

  %}

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }

  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  %pythoncode %{
    def mapSegmentsToCells(self, segments):
      segments = numpy.asarray(segments, dtype="uint32")
      return self._mapSegmentsToCells(segments)
  %}

  PyObject* _mapSegmentsToCells(PyObject *py_segments) const
  {
    nupic::NumpyVectorWeakRefT<nupic::UInt32> segments(py_segments);

    nupic::NumpyVectorT<nupic::UInt32> cellsOut(segments.size());
    self->mapSegmentsToCells(segments.begin(), segments.end(),
                             cellsOut.begin());

    return cellsOut.forPython();
  }

}

%extend nupic::algorithms::connections::Segment
{
  %pythoncode %{

    def __key(self):
      return self.flatIdx

    def __eq__(x, y):
      return x.__key() == y.__key()

    def __hash__(self):
      return hash(self.__key())

    def __str__(self):
      return "{0}-{1}".format(self.flatIdx)

    def __repr__(self):
      return str(self)

  %}
}

%extend nupic::algorithms::connections::Synapse
{
  %pythoncode %{

    def __key(self):
      return self.flatIdx

    def __eq__(x, y):
      return x.__key() == y.__key()

    def __hash__(self):
      return hash(self.__key())

    def __str__(self):
      return "{0}".format(self.flatIdx)

    def __repr__(self):
      return str(self)

  %}
}


//--------------------------------------------------------------------------------
// Temporal Memory
//--------------------------------------------------------------------------------
%inline %{
  template <typename IntType>
  inline PyObject* vectorToList(const vector<IntType> &cellIdxs)
  {
    PyObject *list = PyList_New(cellIdxs.size());
    for (size_t i = 0; i < cellIdxs.size(); i++)
    {
      PyObject *pyIdx = PyInt_FromLong(cellIdxs[i]);
      PyList_SET_ITEM(list, i, pyIdx);
    }
    return list;
  }
%}

%extend nupic::algorithms::temporal_memory::TemporalMemory
{
  %pythoncode %{
    def __init__(self,
                 columnDimensions=(2048,),
                 cellsPerColumn=32,
                 activationThreshold=13,
                 initialPermanence=0.21,
                 connectedPermanence=0.50,
                 minThreshold=10,
                 maxNewSynapseCount=20,
                 permanenceIncrement=0.10,
                 permanenceDecrement=0.10,
                 predictedSegmentDecrement=0.00,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255,
                 seed=42,
                 checkInputs=True):
      self.this = _ALGORITHMS.new_TemporalMemory()
      _ALGORITHMS.TemporalMemory_initialize(
        self, columnDimensions, cellsPerColumn, activationThreshold,
        initialPermanence, connectedPermanence,
        minThreshold, maxNewSynapseCount, permanenceIncrement,
        permanenceDecrement, predictedSegmentDecrement, seed,
        maxSegmentsPerCell, maxSynapsesPerSegment, checkInputs)

    def __getstate__(self):
      # Save the local attributes but override the C++ temporal memory with the
      # string representation.
      d = dict(self.__dict__)
      d["this"] = self.getCState()
      return d

    def __setstate__(self, state):
      # Create an empty C++ temporal memory and populate it from the serialized
      # string.
      self.this = _ALGORITHMS.new_TemporalMemory()
      if isinstance(state, str):
        self.loadFromString(state)
        self.valueToCategory = {}
      else:
        self.loadFromString(state["this"])
        # Use the rest of the state to set local Python attributes.
        del state["this"]
        self.__dict__.update(state)


    def activateCells(self,
                      activeColumns,
                      learn=True):
      """
      Calculate the active cells, using the current active columns and dendrite
      segments. Grow and reinforce synapses.

      @param activeColumns (iterable)
      Indices of active columns.

      @param learn (boolean)
      Whether to grow / reinforce / punish synapses.
      """
      columnsArray = numpy.array(sorted(activeColumns), dtype=uintDType)

      self.convertedActivateCells(columnsArray, learn)


    def compute(self, activeColumns, learn=True):
      """
      Perform one time step of the Temporal Memory algorithm.

      This method calls activateCells, then calls activateDendrites. Using
      the TemporalMemory via its compute method ensures that you'll always
      be able to call getPredictiveCells to get predictions for the next
      time step.

      @param activeColumns (iterable)
      Indices of active columns.

      @param learn (boolean)
      Whether or not learning is enabled.
      """
      activeColumnsArray = numpy.array(sorted(activeColumns), dtype=uintDType)
      self.convertedCompute(activeColumnsArray, learn)

    @classmethod
    def read(cls, proto):
      instance = cls()
      instance.convertedRead(proto)
      return instance

    def write(self, pyBuilder):
      """Serialize the TemporalMemory instance using capnp.

      :param: Destination TemporalMemoryProto message builder
      """
      reader = TemporalMemoryProto.from_bytes(self._writeAsCapnpPyBytes()) # copy
      pyBuilder.from_dict(reader.to_dict())  # copy


    def convertedRead(self, proto):
      """Initialize the TemporalMemory instance from the given TemporalMemoryProto
      reader.

      :param proto: TemporalMemoryProto message reader containing data from a
                    previously serialized TemporalMemory instance.

      """
      self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2

  %}

  inline PyObject* getActiveCells()
  {
    const vector<CellIdx> activeCells = self->getActiveCells();

    return nupic::NumpyVectorT<nupic::UInt32>(
      activeCells.size(), activeCells.data()
    ).forPython();
  }

  inline PyObject* getPredictiveCells()
  {
    const vector<CellIdx> predictiveCells = self->getPredictiveCells();

    return nupic::NumpyVectorT<nupic::UInt32>(
      predictiveCells.size(), predictiveCells.data()
    ).forPython();
  }

  inline PyObject* getWinnerCells()
  {
    const vector<CellIdx> winnerCells = self->getWinnerCells();

    return nupic::NumpyVectorT<nupic::UInt32>(
      winnerCells.size(), winnerCells.data()
    ).forPython();
  }

  inline PyObject* getActiveSegments()
  {
    const vector<UInt32> activeSegments = self->getActiveSegments();

    return nupic::NumpyVectorT<nupic::UInt32>(
      activeSegments.size(), activeSegments.data()
    ).forPython();
  }

  inline PyObject* getMatchingSegments()
  {
    const vector<UInt32> matchingSegments = self->getMatchingSegments();

    return nupic::NumpyVectorT<nupic::UInt32>(
      matchingSegments.size(), matchingSegments.data()
    ).forPython();
  }

  inline PyObject* cellsForColumn(UInt columnIdx)
  {
    const vector<CellIdx> cells = self->cellsForColumn(columnIdx);

    return nupic::NumpyVectorT<nupic::UInt32>(
      cells.size(), cells.data()
    ).forPython();
  }

  inline void convertedActivateCells(PyObject *py_activeColumns,
                                     bool learn)
  {
    PyArrayObject* _activeColumns =
      (PyArrayObject*) py_activeColumns;
    size_t activeColumnsSize =
      PyArray_DIMS(_activeColumns)[0];
    UInt32* activeColumns =
      (UInt32*)PyArray_DATA(_activeColumns);

    self->activateCells(activeColumnsSize,
                        activeColumns,
                        learn);
  }

  inline void convertedCompute(PyObject *py_activeColumns,
                               bool learn)
  {
    PyArrayObject* _activeColumns =
      (PyArrayObject*) py_activeColumns;
    size_t activeColumnsSize =
      PyArray_DIMS(_activeColumns)[0];
    UInt32* activeColumns =
      (UInt32*)PyArray_DATA(_activeColumns);

    self->compute(activeColumnsSize, activeColumns, learn);
  }

  inline PyObject* _writeAsCapnpPyBytes() const
  {
    return nupic::PyCapnpHelper::writeAsPyBytes(*self);
  }

  inline void _initFromCapnpPyBytes(PyObject* pyBytes)
  {
    nupic::PyCapnpHelper::initFromPyBytes(*self, pyBytes);
  }

  void loadFromString(const std::string& inString)
  {
    std::istringstream inStream(inString);
    self->load(inStream);
  }

  PyObject* getCState()
  {
    SharedPythonOStream py_s(self->persistentSize());
    std::ostream& s = py_s.getStream();
    // TODO: Consider writing floats as binary instead.
    s.flags(ios::scientific);
    s.precision(numeric_limits<double>::digits10 + 1);
    self->save(s);
    return py_s.close();
  }
}

%ignore nupic::algorithms::temporal_memory::TemporalMemory::getActiveCells;
%ignore nupic::algorithms::temporal_memory::TemporalMemory::getPredictiveCells;
%ignore nupic::algorithms::temporal_memory::TemporalMemory::getWinnerCells;
%ignore nupic::algorithms::temporal_memory::TemporalMemory::getActiveSegments;
%ignore nupic::algorithms::temporal_memory::TemporalMemory::getMatchingSegments;
%ignore nupic::algorithms::temporal_memory::TemporalMemory::cellsForColumn;


%include <nupic/algorithms/TemporalMemory.hpp>
