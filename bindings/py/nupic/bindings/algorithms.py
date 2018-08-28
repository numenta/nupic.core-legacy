# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_algorithms', [dirname(__file__)])
        except ImportError:
            import _algorithms
            return _algorithms
        if fp is not None:
            try:
                _mod = imp.load_module('_algorithms', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _algorithms = swig_import_helper()
    del swig_import_helper
else:
    import _algorithms
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


def _swig_setattr_nondynamic_method(set):
    def set_attr(self,name,value):
        if (name == "thisown"): return self.this.own(value)
        if hasattr(self,name) or (name == "this"):
            set(self,name,value)
        else:
            raise AttributeError("You cannot add attributes to %s" % self)
    return set_attr


try:
    import weakref
    weakref_proxy = weakref.proxy
except:
    weakref_proxy = lambda x: x


import os

_ALGORITHMS = _algorithms

uintDType = "uint32"


def forceRetentionOfGaborComputeWithinLibrary():
  """forceRetentionOfGaborComputeWithinLibrary()"""
  return _algorithms.forceRetentionOfGaborComputeWithinLibrary()

def forceRetentionOfImageSensorLiteLibrary():
  """forceRetentionOfImageSensorLiteLibrary()"""
  return _algorithms.forceRetentionOfImageSensorLiteLibrary()
class svm_problem(object):
    """Proxy of C++ nupic::algorithms::svm::svm_problem class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    recover_ = _swig_property(_algorithms.svm_problem_recover__get, _algorithms.svm_problem_recover__set)
    n_dims_ = _swig_property(_algorithms.svm_problem_n_dims__get, _algorithms.svm_problem_n_dims__set)
    x_ = _swig_property(_algorithms.svm_problem_x__get, _algorithms.svm_problem_x__set)
    y_ = _swig_property(_algorithms.svm_problem_y__get, _algorithms.svm_problem_y__set)
    def __init__(self, *args): 
        """
        __init__(self, n_dims, recover, arg4=0) -> svm_problem
        __init__(self, n_dims, size, recover, arg5=0) -> svm_problem
        __init__(self, inStream) -> svm_problem
        """
        this = _algorithms.new_svm_problem(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_svm_problem
    def size(self):
        """size(self) -> int"""
        return _algorithms.svm_problem_size(self)

    def n_dims(self):
        """n_dims(self) -> int"""
        return _algorithms.svm_problem_n_dims(self)

    def nnz(self, *args, **kwargs):
        """nnz(self, i) -> int"""
        return _algorithms.svm_problem_nnz(self, *args, **kwargs)

    def resize(self, *args, **kwargs):
        """resize(self, n)"""
        return _algorithms.svm_problem_resize(self, *args, **kwargs)

    def set_sample(self, *args, **kwargs):
        """set_sample(self, i, s)"""
        return _algorithms.svm_problem_set_sample(self, *args, **kwargs)

    def get_sample(self, *args, **kwargs):
        """get_sample(self, i) -> nupic::algorithms::svm::svm_problem::sample_type"""
        return _algorithms.svm_problem_get_sample(self, *args, **kwargs)

    def dense(self, *args, **kwargs):
        """dense(self, i, sv)"""
        return _algorithms.svm_problem_dense(self, *args, **kwargs)

    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_problem_persistent_size(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.svm_problem_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.svm_problem_load(self, *args, **kwargs)

    def get_samples(self, *args, **kwargs):
        """get_samples(self, samplesIn)"""
        return _algorithms.svm_problem_get_samples(self, *args, **kwargs)

svm_problem_swigregister = _algorithms.svm_problem_swigregister
svm_problem_swigregister(svm_problem)

class svm_problem01(object):
    """Proxy of C++ nupic::algorithms::svm::svm_problem01 class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    recover_ = _swig_property(_algorithms.svm_problem01_recover__get, _algorithms.svm_problem01_recover__set)
    n_dims_ = _swig_property(_algorithms.svm_problem01_n_dims__get, _algorithms.svm_problem01_n_dims__set)
    threshold_ = _swig_property(_algorithms.svm_problem01_threshold__get, _algorithms.svm_problem01_threshold__set)
    nnz_ = _swig_property(_algorithms.svm_problem01_nnz__get, _algorithms.svm_problem01_nnz__set)
    x_ = _swig_property(_algorithms.svm_problem01_x__get, _algorithms.svm_problem01_x__set)
    y_ = _swig_property(_algorithms.svm_problem01_y__get, _algorithms.svm_problem01_y__set)
    buf_ = _swig_property(_algorithms.svm_problem01_buf__get, _algorithms.svm_problem01_buf__set)
    def __init__(self, *args): 
        """
        __init__(self, n_dims, recover, threshold=.9) -> svm_problem01
        __init__(self, n_dims, size, recover, threshold=.9) -> svm_problem01
        __init__(self, inStream) -> svm_problem01
        """
        this = _algorithms.new_svm_problem01(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_svm_problem01
    def size(self):
        """size(self) -> size_t"""
        return _algorithms.svm_problem01_size(self)

    def n_dims(self):
        """n_dims(self) -> int"""
        return _algorithms.svm_problem01_n_dims(self)

    def nnz(self, *args, **kwargs):
        """nnz(self, i) -> int"""
        return _algorithms.svm_problem01_nnz(self, *args, **kwargs)

    def resize(self, *args, **kwargs):
        """resize(self, n)"""
        return _algorithms.svm_problem01_resize(self, *args, **kwargs)

    def set_sample(self, *args, **kwargs):
        """set_sample(self, i, s)"""
        return _algorithms.svm_problem01_set_sample(self, *args, **kwargs)

    def get_sample(self, *args, **kwargs):
        """get_sample(self, i) -> nupic::algorithms::svm::svm_problem01::sample_type"""
        return _algorithms.svm_problem01_get_sample(self, *args, **kwargs)

    def dense(self, *args, **kwargs):
        """dense(self, i, sv)"""
        return _algorithms.svm_problem01_dense(self, *args, **kwargs)

    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_problem01_persistent_size(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.svm_problem01_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.svm_problem01_load(self, *args, **kwargs)

    def get_samples(self, *args, **kwargs):
        """get_samples(self, samplesIn)"""
        return _algorithms.svm_problem01_get_samples(self, *args, **kwargs)

svm_problem01_swigregister = _algorithms.svm_problem01_swigregister
svm_problem01_swigregister(svm_problem01)

class decision_function(object):
    """Proxy of C++ nupic::algorithms::svm::decision_function class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> decision_function"""
        this = _algorithms.new_decision_function()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_decision_function
    alpha = _swig_property(_algorithms.decision_function_alpha_get, _algorithms.decision_function_alpha_set)
    rho = _swig_property(_algorithms.decision_function_rho_get, _algorithms.decision_function_rho_set)
decision_function_swigregister = _algorithms.decision_function_swigregister
decision_function_swigregister(decision_function)

class svm_model(object):
    """Proxy of C++ nupic::algorithms::svm::svm_model class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    n_dims_ = _swig_property(_algorithms.svm_model_n_dims__get, _algorithms.svm_model_n_dims__set)
    sv_mem = _swig_property(_algorithms.svm_model_sv_mem_get, _algorithms.svm_model_sv_mem_set)
    sv = _swig_property(_algorithms.svm_model_sv_get, _algorithms.svm_model_sv_set)
    sv_coef = _swig_property(_algorithms.svm_model_sv_coef_get, _algorithms.svm_model_sv_coef_set)
    rho = _swig_property(_algorithms.svm_model_rho_get, _algorithms.svm_model_rho_set)
    label = _swig_property(_algorithms.svm_model_label_get, _algorithms.svm_model_label_set)
    n_sv = _swig_property(_algorithms.svm_model_n_sv_get, _algorithms.svm_model_n_sv_set)
    probA = _swig_property(_algorithms.svm_model_probA_get, _algorithms.svm_model_probA_set)
    probB = _swig_property(_algorithms.svm_model_probB_get, _algorithms.svm_model_probB_set)
    w = _swig_property(_algorithms.svm_model_w_get, _algorithms.svm_model_w_set)
    def size(self):
        """size(self) -> int"""
        return _algorithms.svm_model_size(self)

    def n_dims(self):
        """n_dims(self) -> int"""
        return _algorithms.svm_model_n_dims(self)

    def n_class(self):
        """n_class(self) -> int"""
        return _algorithms.svm_model_n_class(self)

    def __init__(self): 
        """__init__(self) -> svm_model"""
        this = _algorithms.new_svm_model()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_svm_model
    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_model_persistent_size(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.svm_model_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.svm_model_load(self, *args, **kwargs)

    def get_support_vectors(self, *args, **kwargs):
        """get_support_vectors(self, svIn)"""
        return _algorithms.svm_model_get_support_vectors(self, *args, **kwargs)

    def get_support_vector_coefficients(self, *args, **kwargs):
        """get_support_vector_coefficients(self, svCoeffIn)"""
        return _algorithms.svm_model_get_support_vector_coefficients(self, *args, **kwargs)

    def get_hyperplanes(self):
        """get_hyperplanes(self) -> PyObject *"""
        return _algorithms.svm_model_get_hyperplanes(self)

svm_model_swigregister = _algorithms.svm_model_swigregister
svm_model_swigregister(svm_model)

class QMatrix(object):
    """Proxy of C++ nupic::algorithms::svm::QMatrix class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """__init__(self, prob, g, kernel, cache_size) -> QMatrix"""
        this = _algorithms.new_QMatrix(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_QMatrix
    def get_Q(self, *args, **kwargs):
        """get_Q(self, i, len) -> float *"""
        return _algorithms.QMatrix_get_Q(self, *args, **kwargs)

    def get_QD(self):
        """get_QD(self) -> float *"""
        return _algorithms.QMatrix_get_QD(self)

    def swap_index(self, *args, **kwargs):
        """swap_index(self, i, j)"""
        return _algorithms.QMatrix_swap_index(self, *args, **kwargs)

    def dot(self, *args, **kwargs):
        """dot(self, i, j) -> nupic::algorithms::svm::QMatrix::feature_type"""
        return _algorithms.QMatrix_dot(self, *args, **kwargs)

    def linear_kernel(self, *args, **kwargs):
        """linear_kernel(self, i, j) -> float"""
        return _algorithms.QMatrix_linear_kernel(self, *args, **kwargs)

    def rbf_kernel(self, *args, **kwargs):
        """rbf_kernel(self, i, j) -> float"""
        return _algorithms.QMatrix_rbf_kernel(self, *args, **kwargs)

QMatrix_swigregister = _algorithms.QMatrix_swigregister
QMatrix_swigregister(QMatrix)

class QMatrix01(object):
    """Proxy of C++ nupic::algorithms::svm::QMatrix01 class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """__init__(self, prob, g, kernel, cache_size) -> QMatrix01"""
        this = _algorithms.new_QMatrix01(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_QMatrix01
    def get_Q(self, *args, **kwargs):
        """get_Q(self, i, len) -> float *"""
        return _algorithms.QMatrix01_get_Q(self, *args, **kwargs)

    def get_QD(self):
        """get_QD(self) -> float *"""
        return _algorithms.QMatrix01_get_QD(self)

    def swap_index(self, *args, **kwargs):
        """swap_index(self, i, j)"""
        return _algorithms.QMatrix01_swap_index(self, *args, **kwargs)

    def dot(self, *args, **kwargs):
        """dot(self, i, j) -> nupic::algorithms::svm::QMatrix01::feature_type"""
        return _algorithms.QMatrix01_dot(self, *args, **kwargs)

    def linear_kernel(self, *args, **kwargs):
        """linear_kernel(self, i, j) -> float"""
        return _algorithms.QMatrix01_linear_kernel(self, *args, **kwargs)

    def rbf_kernel(self, *args, **kwargs):
        """rbf_kernel(self, i, j) -> float"""
        return _algorithms.QMatrix01_rbf_kernel(self, *args, **kwargs)

QMatrix01_swigregister = _algorithms.QMatrix01_swigregister
QMatrix01_swigregister(QMatrix01)

class svm_parameter(object):
    """Proxy of C++ nupic::algorithms::svm::svm_parameter class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """__init__(self, k, p, g, c, e, cs, s) -> svm_parameter"""
        this = _algorithms.new_svm_parameter(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    kernel = _swig_property(_algorithms.svm_parameter_kernel_get, _algorithms.svm_parameter_kernel_set)
    probability = _swig_property(_algorithms.svm_parameter_probability_get, _algorithms.svm_parameter_probability_set)
    gamma = _swig_property(_algorithms.svm_parameter_gamma_get, _algorithms.svm_parameter_gamma_set)
    C = _swig_property(_algorithms.svm_parameter_C_get, _algorithms.svm_parameter_C_set)
    eps = _swig_property(_algorithms.svm_parameter_eps_get, _algorithms.svm_parameter_eps_set)
    cache_size = _swig_property(_algorithms.svm_parameter_cache_size_get, _algorithms.svm_parameter_cache_size_set)
    shrinking = _swig_property(_algorithms.svm_parameter_shrinking_get, _algorithms.svm_parameter_shrinking_set)
    weight_label = _swig_property(_algorithms.svm_parameter_weight_label_get, _algorithms.svm_parameter_weight_label_set)
    weight = _swig_property(_algorithms.svm_parameter_weight_get, _algorithms.svm_parameter_weight_set)
    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_parameter_persistent_size(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.svm_parameter_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.svm_parameter_load(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_svm_parameter
svm_parameter_swigregister = _algorithms.svm_parameter_swigregister
svm_parameter_swigregister(svm_parameter)

class svm_std_traits(object):
    """Proxy of C++ nupic::algorithms::svm::svm_std_traits class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> svm_std_traits"""
        this = _algorithms.new_svm_std_traits()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_svm_std_traits
svm_std_traits_swigregister = _algorithms.svm_std_traits_swigregister
svm_std_traits_swigregister(svm_std_traits)

class svm_01_traits(object):
    """Proxy of C++ nupic::algorithms::svm::svm_01_traits class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> svm_01_traits"""
        this = _algorithms.new_svm_01_traits()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_svm_01_traits
svm_01_traits_swigregister = _algorithms.svm_01_traits_swigregister
svm_01_traits_swigregister(svm_01_traits)

class svm_dense(object):
    """Proxy of C++ nupic::algorithms::svm::svm_dense class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """
        __init__(self, kernel=0, n_dims=0, threshold=.9, cache_size=100, shrinking=1, probability=False, 
            seed=-1) -> svm_dense
        """
        this = _algorithms.new_svm_dense(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    def train(self, *args, **kwargs):
        """train(self, gamma, C, eps)"""
        return _algorithms.svm_dense_train(self, *args, **kwargs)

    def get_problem(self):
        """get_problem(self) -> svm_problem"""
        return _algorithms.svm_dense_get_problem(self)

    def get_model(self):
        """get_model(self) -> svm_model"""
        return _algorithms.svm_dense_get_model(self)

    def get_parameter(self):
        """get_parameter(self) -> svm_parameter"""
        return _algorithms.svm_dense_get_parameter(self)

    def discard_problem(self):
        """discard_problem(self)"""
        return _algorithms.svm_dense_discard_problem(self)

    def cross_validation(self, *args, **kwargs):
        """cross_validation(self, n_fold, gamma, C, eps) -> float"""
        return _algorithms.svm_dense_cross_validation(self, *args, **kwargs)

    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_dense_persistent_size(self)

    def __getstate__(self):
        """__getstate__(self) -> PyObject *"""
        return _algorithms.svm_dense___getstate__(self)

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




    def loadFromString(self, *args, **kwargs):
        """loadFromString(self, inString)"""
        return _algorithms.svm_dense_loadFromString(self, *args, **kwargs)

    def add_sample(self, *args, **kwargs):
        """add_sample(self, y_val, x_vector)"""
        return _algorithms.svm_dense_add_sample(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """predict(self, x_vector) -> float"""
        return _algorithms.svm_dense_predict(self, *args, **kwargs)

    def predict_probability(self, *args, **kwargs):
        """predict_probability(self, x_vector, proba_vector) -> float"""
        return _algorithms.svm_dense_predict_probability(self, *args, **kwargs)

    def save(self, *args):
        """
        save(self, outStream)
        save(self, filename)
        """
        return _algorithms.svm_dense_save(self, *args)

    def load(self, *args):
        """
        load(self, inStream)
        load(self, filename)
        """
        return _algorithms.svm_dense_load(self, *args)

    def cross_validate(self, *args, **kwargs):
        """cross_validate(self, n_fold, gamma, C, eps) -> float"""
        return _algorithms.svm_dense_cross_validate(self, *args, **kwargs)

    def trainReleaseGIL(self, *args, **kwargs):
        """trainReleaseGIL(self, gamma, C, eps)"""
        return _algorithms.svm_dense_trainReleaseGIL(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_svm_dense
svm_dense_swigregister = _algorithms.svm_dense_swigregister
svm_dense_swigregister(svm_dense)

class svm_01(object):
    """Proxy of C++ nupic::algorithms::svm::svm_01 class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        """
        __init__(self, kernel=0, n_dims=0, threshold=.9, cache_size=100, shrinking=1, probability=False, 
            seed=-1) -> svm_01
        """
        this = _algorithms.new_svm_01(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    def train(self, *args, **kwargs):
        """train(self, gamma, C, eps)"""
        return _algorithms.svm_01_train(self, *args, **kwargs)

    def get_problem(self):
        """get_problem(self) -> svm_problem01"""
        return _algorithms.svm_01_get_problem(self)

    def get_model(self):
        """get_model(self) -> svm_model"""
        return _algorithms.svm_01_get_model(self)

    def get_parameter(self):
        """get_parameter(self) -> svm_parameter"""
        return _algorithms.svm_01_get_parameter(self)

    def discard_problem(self):
        """discard_problem(self)"""
        return _algorithms.svm_01_discard_problem(self)

    def cross_validation(self, *args, **kwargs):
        """cross_validation(self, n_fold, gamma, C, eps) -> float"""
        return _algorithms.svm_01_cross_validation(self, *args, **kwargs)

    def persistent_size(self):
        """persistent_size(self) -> size_t"""
        return _algorithms.svm_01_persistent_size(self)

    def __getstate__(self):
        """__getstate__(self) -> PyObject *"""
        return _algorithms.svm_01___getstate__(self)

    def __setstate__(self, inString):
      self.this = _ALGORITHMS.new_svm_01()
      self.thisown = 1
      self.loadFromString(inString)

    def add_sample(self, *args, **kwargs):
        """add_sample(self, y_val, x_vector)"""
        return _algorithms.svm_01_add_sample(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """predict(self, x_vector) -> float"""
        return _algorithms.svm_01_predict(self, *args, **kwargs)

    def predict_probability(self, *args, **kwargs):
        """predict_probability(self, x_vector, proba_vector) -> float"""
        return _algorithms.svm_01_predict_probability(self, *args, **kwargs)

    def cross_validate(self, *args, **kwargs):
        """cross_validate(self, n_fold, gamma, C, eps) -> float"""
        return _algorithms.svm_01_cross_validate(self, *args, **kwargs)

    def trainReleaseGIL(self, *args, **kwargs):
        """trainReleaseGIL(self, gamma, C, eps)"""
        return _algorithms.svm_01_trainReleaseGIL(self, *args, **kwargs)

    def save(self, *args):
        """
        save(self, outStream)
        save(self, filename)
        """
        return _algorithms.svm_01_save(self, *args)

    def load(self, *args):
        """
        load(self, inStream)
        load(self, filename)
        """
        return _algorithms.svm_01_load(self, *args)

    __swig_destroy__ = _algorithms.delete_svm_01
svm_01_swigregister = _algorithms.svm_01_swigregister
svm_01_swigregister(svm_01)

class Float32SeparableConvolution2D(object):
    """Proxy of C++ SeparableConvolution2D<(float)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    nrows_ = _swig_property(_algorithms.Float32SeparableConvolution2D_nrows__get, _algorithms.Float32SeparableConvolution2D_nrows__set)
    ncols_ = _swig_property(_algorithms.Float32SeparableConvolution2D_ncols__get, _algorithms.Float32SeparableConvolution2D_ncols__set)
    f1_size_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f1_size__get, _algorithms.Float32SeparableConvolution2D_f1_size__set)
    f2_size_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f2_size__get, _algorithms.Float32SeparableConvolution2D_f2_size__set)
    f1_end_j_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f1_end_j__get, _algorithms.Float32SeparableConvolution2D_f1_end_j__set)
    f2_end_i_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f2_end_i__get, _algorithms.Float32SeparableConvolution2D_f2_end_i__set)
    f1_middle_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f1_middle__get, _algorithms.Float32SeparableConvolution2D_f1_middle__set)
    f2_middle_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f2_middle__get, _algorithms.Float32SeparableConvolution2D_f2_middle__set)
    f1_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f1__get, _algorithms.Float32SeparableConvolution2D_f1__set)
    f2_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f2__get, _algorithms.Float32SeparableConvolution2D_f2__set)
    f1_end_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f1_end__get, _algorithms.Float32SeparableConvolution2D_f1_end__set)
    f2_end_ = _swig_property(_algorithms.Float32SeparableConvolution2D_f2_end__get, _algorithms.Float32SeparableConvolution2D_f2_end__set)
    buffer_ = _swig_property(_algorithms.Float32SeparableConvolution2D_buffer__get, _algorithms.Float32SeparableConvolution2D_buffer__set)
    def __init__(self): 
        """__init__(self) -> Float32SeparableConvolution2D"""
        this = _algorithms.new_Float32SeparableConvolution2D()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_Float32SeparableConvolution2D
    def init(self, *args):
        """
        init(self, nrows, ncols, f1_size, f2_size, f1, f2)
        init(self, nrows, ncols, f1_size, f2_size, pyF1, pyF2)
        """
        return _algorithms.Float32SeparableConvolution2D_init(self, *args)

    def compute(self, *args):
        """
        compute(self, data, convolved, rotated45=False)
        compute(self, pyData, pyConvolved, rotated45=False)
        """
        return _algorithms.Float32SeparableConvolution2D_compute(self, *args)

    def getBuffer(self, *args, **kwargs):
        """getBuffer(self, pyBuffer)"""
        return _algorithms.Float32SeparableConvolution2D_getBuffer(self, *args, **kwargs)

Float32SeparableConvolution2D_swigregister = _algorithms.Float32SeparableConvolution2D_swigregister
Float32SeparableConvolution2D_swigregister(Float32SeparableConvolution2D)

cos45 = _algorithms.cos45
class Float32Rotation45(object):
    """Proxy of C++ Rotation45<(float)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    srow_ = _swig_property(_algorithms.Float32Rotation45_srow__get, _algorithms.Float32Rotation45_srow__set)
    scol_ = _swig_property(_algorithms.Float32Rotation45_scol__get, _algorithms.Float32Rotation45_scol__set)
    offset_ = _swig_property(_algorithms.Float32Rotation45_offset__get, _algorithms.Float32Rotation45_offset__set)
    def round(self, *args, **kwargs):
        """round(self, x) -> float"""
        return _algorithms.Float32Rotation45_round(self, *args, **kwargs)

    def rotate(self, *args):
        """
        rotate(self, original, rotated, nrows, ncols, z)
        rotate(self, pyOriginal, pyRotated, nrows, ncols, z)
        """
        return _algorithms.Float32Rotation45_rotate(self, *args)

    def unrotate(self, *args):
        """
        unrotate(self, unrotated, rotated, nrows, ncols, z)
        unrotate(self, pyUnrotated, pyRotated, nrows, ncols, z)
        """
        return _algorithms.Float32Rotation45_unrotate(self, *args)

    def __init__(self): 
        """__init__(self) -> Float32Rotation45"""
        this = _algorithms.new_Float32Rotation45()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_Float32Rotation45
Float32Rotation45_swigregister = _algorithms.Float32Rotation45_swigregister
Float32Rotation45_swigregister(Float32Rotation45)

class Float32Erosion(object):
    """Proxy of C++ Erosion<(float)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    nrows_ = _swig_property(_algorithms.Float32Erosion_nrows__get, _algorithms.Float32Erosion_nrows__set)
    ncols_ = _swig_property(_algorithms.Float32Erosion_ncols__get, _algorithms.Float32Erosion_ncols__set)
    buffer_ = _swig_property(_algorithms.Float32Erosion_buffer__get, _algorithms.Float32Erosion_buffer__set)
    def __init__(self): 
        """__init__(self) -> Float32Erosion"""
        this = _algorithms.new_Float32Erosion()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_Float32Erosion
    def init(self, *args):
        """
        init(self, nrows, ncols)
        init(self, nrows, ncols)
        """
        return _algorithms.Float32Erosion_init(self, *args)

    def compute(self, *args):
        """
        compute(self, data, eroded, iterations, dilate=False)
        compute(self, pyData, pyEroded, iterations, dilate=False)
        """
        return _algorithms.Float32Erosion_compute(self, *args)

    def getBuffer(self, *args, **kwargs):
        """getBuffer(self, pyBuffer)"""
        return _algorithms.Float32Erosion_getBuffer(self, *args, **kwargs)

Float32Erosion_swigregister = _algorithms.Float32Erosion_swigregister
Float32Erosion_swigregister(Float32Erosion)


def computeAlpha(*args):
  """
    computeAlpha(xstep, ystep, widthS, heightS, imageWidth, imageHeight, xcount, ycount, weightWidth, 
        sharpness, data, values, counts, weights)
    computeAlpha(xstep, ystep, widthS, heightS, imageWidth, imageHeight, xcount, ycount, weightWidth, 
        sharpness, pyData, pyValues, pyCounts, pyWeights)
    """
  return _algorithms.computeAlpha(*args)
class Byte_Vector(object):
    """Proxy of C++ std::vector<(nupic::Byte)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _algorithms.Byte_Vector_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _algorithms.Byte_Vector___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _algorithms.Byte_Vector___bool__(self)

    def __len__(self):
        """__len__(self) -> std::vector< char >::size_type"""
        return _algorithms.Byte_Vector___len__(self)

    def pop(self):
        """pop(self) -> std::vector< char >::value_type"""
        return _algorithms.Byte_Vector_pop(self)

    def __getslice__(self, *args, **kwargs):
        """__getslice__(self, i, j) -> Byte_Vector"""
        return _algorithms.Byte_Vector___getslice__(self, *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        """__setslice__(self, i, j, v=std::vector< char,std::allocator< char > >())"""
        return _algorithms.Byte_Vector___setslice__(self, *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        """__delslice__(self, i, j)"""
        return _algorithms.Byte_Vector___delslice__(self, *args, **kwargs)

    def __delitem__(self, *args):
        """
        __delitem__(self, i)
        __delitem__(self, slice)
        """
        return _algorithms.Byte_Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        """
        __getitem__(self, slice) -> Byte_Vector
        __getitem__(self, i) -> std::vector< char >::value_type const &
        """
        return _algorithms.Byte_Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        """
        __setitem__(self, slice, v)
        __setitem__(self, slice)
        __setitem__(self, i, x)
        """
        return _algorithms.Byte_Vector___setitem__(self, *args)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _algorithms.Byte_Vector_append(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.Byte_Vector_empty(self)

    def size(self):
        """size(self) -> std::vector< char >::size_type"""
        return _algorithms.Byte_Vector_size(self)

    def clear(self):
        """clear(self)"""
        return _algorithms.Byte_Vector_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _algorithms.Byte_Vector_swap(self, *args, **kwargs)

    def get_allocator(self):
        """get_allocator(self) -> std::vector< char >::allocator_type"""
        return _algorithms.Byte_Vector_get_allocator(self)

    def begin(self):
        """begin(self) -> std::vector< char >::iterator"""
        return _algorithms.Byte_Vector_begin(self)

    def end(self):
        """end(self) -> std::vector< char >::iterator"""
        return _algorithms.Byte_Vector_end(self)

    def rbegin(self):
        """rbegin(self) -> std::vector< char >::reverse_iterator"""
        return _algorithms.Byte_Vector_rbegin(self)

    def rend(self):
        """rend(self) -> std::vector< char >::reverse_iterator"""
        return _algorithms.Byte_Vector_rend(self)

    def pop_back(self):
        """pop_back(self)"""
        return _algorithms.Byte_Vector_pop_back(self)

    def erase(self, *args):
        """
        erase(self, pos) -> std::vector< char >::iterator
        erase(self, first, last) -> std::vector< char >::iterator
        """
        return _algorithms.Byte_Vector_erase(self, *args)

    def __init__(self, *args): 
        """
        __init__(self) -> Byte_Vector
        __init__(self, arg2) -> Byte_Vector
        __init__(self, size) -> Byte_Vector
        __init__(self, size, value) -> Byte_Vector
        """
        this = _algorithms.new_Byte_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args, **kwargs):
        """push_back(self, x)"""
        return _algorithms.Byte_Vector_push_back(self, *args, **kwargs)

    def front(self):
        """front(self) -> std::vector< char >::value_type const &"""
        return _algorithms.Byte_Vector_front(self)

    def back(self):
        """back(self) -> std::vector< char >::value_type const &"""
        return _algorithms.Byte_Vector_back(self)

    def assign(self, *args, **kwargs):
        """assign(self, n, x)"""
        return _algorithms.Byte_Vector_assign(self, *args, **kwargs)

    def resize(self, *args):
        """
        resize(self, new_size)
        resize(self, new_size, x)
        """
        return _algorithms.Byte_Vector_resize(self, *args)

    def insert(self, *args):
        """
        insert(self, pos, x) -> std::vector< char >::iterator
        insert(self, pos, n, x)
        """
        return _algorithms.Byte_Vector_insert(self, *args)

    def reserve(self, *args, **kwargs):
        """reserve(self, n)"""
        return _algorithms.Byte_Vector_reserve(self, *args, **kwargs)

    def capacity(self):
        """capacity(self) -> std::vector< char >::size_type"""
        return _algorithms.Byte_Vector_capacity(self)

    __swig_destroy__ = _algorithms.delete_Byte_Vector
Byte_Vector_swigregister = _algorithms.Byte_Vector_swigregister
Byte_Vector_swigregister(Byte_Vector)

class ByteVector(Byte_Vector):
    """Proxy of C++ nupic::ByteVector class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, n=0): 
        """__init__(self, n=0) -> ByteVector"""
        this = _algorithms.new_ByteVector(n)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_ByteVector
ByteVector_swigregister = _algorithms.ByteVector_swigregister
ByteVector_swigregister(ByteVector)

class Size_T_Vector(object):
    """Proxy of C++ std::vector<(size_t)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _algorithms.Size_T_Vector_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _algorithms.Size_T_Vector___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _algorithms.Size_T_Vector___bool__(self)

    def __len__(self):
        """__len__(self) -> std::vector< size_t >::size_type"""
        return _algorithms.Size_T_Vector___len__(self)

    def pop(self):
        """pop(self) -> std::vector< size_t >::value_type"""
        return _algorithms.Size_T_Vector_pop(self)

    def __getslice__(self, *args, **kwargs):
        """__getslice__(self, i, j) -> Size_T_Vector"""
        return _algorithms.Size_T_Vector___getslice__(self, *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        """__setslice__(self, i, j, v=std::vector< size_t,std::allocator< size_t > >())"""
        return _algorithms.Size_T_Vector___setslice__(self, *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        """__delslice__(self, i, j)"""
        return _algorithms.Size_T_Vector___delslice__(self, *args, **kwargs)

    def __delitem__(self, *args):
        """
        __delitem__(self, i)
        __delitem__(self, slice)
        """
        return _algorithms.Size_T_Vector___delitem__(self, *args)

    def __getitem__(self, *args):
        """
        __getitem__(self, slice) -> Size_T_Vector
        __getitem__(self, i) -> std::vector< size_t >::value_type const &
        """
        return _algorithms.Size_T_Vector___getitem__(self, *args)

    def __setitem__(self, *args):
        """
        __setitem__(self, slice, v)
        __setitem__(self, slice)
        __setitem__(self, i, x)
        """
        return _algorithms.Size_T_Vector___setitem__(self, *args)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _algorithms.Size_T_Vector_append(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.Size_T_Vector_empty(self)

    def size(self):
        """size(self) -> std::vector< size_t >::size_type"""
        return _algorithms.Size_T_Vector_size(self)

    def clear(self):
        """clear(self)"""
        return _algorithms.Size_T_Vector_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _algorithms.Size_T_Vector_swap(self, *args, **kwargs)

    def get_allocator(self):
        """get_allocator(self) -> std::vector< size_t >::allocator_type"""
        return _algorithms.Size_T_Vector_get_allocator(self)

    def begin(self):
        """begin(self) -> std::vector< size_t >::iterator"""
        return _algorithms.Size_T_Vector_begin(self)

    def end(self):
        """end(self) -> std::vector< size_t >::iterator"""
        return _algorithms.Size_T_Vector_end(self)

    def rbegin(self):
        """rbegin(self) -> std::vector< size_t >::reverse_iterator"""
        return _algorithms.Size_T_Vector_rbegin(self)

    def rend(self):
        """rend(self) -> std::vector< size_t >::reverse_iterator"""
        return _algorithms.Size_T_Vector_rend(self)

    def pop_back(self):
        """pop_back(self)"""
        return _algorithms.Size_T_Vector_pop_back(self)

    def erase(self, *args):
        """
        erase(self, pos) -> std::vector< size_t >::iterator
        erase(self, first, last) -> std::vector< size_t >::iterator
        """
        return _algorithms.Size_T_Vector_erase(self, *args)

    def __init__(self, *args): 
        """
        __init__(self) -> Size_T_Vector
        __init__(self, arg2) -> Size_T_Vector
        __init__(self, size) -> Size_T_Vector
        __init__(self, size, value) -> Size_T_Vector
        """
        this = _algorithms.new_Size_T_Vector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args, **kwargs):
        """push_back(self, x)"""
        return _algorithms.Size_T_Vector_push_back(self, *args, **kwargs)

    def front(self):
        """front(self) -> std::vector< size_t >::value_type const &"""
        return _algorithms.Size_T_Vector_front(self)

    def back(self):
        """back(self) -> std::vector< size_t >::value_type const &"""
        return _algorithms.Size_T_Vector_back(self)

    def assign(self, *args, **kwargs):
        """assign(self, n, x)"""
        return _algorithms.Size_T_Vector_assign(self, *args, **kwargs)

    def resize(self, *args):
        """
        resize(self, new_size)
        resize(self, new_size, x)
        """
        return _algorithms.Size_T_Vector_resize(self, *args)

    def insert(self, *args):
        """
        insert(self, pos, x) -> std::vector< size_t >::iterator
        insert(self, pos, n, x)
        """
        return _algorithms.Size_T_Vector_insert(self, *args)

    def reserve(self, *args, **kwargs):
        """reserve(self, n)"""
        return _algorithms.Size_T_Vector_reserve(self, *args, **kwargs)

    def capacity(self):
        """capacity(self) -> std::vector< size_t >::size_type"""
        return _algorithms.Size_T_Vector_capacity(self)

    __swig_destroy__ = _algorithms.delete_Size_T_Vector
Size_T_Vector_swigregister = _algorithms.Size_T_Vector_swigregister
Size_T_Vector_swigregister(Size_T_Vector)


def non_zeros_ui8(*args, **kwargs):
  """non_zeros_ui8(py_x, py_y) -> nupic::UInt32"""
  return _algorithms.non_zeros_ui8(*args, **kwargs)

def non_zeros_i32(*args, **kwargs):
  """non_zeros_i32(py_x, py_y) -> nupic::UInt32"""
  return _algorithms.non_zeros_i32(*args, **kwargs)

def non_zeros_f32(*args, **kwargs):
  """non_zeros_f32(py_x, py_y) -> nupic::UInt32"""
  return _algorithms.non_zeros_f32(*args, **kwargs)

def rightVecProdAtIndices(*args, **kwargs):
  """rightVecProdAtIndices(py_ind, py_x, py_y)"""
  return _algorithms.rightVecProdAtIndices(*args, **kwargs)

def getSegmentActivityLevel(*args, **kwargs):
  """getSegmentActivityLevel(py_seg, py_state, connectedSynapsesOnly, connectedPerm) -> nupic::UInt32"""
  return _algorithms.getSegmentActivityLevel(*args, **kwargs)

def isSegmentActive(*args, **kwargs):
  """isSegmentActive(py_seg, py_state, connectedPerm, activationThreshold) -> bool"""
  return _algorithms.isSegmentActive(*args, **kwargs)
class CState(object):
    """Proxy of C++ nupic::algorithms::Cells4::CState class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    VERSION = _algorithms.CState_VERSION
    def __init__(self): 
        """__init__(self) -> CState"""
        this = _algorithms.new_CState()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_CState
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.CState___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.CState___ne__(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        """initialize(self, nCells) -> bool"""
        return _algorithms.CState_initialize(self, *args, **kwargs)

    def usePythonMemory(self, *args, **kwargs):
        """usePythonMemory(self, pData, nCells)"""
        return _algorithms.CState_usePythonMemory(self, *args, **kwargs)

    def isSet(self, *args, **kwargs):
        """isSet(self, cellIdx) -> bool"""
        return _algorithms.CState_isSet(self, *args, **kwargs)

    def set(self, *args, **kwargs):
        """set(self, cellIdx)"""
        return _algorithms.CState_set(self, *args, **kwargs)

    def resetAll(self):
        """resetAll(self)"""
        return _algorithms.CState_resetAll(self)

    def arrayPtr(self):
        """arrayPtr(self) -> nupic::Byte *"""
        return _algorithms.CState_arrayPtr(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.CState_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.CState_load(self, *args, **kwargs)

    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.CState_version(self)

CState_swigregister = _algorithms.CState_swigregister
CState_swigregister(CState)

class CStateIndexed(CState):
    """Proxy of C++ nupic::algorithms::Cells4::CStateIndexed class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    VERSION = _algorithms.CStateIndexed_VERSION
    def __init__(self): 
        """__init__(self) -> CStateIndexed"""
        this = _algorithms.new_CStateIndexed()
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.CStateIndexed___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.CStateIndexed___ne__(self, *args, **kwargs)

    def cellsOn(self, fSorted=False):
        """cellsOn(self, fSorted=False) -> VectorOfUInt32"""
        return _algorithms.CStateIndexed_cellsOn(self, fSorted)

    def set(self, *args, **kwargs):
        """set(self, cellIdx)"""
        return _algorithms.CStateIndexed_set(self, *args, **kwargs)

    def resetAll(self):
        """resetAll(self)"""
        return _algorithms.CStateIndexed_resetAll(self)

    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.CStateIndexed_version(self)

    __swig_destroy__ = _algorithms.delete_CStateIndexed
CStateIndexed_swigregister = _algorithms.CStateIndexed_swigregister
CStateIndexed_swigregister(CStateIndexed)

class InSynapseOrder(object):
    """Proxy of C++ nupic::algorithms::Cells4::InSynapseOrder class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __call__(self, *args, **kwargs):
        """__call__(self, a, b) -> bool"""
        return _algorithms.InSynapseOrder___call__(self, *args, **kwargs)

    def __init__(self): 
        """__init__(self) -> InSynapseOrder"""
        this = _algorithms.new_InSynapseOrder()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_InSynapseOrder
InSynapseOrder_swigregister = _algorithms.InSynapseOrder_swigregister
InSynapseOrder_swigregister(InSynapseOrder)
cvar = _algorithms.cvar
_numTiers = cvar._numTiers
_dutyCycleTiers = cvar._dutyCycleTiers
_dutyCycleAlphas = cvar._dutyCycleAlphas

class Segment(object):
    """Proxy of C++ nupic::algorithms::Cells4::Segment class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    _totalActivations = _swig_property(_algorithms.Segment__totalActivations_get, _algorithms.Segment__totalActivations_set)
    _positiveActivations = _swig_property(_algorithms.Segment__positiveActivations_get, _algorithms.Segment__positiveActivations_set)
    _lastActiveIteration = _swig_property(_algorithms.Segment__lastActiveIteration_get, _algorithms.Segment__lastActiveIteration_set)
    _lastPosDutyCycle = _swig_property(_algorithms.Segment__lastPosDutyCycle_get, _algorithms.Segment__lastPosDutyCycle_set)
    _lastPosDutyCycleIteration = _swig_property(_algorithms.Segment__lastPosDutyCycleIteration_get, _algorithms.Segment__lastPosDutyCycleIteration_set)
    def __eq__(self, *args, **kwargs):
        """__eq__(self, o) -> bool"""
        return _algorithms.Segment___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, o) -> bool"""
        return _algorithms.Segment___ne__(self, *args, **kwargs)

    def __init__(self, *args): 
        """
        __init__(self) -> Segment
        __init__(self, _s, frequency, seqSegFlag, permConnected, iteration) -> Segment
        __init__(self, o) -> Segment
        """
        this = _algorithms.new_Segment(*args)
        try: self.this.append(this)
        except: self.this = this
    def invariants(self):
        """invariants(self) -> bool"""
        return _algorithms.Segment_invariants(self)

    def checkConnected(self, *args, **kwargs):
        """checkConnected(self, permConnected) -> bool"""
        return _algorithms.Segment_checkConnected(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.Segment_empty(self)

    def size(self):
        """size(self) -> size_t"""
        return _algorithms.Segment_size(self)

    def isSequenceSegment(self):
        """isSequenceSegment(self) -> bool"""
        return _algorithms.Segment_isSequenceSegment(self)

    def frequency(self):
        """frequency(self) -> nupic::Real &"""
        return _algorithms.Segment_frequency(self)

    def getFrequency(self):
        """getFrequency(self) -> nupic::Real"""
        return _algorithms.Segment_getFrequency(self)

    def nConnected(self):
        """nConnected(self) -> nupic::UInt"""
        return _algorithms.Segment_nConnected(self)

    def getTotalActivations(self):
        """getTotalActivations(self) -> nupic::UInt"""
        return _algorithms.Segment_getTotalActivations(self)

    def getPositiveActivations(self):
        """getPositiveActivations(self) -> nupic::UInt"""
        return _algorithms.Segment_getPositiveActivations(self)

    def getLastActiveIteration(self):
        """getLastActiveIteration(self) -> nupic::UInt"""
        return _algorithms.Segment_getLastActiveIteration(self)

    def getLastPosDutyCycle(self):
        """getLastPosDutyCycle(self) -> nupic::Real"""
        return _algorithms.Segment_getLastPosDutyCycle(self)

    def getLastPosDutyCycleIteration(self):
        """getLastPosDutyCycleIteration(self) -> nupic::UInt"""
        return _algorithms.Segment_getLastPosDutyCycleIteration(self)

    def has(self, *args, **kwargs):
        """has(self, srcCellIdx) -> bool"""
        return _algorithms.Segment_has(self, *args, **kwargs)

    def setPermanence(self, *args, **kwargs):
        """setPermanence(self, idx, val)"""
        return _algorithms.Segment_setPermanence(self, *args, **kwargs)

    def getPermanence(self, *args, **kwargs):
        """getPermanence(self, idx) -> nupic::Real"""
        return _algorithms.Segment_getPermanence(self, *args, **kwargs)

    def getSrcCellIdx(self, *args, **kwargs):
        """getSrcCellIdx(self, idx) -> nupic::UInt"""
        return _algorithms.Segment_getSrcCellIdx(self, *args, **kwargs)

    def getSrcCellIndices(self, *args, **kwargs):
        """getSrcCellIndices(self, srcCells)"""
        return _algorithms.Segment_getSrcCellIndices(self, *args, **kwargs)

    def clear(self):
        """clear(self)"""
        return _algorithms.Segment_clear(self)

    def addSynapses(self, *args, **kwargs):
        """addSynapses(self, srcCells, initStrength, permConnected)"""
        return _algorithms.Segment_addSynapses(self, *args, **kwargs)

    def recomputeConnected(self, *args, **kwargs):
        """recomputeConnected(self, permConnected)"""
        return _algorithms.Segment_recomputeConnected(self, *args, **kwargs)

    def decaySynapses2(self, *args, **kwargs):
        """decaySynapses2(self, decay, removed, permConnected)"""
        return _algorithms.Segment_decaySynapses2(self, *args, **kwargs)

    def decaySynapses(self, *args, **kwargs):
        """decaySynapses(self, decay, removed, permConnected, doDecay=True)"""
        return _algorithms.Segment_decaySynapses(self, *args, **kwargs)

    def freeNSynapses(self, *args, **kwargs):
        """
        freeNSynapses(self, numToFree, inactiveSynapseIndices, inactiveSegmentIndices, activeSynapseIndices, 
            activeSegmentIndices, removed, verbosity, nCellsPerCol, permMax)
        """
        return _algorithms.Segment_freeNSynapses(self, *args, **kwargs)

    def isActive(self, *args, **kwargs):
        """isActive(self, activities, permConnected, activationThreshold) -> bool"""
        return _algorithms.Segment_isActive(self, *args, **kwargs)

    def computeActivity(self, *args, **kwargs):
        """computeActivity(self, activities, permConnected, connectedSynapsesOnly) -> nupic::UInt"""
        return _algorithms.Segment_computeActivity(self, *args, **kwargs)

    def dutyCycle(self, *args, **kwargs):
        """dutyCycle(self, iteration, active, readOnly) -> nupic::Real"""
        return _algorithms.Segment_dutyCycle(self, *args, **kwargs)

    def atDutyCycleTier(*args, **kwargs):
        """atDutyCycleTier(iteration) -> bool"""
        return _algorithms.Segment_atDutyCycleTier(*args, **kwargs)

    atDutyCycleTier = staticmethod(atDutyCycleTier)
    def persistentSize(self):
        """persistentSize(self) -> nupic::UInt"""
        return _algorithms.Segment_persistentSize(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.Segment_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.Segment_load(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_Segment
Segment_swigregister = _algorithms.Segment_swigregister
Segment_swigregister(Segment)

def Segment_atDutyCycleTier(*args, **kwargs):
  """Segment_atDutyCycleTier(iteration) -> bool"""
  return _algorithms.Segment_atDutyCycleTier(*args, **kwargs)


def __lshift__(*args):
  """
    __lshift__(outStream, seg) -> std::ostream
    __lshift__(outStream, cstate) -> std::ostream
    __lshift__(outStream, cstate) -> std::ostream &
    """
  return _algorithms.__lshift__(*args)
class SegmentUpdate(object):
    """Proxy of C++ nupic::algorithms::Cells4::SegmentUpdate class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> SegmentUpdate
        __init__(self, cellIdx, segIdx, sequenceSegment, timeStamp, synapses=std::vector< nupic::UInt >(), 
            phase1Flag=False, weaklyPredicting=False, cells=None) -> SegmentUpdate
        __init__(self, o) -> SegmentUpdate
        """
        this = _algorithms.new_SegmentUpdate(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.SegmentUpdate___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.SegmentUpdate___ne__(self, *args, **kwargs)

    def isSequenceSegment(self):
        """isSequenceSegment(self) -> bool"""
        return _algorithms.SegmentUpdate_isSequenceSegment(self)

    def cellIdx(self):
        """cellIdx(self) -> nupic::UInt"""
        return _algorithms.SegmentUpdate_cellIdx(self)

    def segIdx(self):
        """segIdx(self) -> nupic::UInt"""
        return _algorithms.SegmentUpdate_segIdx(self)

    def timeStamp(self):
        """timeStamp(self) -> nupic::UInt"""
        return _algorithms.SegmentUpdate_timeStamp(self)

    def begin(self):
        """begin(self) -> nupic::algorithms::Cells4::SegmentUpdate::const_iterator"""
        return _algorithms.SegmentUpdate_begin(self)

    def end(self):
        """end(self) -> nupic::algorithms::Cells4::SegmentUpdate::const_iterator"""
        return _algorithms.SegmentUpdate_end(self)

    def size(self):
        """size(self) -> nupic::UInt"""
        return _algorithms.SegmentUpdate_size(self)

    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.SegmentUpdate_empty(self)

    def isNewSegment(self):
        """isNewSegment(self) -> bool"""
        return _algorithms.SegmentUpdate_isNewSegment(self)

    def isPhase1Segment(self):
        """isPhase1Segment(self) -> bool"""
        return _algorithms.SegmentUpdate_isPhase1Segment(self)

    def isWeaklyPredicting(self):
        """isWeaklyPredicting(self) -> bool"""
        return _algorithms.SegmentUpdate_isWeaklyPredicting(self)

    def invariants(self, cells=None):
        """invariants(self, cells=None) -> bool"""
        return _algorithms.SegmentUpdate_invariants(self, cells)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.SegmentUpdate_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.SegmentUpdate_load(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_SegmentUpdate
SegmentUpdate_swigregister = _algorithms.SegmentUpdate_swigregister
SegmentUpdate_swigregister(SegmentUpdate)

class OutSynapse(object):
    """Proxy of C++ nupic::algorithms::Cells4::OutSynapse class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self, dstCellIdx=(UInt) -1, dstSegIdx=(UInt) -1) -> OutSynapse
        __init__(self, o) -> OutSynapse
        """
        this = _algorithms.new_OutSynapse(*args)
        try: self.this.append(this)
        except: self.this = this
    def dstCellIdx(self):
        """dstCellIdx(self) -> nupic::UInt"""
        return _algorithms.OutSynapse_dstCellIdx(self)

    def dstSegIdx(self):
        """dstSegIdx(self) -> nupic::UInt"""
        return _algorithms.OutSynapse_dstSegIdx(self)

    def goesTo(self, *args, **kwargs):
        """goesTo(self, dstCellIdx, dstSegIdx) -> bool"""
        return _algorithms.OutSynapse_goesTo(self, *args, **kwargs)

    def equals(self, *args, **kwargs):
        """equals(self, o) -> bool"""
        return _algorithms.OutSynapse_equals(self, *args, **kwargs)

    def invariants(self, cells=None):
        """invariants(self, cells=None) -> bool"""
        return _algorithms.OutSynapse_invariants(self, cells)

    __swig_destroy__ = _algorithms.delete_OutSynapse
OutSynapse_swigregister = _algorithms.OutSynapse_swigregister
OutSynapse_swigregister(OutSynapse)


def __eq__(*args, **kwargs):
  """__eq__(a, b) -> bool"""
  return _algorithms.__eq__(*args, **kwargs)
class InSynapse(object):
    """Proxy of C++ nupic::algorithms::Cells4::InSynapse class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> InSynapse
        __init__(self, srcCellIdx, permanence) -> InSynapse
        __init__(self, o) -> InSynapse
        """
        this = _algorithms.new_InSynapse(*args)
        try: self.this.append(this)
        except: self.this = this
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.InSynapse___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.InSynapse___ne__(self, *args, **kwargs)

    def srcCellIdx(self):
        """srcCellIdx(self) -> nupic::UInt"""
        return _algorithms.InSynapse_srcCellIdx(self)

    def permanence(self, *args):
        """
        permanence(self) -> nupic::Real const
        permanence(self) -> nupic::Real &
        """
        return _algorithms.InSynapse_permanence(self, *args)

    __swig_destroy__ = _algorithms.delete_InSynapse
InSynapse_swigregister = _algorithms.InSynapse_swigregister
InSynapse_swigregister(InSynapse)

class Cell(object):
    """Proxy of C++ nupic::algorithms::Cells4::Cell class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self): 
        """__init__(self) -> Cell"""
        this = _algorithms.new_Cell()
        try: self.this.append(this)
        except: self.this = this
    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.Cell_empty(self)

    def nSynapses(self):
        """nSynapses(self) -> nupic::UInt"""
        return _algorithms.Cell_nSynapses(self)

    def size(self):
        """size(self) -> nupic::UInt"""
        return _algorithms.Cell_size(self)

    def nSegments(self):
        """nSegments(self) -> nupic::UInt"""
        return _algorithms.Cell_nSegments(self)

    def getNonEmptySegList(self):
        """getNonEmptySegList(self) -> VectorOfUInt32"""
        return _algorithms.Cell_getNonEmptySegList(self)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.Cell___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.Cell___ne__(self, *args, **kwargs)

    def getSegment(self, *args, **kwargs):
        """getSegment(self, segIdx) -> Segment"""
        return _algorithms.Cell_getSegment(self, *args, **kwargs)

    def getFreeSegment(self, *args, **kwargs):
        """getFreeSegment(self, synapses, initFrequency, sequenceSegmentFlag, permConnected, iteration) -> nupic::UInt"""
        return _algorithms.Cell_getFreeSegment(self, *args, **kwargs)

    def setSegmentOrder(*args, **kwargs):
        """setSegmentOrder(matchPythonOrder)"""
        return _algorithms.Cell_setSegmentOrder(*args, **kwargs)

    setSegmentOrder = staticmethod(setSegmentOrder)
    def updateDutyCycle(self, *args, **kwargs):
        """updateDutyCycle(self, iterations)"""
        return _algorithms.Cell_updateDutyCycle(self, *args, **kwargs)

    def rebalanceSegments(self):
        """rebalanceSegments(self)"""
        return _algorithms.Cell_rebalanceSegments(self)

    def getMostActiveSegment(self):
        """getMostActiveSegment(self) -> nupic::UInt"""
        return _algorithms.Cell_getMostActiveSegment(self)

    def releaseSegment(self, *args, **kwargs):
        """releaseSegment(self, segIdx)"""
        return _algorithms.Cell_releaseSegment(self, *args, **kwargs)

    def invariants(self, arg2=None):
        """invariants(self, arg2=None) -> bool"""
        return _algorithms.Cell_invariants(self, arg2)

    def persistentSize(self):
        """persistentSize(self) -> nupic::UInt"""
        return _algorithms.Cell_persistentSize(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.Cell_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.Cell_load(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_Cell
Cell_swigregister = _algorithms.Cell_swigregister
Cell_swigregister(Cell)

def Cell_setSegmentOrder(*args, **kwargs):
  """Cell_setSegmentOrder(matchPythonOrder)"""
  return _algorithms.Cell_setSegmentOrder(*args, **kwargs)

def Segment3(*args, **keywords):
   return Segment3_32(*args)

class Cells4(object):
    """Proxy of C++ nupic::algorithms::Cells4::Cells4 class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    VERSION = _algorithms.Cells4_VERSION
    def __init__(self, *args, **kwargs): 
        """
        __init__(self, nColumns=0, nCellsPerCol=0, activationThreshold=1, minThreshold=1, newSynapseCount=1, 
            segUpdateValidDuration=1, permInitial=.5, permConnected=.8, permMax=1, 
            permDec=.1, permInc=.1, globalDecay=0, doPooling=False, seed=-1, initFromCpp=True, 
            checkSynapseConsistency=False) -> Cells4
        """
        this = _algorithms.new_Cells4(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    def initialize(self, *args, **kwargs):
        """
        initialize(self, nColumns=0, nCellsPerCol=0, activationThreshold=1, minThreshold=1, newSynapseCount=1, 
            segUpdateValidDuration=1, permInitial=.5, permConnected=.8, permMax=1, 
            permDec=.1, permInc=.1, globalDecay=.1, doPooling=False, initFromCpp=True, 
            checkSynapseConsistency=False)
        """
        return _algorithms.Cells4_initialize(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_Cells4
    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.Cells4___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.Cells4___ne__(self, *args, **kwargs)

    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.Cells4_version(self)

    def getStatePointers(self, *args, **kwargs):
        """getStatePointers(self, activeT, activeT1, predT, predT1, colConfidenceT, colConfidenceT1, confidenceT, confidenceT1)"""
        return _algorithms.Cells4_getStatePointers(self, *args, **kwargs)

    def getLearnStatePointers(self, *args, **kwargs):
        """getLearnStatePointers(self, activeT, activeT1, predT, predT1)"""
        return _algorithms.Cells4_getLearnStatePointers(self, *args, **kwargs)

    def getInfActiveStateT(self):
        """getInfActiveStateT(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getInfActiveStateT(self)

    def getInfActiveStateT1(self):
        """getInfActiveStateT1(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getInfActiveStateT1(self)

    def getInfPredictedStateT(self):
        """getInfPredictedStateT(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getInfPredictedStateT(self)

    def getInfPredictedStateT1(self):
        """getInfPredictedStateT1(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getInfPredictedStateT1(self)

    def getLearnActiveStateT(self):
        """getLearnActiveStateT(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getLearnActiveStateT(self)

    def getLearnActiveStateT1(self):
        """getLearnActiveStateT1(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getLearnActiveStateT1(self)

    def getLearnPredictedStateT(self):
        """getLearnPredictedStateT(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getLearnPredictedStateT(self)

    def getLearnPredictedStateT1(self):
        """getLearnPredictedStateT1(self) -> nupic::Byte *"""
        return _algorithms.Cells4_getLearnPredictedStateT1(self)

    def getCellConfidenceT(self):
        """getCellConfidenceT(self) -> nupic::Real *"""
        return _algorithms.Cells4_getCellConfidenceT(self)

    def getCellConfidenceT1(self):
        """getCellConfidenceT1(self) -> nupic::Real *"""
        return _algorithms.Cells4_getCellConfidenceT1(self)

    def getColConfidenceT(self):
        """getColConfidenceT(self) -> nupic::Real *"""
        return _algorithms.Cells4_getColConfidenceT(self)

    def getColConfidenceT1(self):
        """getColConfidenceT1(self) -> nupic::Real *"""
        return _algorithms.Cells4_getColConfidenceT1(self)

    def nSegments(self):
        """nSegments(self) -> nupic::UInt"""
        return _algorithms.Cells4_nSegments(self)

    def nCells(self):
        """nCells(self) -> nupic::UInt"""
        return _algorithms.Cells4_nCells(self)

    def nColumns(self):
        """nColumns(self) -> nupic::UInt"""
        return _algorithms.Cells4_nColumns(self)

    def nCellsPerCol(self):
        """nCellsPerCol(self) -> nupic::UInt"""
        return _algorithms.Cells4_nCellsPerCol(self)

    def getPermInitial(self):
        """getPermInitial(self) -> nupic::Real"""
        return _algorithms.Cells4_getPermInitial(self)

    def getMinThreshold(self):
        """getMinThreshold(self) -> nupic::UInt"""
        return _algorithms.Cells4_getMinThreshold(self)

    def getPermConnected(self):
        """getPermConnected(self) -> nupic::Real"""
        return _algorithms.Cells4_getPermConnected(self)

    def getNewSynapseCount(self):
        """getNewSynapseCount(self) -> nupic::UInt"""
        return _algorithms.Cells4_getNewSynapseCount(self)

    def getPermInc(self):
        """getPermInc(self) -> nupic::Real"""
        return _algorithms.Cells4_getPermInc(self)

    def getPermDec(self):
        """getPermDec(self) -> nupic::Real"""
        return _algorithms.Cells4_getPermDec(self)

    def getPermMax(self):
        """getPermMax(self) -> nupic::Real"""
        return _algorithms.Cells4_getPermMax(self)

    def getGlobalDecay(self):
        """getGlobalDecay(self) -> nupic::Real"""
        return _algorithms.Cells4_getGlobalDecay(self)

    def getActivationThreshold(self):
        """getActivationThreshold(self) -> nupic::UInt"""
        return _algorithms.Cells4_getActivationThreshold(self)

    def getDoPooling(self):
        """getDoPooling(self) -> bool"""
        return _algorithms.Cells4_getDoPooling(self)

    def getSegUpdateValidDuration(self):
        """getSegUpdateValidDuration(self) -> nupic::UInt"""
        return _algorithms.Cells4_getSegUpdateValidDuration(self)

    def getVerbosity(self):
        """getVerbosity(self) -> nupic::UInt"""
        return _algorithms.Cells4_getVerbosity(self)

    def getMaxAge(self):
        """getMaxAge(self) -> nupic::UInt"""
        return _algorithms.Cells4_getMaxAge(self)

    def getPamLength(self):
        """getPamLength(self) -> nupic::UInt"""
        return _algorithms.Cells4_getPamLength(self)

    def getMaxInfBacktrack(self):
        """getMaxInfBacktrack(self) -> nupic::UInt"""
        return _algorithms.Cells4_getMaxInfBacktrack(self)

    def getMaxLrnBacktrack(self):
        """getMaxLrnBacktrack(self) -> nupic::UInt"""
        return _algorithms.Cells4_getMaxLrnBacktrack(self)

    def getPamCounter(self):
        """getPamCounter(self) -> nupic::UInt"""
        return _algorithms.Cells4_getPamCounter(self)

    def getMaxSeqLength(self):
        """getMaxSeqLength(self) -> nupic::UInt"""
        return _algorithms.Cells4_getMaxSeqLength(self)

    def getAvgLearnedSeqLength(self):
        """getAvgLearnedSeqLength(self) -> nupic::Real"""
        return _algorithms.Cells4_getAvgLearnedSeqLength(self)

    def getNLrnIterations(self):
        """getNLrnIterations(self) -> nupic::UInt"""
        return _algorithms.Cells4_getNLrnIterations(self)

    def getMaxSegmentsPerCell(self):
        """getMaxSegmentsPerCell(self) -> nupic::Int"""
        return _algorithms.Cells4_getMaxSegmentsPerCell(self)

    def getMaxSynapsesPerSegment(self):
        """getMaxSynapsesPerSegment(self) -> nupic::Int"""
        return _algorithms.Cells4_getMaxSynapsesPerSegment(self)

    def getCheckSynapseConsistency(self):
        """getCheckSynapseConsistency(self) -> bool"""
        return _algorithms.Cells4_getCheckSynapseConsistency(self)

    def setMaxInfBacktrack(self, *args, **kwargs):
        """setMaxInfBacktrack(self, t)"""
        return _algorithms.Cells4_setMaxInfBacktrack(self, *args, **kwargs)

    def setMaxLrnBacktrack(self, *args, **kwargs):
        """setMaxLrnBacktrack(self, t)"""
        return _algorithms.Cells4_setMaxLrnBacktrack(self, *args, **kwargs)

    def setVerbosity(self, *args, **kwargs):
        """setVerbosity(self, v)"""
        return _algorithms.Cells4_setVerbosity(self, *args, **kwargs)

    def setMaxAge(self, *args, **kwargs):
        """setMaxAge(self, a)"""
        return _algorithms.Cells4_setMaxAge(self, *args, **kwargs)

    def setMaxSeqLength(self, *args, **kwargs):
        """setMaxSeqLength(self, v)"""
        return _algorithms.Cells4_setMaxSeqLength(self, *args, **kwargs)

    def setCheckSynapseConsistency(self, *args, **kwargs):
        """setCheckSynapseConsistency(self, val)"""
        return _algorithms.Cells4_setCheckSynapseConsistency(self, *args, **kwargs)

    def setMaxSegmentsPerCell(self, *args, **kwargs):
        """setMaxSegmentsPerCell(self, maxSegs)"""
        return _algorithms.Cells4_setMaxSegmentsPerCell(self, *args, **kwargs)

    def setMaxSynapsesPerCell(self, *args, **kwargs):
        """setMaxSynapsesPerCell(self, maxSyns)"""
        return _algorithms.Cells4_setMaxSynapsesPerCell(self, *args, **kwargs)

    def setPamLength(self, *args, **kwargs):
        """setPamLength(self, pl)"""
        return _algorithms.Cells4_setPamLength(self, *args, **kwargs)

    def nSegmentsOnCell(self, *args, **kwargs):
        """nSegmentsOnCell(self, colIdx, cellIdxInCol) -> nupic::UInt"""
        return _algorithms.Cells4_nSegmentsOnCell(self, *args, **kwargs)

    def nSynapses(self):
        """nSynapses(self) -> nupic::UInt"""
        return _algorithms.Cells4_nSynapses(self)

    def __nSegmentsOnCell(self, *args, **kwargs):
        """__nSegmentsOnCell(self, cellIdx) -> nupic::UInt"""
        return _algorithms.Cells4___nSegmentsOnCell(self, *args, **kwargs)

    def nSynapsesInCell(self, *args, **kwargs):
        """nSynapsesInCell(self, cellIdx) -> nupic::UInt"""
        return _algorithms.Cells4_nSynapsesInCell(self, *args, **kwargs)

    def getCell(self, *args, **kwargs):
        """getCell(self, colIdx, cellIdxInCol) -> Cell"""
        return _algorithms.Cells4_getCell(self, *args, **kwargs)

    def getCellIdx(self, *args, **kwargs):
        """getCellIdx(self, colIdx, cellIdxInCol) -> nupic::UInt"""
        return _algorithms.Cells4_getCellIdx(self, *args, **kwargs)

    def getSegment(self, *args, **kwargs):
        """getSegment(self, colIdx, cellIdxInCol, segIdx) -> Segment"""
        return _algorithms.Cells4_getSegment(self, *args, **kwargs)

    def segment(self, *args, **kwargs):
        """segment(self, cellIdx, segIdx) -> Segment"""
        return _algorithms.Cells4_segment(self, *args, **kwargs)

    def reset(self):
        """reset(self)"""
        return _algorithms.Cells4_reset(self)

    def isActive(self, *args, **kwargs):
        """isActive(self, cellIdx, segIdx, state) -> bool"""
        return _algorithms.Cells4_isActive(self, *args, **kwargs)

    def getBestMatchingCellT(self, *args, **kwargs):
        """getBestMatchingCellT(self, colIdx, state, minThreshold) -> PairOfUInt32"""
        return _algorithms.Cells4_getBestMatchingCellT(self, *args, **kwargs)

    def getBestMatchingCellT1(self, *args, **kwargs):
        """getBestMatchingCellT1(self, colIdx, state, minThreshold) -> PairOfUInt32"""
        return _algorithms.Cells4_getBestMatchingCellT1(self, *args, **kwargs)

    def computeForwardPropagation(self, *args):
        """
        computeForwardPropagation(self, state)
        computeForwardPropagation(self, state)
        """
        return _algorithms.Cells4_computeForwardPropagation(self, *args)

    def updateInferenceState(self, *args, **kwargs):
        """updateInferenceState(self, activeColumns)"""
        return _algorithms.Cells4_updateInferenceState(self, *args, **kwargs)

    def inferPhase1(self, *args, **kwargs):
        """inferPhase1(self, activeColumns, useStartCells) -> bool"""
        return _algorithms.Cells4_inferPhase1(self, *args, **kwargs)

    def inferPhase2(self):
        """inferPhase2(self) -> bool"""
        return _algorithms.Cells4_inferPhase2(self)

    def inferBacktrack(self, *args, **kwargs):
        """inferBacktrack(self, activeColumns)"""
        return _algorithms.Cells4_inferBacktrack(self, *args, **kwargs)

    def updateLearningState(self, *args, **kwargs):
        """updateLearningState(self, activeColumns, input)"""
        return _algorithms.Cells4_updateLearningState(self, *args, **kwargs)

    def learnPhase1(self, *args, **kwargs):
        """learnPhase1(self, activeColumns, readOnly) -> bool"""
        return _algorithms.Cells4_learnPhase1(self, *args, **kwargs)

    def learnPhase2(self, *args, **kwargs):
        """learnPhase2(self, readOnly)"""
        return _algorithms.Cells4_learnPhase2(self, *args, **kwargs)

    def learnBacktrack(self):
        """learnBacktrack(self) -> nupic::UInt"""
        return _algorithms.Cells4_learnBacktrack(self)

    def learnBacktrackFrom(self, *args, **kwargs):
        """learnBacktrackFrom(self, startOffset, readOnly) -> bool"""
        return _algorithms.Cells4_learnBacktrackFrom(self, *args, **kwargs)

    def _updateAvgLearnedSeqLength(self, *args, **kwargs):
        """_updateAvgLearnedSeqLength(self, prevSeqLength)"""
        return _algorithms.Cells4__updateAvgLearnedSeqLength(self, *args, **kwargs)

    def chooseCellsToLearnFrom(self, *args, **kwargs):
        """chooseCellsToLearnFrom(self, cellIdx, segIdx, nSynToAdd, state, srcCells)"""
        return _algorithms.Cells4_chooseCellsToLearnFrom(self, *args, **kwargs)

    def getCellForNewSegment(self, *args, **kwargs):
        """getCellForNewSegment(self, colIdx) -> nupic::UInt"""
        return _algorithms.Cells4_getCellForNewSegment(self, *args, **kwargs)

    def computeUpdate(self, *args, **kwargs):
        """computeUpdate(self, cellIdx, segIdx, activeState, sequenceSegmentFlag, newSynapsesFlag) -> bool"""
        return _algorithms.Cells4_computeUpdate(self, *args, **kwargs)

    def eraseOutSynapses(self, *args, **kwargs):
        """eraseOutSynapses(self, dstCellIdx, dstSegIdx, srcCells)"""
        return _algorithms.Cells4_eraseOutSynapses(self, *args, **kwargs)

    def processSegmentUpdates(self, *args, **kwargs):
        """processSegmentUpdates(self, input, predictedState)"""
        return _algorithms.Cells4_processSegmentUpdates(self, *args, **kwargs)

    def cleanUpdatesList(self, *args, **kwargs):
        """cleanUpdatesList(self, cellIdx, segIdx)"""
        return _algorithms.Cells4_cleanUpdatesList(self, *args, **kwargs)

    def applyGlobalDecay(self):
        """applyGlobalDecay(self)"""
        return _algorithms.Cells4_applyGlobalDecay(self)

    def _generateListsOfSynapsesToAdjustForAdaptSegment(*args, **kwargs):
        """
        _generateListsOfSynapsesToAdjustForAdaptSegment(segment, synapsesSet, inactiveSrcCellIdxs, inactiveSynapseIdxs, activeSrcCellIdxs, 
            activeSynapseIdxs)
        """
        return _algorithms.Cells4__generateListsOfSynapsesToAdjustForAdaptSegment(*args, **kwargs)

    _generateListsOfSynapsesToAdjustForAdaptSegment = staticmethod(_generateListsOfSynapsesToAdjustForAdaptSegment)
    def adaptSegment(self, *args, **kwargs):
        """adaptSegment(self, update)"""
        return _algorithms.Cells4_adaptSegment(self, *args, **kwargs)

    def trimSegments(self, *args, **kwargs):
        """trimSegments(self, minPermanence, minNumSyns) -> PairOfUInt32"""
        return _algorithms.Cells4_trimSegments(self, *args, **kwargs)

    def persistentSize(self):
        """persistentSize(self) -> nupic::UInt"""
        return _algorithms.Cells4_persistentSize(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.Cells4_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.Cells4_load(self, *args, **kwargs)

    def __lshift__(self, *args, **kwargs):
        """__lshift__(self, outStream) -> std::ostream &"""
        return _algorithms.Cells4___lshift__(self, *args, **kwargs)

    def setCellSegmentOrder(self, *args, **kwargs):
        """setCellSegmentOrder(self, matchPythonOrder)"""
        return _algorithms.Cells4_setCellSegmentOrder(self, *args, **kwargs)

    def addNewSegment(self, *args, **kwargs):
        """addNewSegment(self, colIdx, cellIdxInCol, sequenceSegmentFlag, extSynapses)"""
        return _algorithms.Cells4_addNewSegment(self, *args, **kwargs)

    def updateSegment(self, *args, **kwargs):
        """updateSegment(self, colIdx, cellIdxInCol, segIdx, extSynapses)"""
        return _algorithms.Cells4_updateSegment(self, *args, **kwargs)

    def _rebalance(self):
        """_rebalance(self)"""
        return _algorithms.Cells4__rebalance(self)

    def rebuildOutSynapses(self):
        """rebuildOutSynapses(self)"""
        return _algorithms.Cells4_rebuildOutSynapses(self)

    def trimOldSegments(self, *args, **kwargs):
        """trimOldSegments(self, age)"""
        return _algorithms.Cells4_trimOldSegments(self, *args, **kwargs)

    def printStates(self):
        """printStates(self)"""
        return _algorithms.Cells4_printStates(self)

    def printState(self, *args, **kwargs):
        """printState(self, state)"""
        return _algorithms.Cells4_printState(self, *args, **kwargs)

    def printConfidence(self, *args, **kwargs):
        """printConfidence(self, confidence, len)"""
        return _algorithms.Cells4_printConfidence(self, *args, **kwargs)

    def dumpPrevPatterns(self, *args, **kwargs):
        """dumpPrevPatterns(self, patterns)"""
        return _algorithms.Cells4_dumpPrevPatterns(self, *args, **kwargs)

    def dumpSegmentUpdates(self):
        """dumpSegmentUpdates(self)"""
        return _algorithms.Cells4_dumpSegmentUpdates(self)

    def getNonEmptySegList(self, *args, **kwargs):
        """getNonEmptySegList(self, colIdx, cellIdxInCol) -> VectorOfUInt32"""
        return _algorithms.Cells4_getNonEmptySegList(self, *args, **kwargs)

    def dumpTiming(self):
        """dumpTiming(self)"""
        return _algorithms.Cells4_dumpTiming(self)

    def resetTimers(self):
        """resetTimers(self)"""
        return _algorithms.Cells4_resetTimers(self)

    def invariants(self, verbose=False):
        """invariants(self, verbose=False) -> bool"""
        return _algorithms.Cells4_invariants(self, verbose)

    def stats(self):
        """stats(self)"""
        return _algorithms.Cells4_stats(self)

    def __init__(self, *args, **kwargs):
      self.this = _ALGORITHMS.new_Cells4(*args, **kwargs)

    def __setstate__(self, inString):
      self.this = _ALGORITHMS.new_Cells4()
      self.loadFromString(inString)


    def loadFromString(self, *args, **kwargs):
        """loadFromString(self, inString)"""
        return _algorithms.Cells4_loadFromString(self, *args, **kwargs)

    def __getstate__(self):
        """__getstate__(self) -> PyObject *"""
        return _algorithms.Cells4___getstate__(self)

    def setStatePointers(self, *args):
        """
        setStatePointers(self, infActiveT, infActiveT1, infPredT, infPredT1, colConfidenceT, colConfidenceT1, cellConfidenceT, 
            cellConfidenceT1)
        setStatePointers(self, py_infActiveStateT, py_infActiveStateT1, py_infPredictedStateT, py_infPredictedStateT1, 
            py_colConfidenceT, py_colConfidenceT1, py_cellConfidenceT, py_cellConfidenceT1)
        """
        return _algorithms.Cells4_setStatePointers(self, *args)

    def getStates(self):
        """getStates(self) -> PyObject *"""
        return _algorithms.Cells4_getStates(self)

    def getLearnStates(self):
        """getLearnStates(self) -> PyObject *"""
        return _algorithms.Cells4_getLearnStates(self)

    def compute(self, *args):
        """
        compute(self, input, output, doInference, doLearning)
        compute(self, py_x, doInference, doLearning) -> PyObject *
        """
        return _algorithms.Cells4_compute(self, *args)

Cells4_swigregister = _algorithms.Cells4_swigregister
Cells4_swigregister(Cells4)
_MAX_CELLS = cvar._MAX_CELLS
_MAX_SEGS = cvar._MAX_SEGS

def Cells4__generateListsOfSynapsesToAdjustForAdaptSegment(*args, **kwargs):
  """
    Cells4__generateListsOfSynapsesToAdjustForAdaptSegment(segment, synapsesSet, inactiveSrcCellIdxs, inactiveSynapseIdxs, activeSrcCellIdxs, 
        activeSynapseIdxs)
    """
  return _algorithms.Cells4__generateListsOfSynapsesToAdjustForAdaptSegment(*args, **kwargs)

class SpatialPooler(object):
    """Proxy of C++ nupic::algorithms::spatial_pooler::SpatialPooler class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> SpatialPooler
        __init__(self, inputDimensions, columnDimensions, potentialRadius=16, potentialPct=0.5, globalInhibition=True, 
            localAreaDensity=-1.0, numActiveColumnsPerInhArea=10, stimulusThreshold=0, 
            synPermInactiveDec=0.008, synPermActiveInc=0.05, synPermConnected=0.1, 
            minPctOverlapDutyCycles=0.001, dutyCyclePeriod=1000, boostStrength=0.0, 
            seed=1, spVerbosity=0, wrapAround=True) -> SpatialPooler
        """
        this = _algorithms.new_SpatialPooler(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_SpatialPooler
    def initialize(self, *args, **kwargs):
        """
        initialize(self, inputDimensions, columnDimensions, potentialRadius=16, potentialPct=0.5, globalInhibition=True, 
            localAreaDensity=-1.0, numActiveColumnsPerInhArea=10, stimulusThreshold=0, 
            synPermInactiveDec=0.01, synPermActiveInc=0.1, synPermConnected=0.1, 
            minPctOverlapDutyCycles=0.001, dutyCyclePeriod=1000, boostStrength=0.0, 
            seed=1, spVerbosity=0, wrapAround=True)
        """
        return _algorithms.SpatialPooler_initialize(self, *args, **kwargs)

    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_version(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.SpatialPooler_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.SpatialPooler_load(self, *args, **kwargs)

    def persistentSize(self):
        """persistentSize(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_persistentSize(self)

    def getColumnDimensions(self):
        """getColumnDimensions(self) -> VectorOfUInt32"""
        return _algorithms.SpatialPooler_getColumnDimensions(self)

    def getInputDimensions(self):
        """getInputDimensions(self) -> VectorOfUInt32"""
        return _algorithms.SpatialPooler_getInputDimensions(self)

    def getNumColumns(self):
        """getNumColumns(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getNumColumns(self)

    def getNumInputs(self):
        """getNumInputs(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getNumInputs(self)

    def getPotentialRadius(self):
        """getPotentialRadius(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getPotentialRadius(self)

    def setPotentialRadius(self, *args, **kwargs):
        """setPotentialRadius(self, potentialRadius)"""
        return _algorithms.SpatialPooler_setPotentialRadius(self, *args, **kwargs)

    def getPotentialPct(self):
        """getPotentialPct(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getPotentialPct(self)

    def setPotentialPct(self, *args, **kwargs):
        """setPotentialPct(self, potentialPct)"""
        return _algorithms.SpatialPooler_setPotentialPct(self, *args, **kwargs)

    def getGlobalInhibition(self):
        """getGlobalInhibition(self) -> bool"""
        return _algorithms.SpatialPooler_getGlobalInhibition(self)

    def setGlobalInhibition(self, *args, **kwargs):
        """setGlobalInhibition(self, globalInhibition)"""
        return _algorithms.SpatialPooler_setGlobalInhibition(self, *args, **kwargs)

    def getNumActiveColumnsPerInhArea(self):
        """getNumActiveColumnsPerInhArea(self) -> nupic::Int"""
        return _algorithms.SpatialPooler_getNumActiveColumnsPerInhArea(self)

    def setNumActiveColumnsPerInhArea(self, *args, **kwargs):
        """setNumActiveColumnsPerInhArea(self, numActiveColumnsPerInhArea)"""
        return _algorithms.SpatialPooler_setNumActiveColumnsPerInhArea(self, *args, **kwargs)

    def getLocalAreaDensity(self):
        """getLocalAreaDensity(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getLocalAreaDensity(self)

    def setLocalAreaDensity(self, *args, **kwargs):
        """setLocalAreaDensity(self, localAreaDensity)"""
        return _algorithms.SpatialPooler_setLocalAreaDensity(self, *args, **kwargs)

    def getStimulusThreshold(self):
        """getStimulusThreshold(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getStimulusThreshold(self)

    def setStimulusThreshold(self, *args, **kwargs):
        """setStimulusThreshold(self, stimulusThreshold)"""
        return _algorithms.SpatialPooler_setStimulusThreshold(self, *args, **kwargs)

    def getInhibitionRadius(self):
        """getInhibitionRadius(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getInhibitionRadius(self)

    def setInhibitionRadius(self, *args, **kwargs):
        """setInhibitionRadius(self, inhibitionRadius)"""
        return _algorithms.SpatialPooler_setInhibitionRadius(self, *args, **kwargs)

    def getDutyCyclePeriod(self):
        """getDutyCyclePeriod(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getDutyCyclePeriod(self)

    def setDutyCyclePeriod(self, *args, **kwargs):
        """setDutyCyclePeriod(self, dutyCyclePeriod)"""
        return _algorithms.SpatialPooler_setDutyCyclePeriod(self, *args, **kwargs)

    def getBoostStrength(self):
        """getBoostStrength(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getBoostStrength(self)

    def setBoostStrength(self, *args, **kwargs):
        """setBoostStrength(self, boostStrength)"""
        return _algorithms.SpatialPooler_setBoostStrength(self, *args, **kwargs)

    def getIterationNum(self):
        """getIterationNum(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getIterationNum(self)

    def setIterationNum(self, *args, **kwargs):
        """setIterationNum(self, iterationNum)"""
        return _algorithms.SpatialPooler_setIterationNum(self, *args, **kwargs)

    def getIterationLearnNum(self):
        """getIterationLearnNum(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getIterationLearnNum(self)

    def setIterationLearnNum(self, *args, **kwargs):
        """setIterationLearnNum(self, iterationLearnNum)"""
        return _algorithms.SpatialPooler_setIterationLearnNum(self, *args, **kwargs)

    def getSpVerbosity(self):
        """getSpVerbosity(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getSpVerbosity(self)

    def setSpVerbosity(self, *args, **kwargs):
        """setSpVerbosity(self, spVerbosity)"""
        return _algorithms.SpatialPooler_setSpVerbosity(self, *args, **kwargs)

    def getWrapAround(self):
        """getWrapAround(self) -> bool"""
        return _algorithms.SpatialPooler_getWrapAround(self)

    def setWrapAround(self, *args, **kwargs):
        """setWrapAround(self, wrapAround)"""
        return _algorithms.SpatialPooler_setWrapAround(self, *args, **kwargs)

    def getUpdatePeriod(self):
        """getUpdatePeriod(self) -> nupic::UInt"""
        return _algorithms.SpatialPooler_getUpdatePeriod(self)

    def setUpdatePeriod(self, *args, **kwargs):
        """setUpdatePeriod(self, updatePeriod)"""
        return _algorithms.SpatialPooler_setUpdatePeriod(self, *args, **kwargs)

    def getSynPermTrimThreshold(self):
        """getSynPermTrimThreshold(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermTrimThreshold(self)

    def setSynPermTrimThreshold(self, *args, **kwargs):
        """setSynPermTrimThreshold(self, synPermTrimThreshold)"""
        return _algorithms.SpatialPooler_setSynPermTrimThreshold(self, *args, **kwargs)

    def getSynPermActiveInc(self):
        """getSynPermActiveInc(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermActiveInc(self)

    def setSynPermActiveInc(self, *args, **kwargs):
        """setSynPermActiveInc(self, synPermActiveInc)"""
        return _algorithms.SpatialPooler_setSynPermActiveInc(self, *args, **kwargs)

    def getSynPermInactiveDec(self):
        """getSynPermInactiveDec(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermInactiveDec(self)

    def setSynPermInactiveDec(self, *args, **kwargs):
        """setSynPermInactiveDec(self, synPermInactiveDec)"""
        return _algorithms.SpatialPooler_setSynPermInactiveDec(self, *args, **kwargs)

    def getSynPermBelowStimulusInc(self):
        """getSynPermBelowStimulusInc(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermBelowStimulusInc(self)

    def setSynPermBelowStimulusInc(self, *args, **kwargs):
        """setSynPermBelowStimulusInc(self, synPermBelowStimulusInc)"""
        return _algorithms.SpatialPooler_setSynPermBelowStimulusInc(self, *args, **kwargs)

    def getSynPermConnected(self):
        """getSynPermConnected(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermConnected(self)

    def setSynPermConnected(self, *args, **kwargs):
        """setSynPermConnected(self, synPermConnected)"""
        return _algorithms.SpatialPooler_setSynPermConnected(self, *args, **kwargs)

    def getSynPermMax(self):
        """getSynPermMax(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getSynPermMax(self)

    def setSynPermMax(self, *args, **kwargs):
        """setSynPermMax(self, synPermMax)"""
        return _algorithms.SpatialPooler_setSynPermMax(self, *args, **kwargs)

    def getMinPctOverlapDutyCycles(self):
        """getMinPctOverlapDutyCycles(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_getMinPctOverlapDutyCycles(self)

    def setMinPctOverlapDutyCycles(self, *args, **kwargs):
        """setMinPctOverlapDutyCycles(self, minPctOverlapDutyCycles)"""
        return _algorithms.SpatialPooler_setMinPctOverlapDutyCycles(self, *args, **kwargs)

    def printParameters(self):
        """printParameters(self)"""
        return _algorithms.SpatialPooler_printParameters(self)

    def getOverlapsTuple(self):
        """getOverlapsTuple(self) -> VectorOfUInt32"""
        return _algorithms.SpatialPooler_getOverlapsTuple(self)

    def getBoostedOverlapsTuple(self):
        """getBoostedOverlapsTuple(self) -> FloatVector"""
        return _algorithms.SpatialPooler_getBoostedOverlapsTuple(self)

    def toDense_(self, *args, **kwargs):
        """toDense_(self, sparse, dense, n)"""
        return _algorithms.SpatialPooler_toDense_(self, *args, **kwargs)

    def boostOverlaps_(self, *args, **kwargs):
        """boostOverlaps_(self, overlaps, boostedOverlaps)"""
        return _algorithms.SpatialPooler_boostOverlaps_(self, *args, **kwargs)

    def mapColumn_(self, *args, **kwargs):
        """mapColumn_(self, column) -> nupic::UInt"""
        return _algorithms.SpatialPooler_mapColumn_(self, *args, **kwargs)

    def mapPotential_(self, *args, **kwargs):
        """mapPotential_(self, column, wrapAround) -> VectorOfUInt32"""
        return _algorithms.SpatialPooler_mapPotential_(self, *args, **kwargs)

    def initPermConnected_(self):
        """initPermConnected_(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_initPermConnected_(self)

    def initPermNonConnected_(self):
        """initPermNonConnected_(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_initPermNonConnected_(self)

    def initPermanence_(self, *args, **kwargs):
        """initPermanence_(self, potential, connectedPct) -> FloatVector"""
        return _algorithms.SpatialPooler_initPermanence_(self, *args, **kwargs)

    def clip_(self, *args, **kwargs):
        """clip_(self, perm, trim)"""
        return _algorithms.SpatialPooler_clip_(self, *args, **kwargs)

    def countConnected_(self, *args, **kwargs):
        """countConnected_(self, perm) -> nupic::UInt"""
        return _algorithms.SpatialPooler_countConnected_(self, *args, **kwargs)

    def raisePermanencesToThreshold_(self, *args, **kwargs):
        """raisePermanencesToThreshold_(self, perm, potential) -> nupic::UInt"""
        return _algorithms.SpatialPooler_raisePermanencesToThreshold_(self, *args, **kwargs)

    def calculateOverlapPct_(self, *args, **kwargs):
        """calculateOverlapPct_(self, overlaps, overlapPct)"""
        return _algorithms.SpatialPooler_calculateOverlapPct_(self, *args, **kwargs)

    def isWinner_(self, *args, **kwargs):
        """isWinner_(self, score, winners, numWinners) -> bool"""
        return _algorithms.SpatialPooler_isWinner_(self, *args, **kwargs)

    def addToWinners_(self, *args, **kwargs):
        """addToWinners_(self, index, score, winners)"""
        return _algorithms.SpatialPooler_addToWinners_(self, *args, **kwargs)

    def inhibitColumnsGlobal_(self, *args, **kwargs):
        """inhibitColumnsGlobal_(self, overlaps, density, activeColumns)"""
        return _algorithms.SpatialPooler_inhibitColumnsGlobal_(self, *args, **kwargs)

    def inhibitColumnsLocal_(self, *args, **kwargs):
        """inhibitColumnsLocal_(self, overlaps, density, activeColumns)"""
        return _algorithms.SpatialPooler_inhibitColumnsLocal_(self, *args, **kwargs)

    def adaptSynapses_(self, *args, **kwargs):
        """adaptSynapses_(self, inputVector, activeColumns)"""
        return _algorithms.SpatialPooler_adaptSynapses_(self, *args, **kwargs)

    def bumpUpWeakColumns_(self):
        """bumpUpWeakColumns_(self)"""
        return _algorithms.SpatialPooler_bumpUpWeakColumns_(self)

    def updateInhibitionRadius_(self):
        """updateInhibitionRadius_(self)"""
        return _algorithms.SpatialPooler_updateInhibitionRadius_(self)

    def avgColumnsPerInput_(self):
        """avgColumnsPerInput_(self) -> nupic::Real"""
        return _algorithms.SpatialPooler_avgColumnsPerInput_(self)

    def avgConnectedSpanForColumn1D_(self, *args, **kwargs):
        """avgConnectedSpanForColumn1D_(self, column) -> nupic::Real"""
        return _algorithms.SpatialPooler_avgConnectedSpanForColumn1D_(self, *args, **kwargs)

    def avgConnectedSpanForColumn2D_(self, *args, **kwargs):
        """avgConnectedSpanForColumn2D_(self, column) -> nupic::Real"""
        return _algorithms.SpatialPooler_avgConnectedSpanForColumn2D_(self, *args, **kwargs)

    def avgConnectedSpanForColumnND_(self, *args, **kwargs):
        """avgConnectedSpanForColumnND_(self, column) -> nupic::Real"""
        return _algorithms.SpatialPooler_avgConnectedSpanForColumnND_(self, *args, **kwargs)

    def updateMinDutyCycles_(self):
        """updateMinDutyCycles_(self)"""
        return _algorithms.SpatialPooler_updateMinDutyCycles_(self)

    def updateMinDutyCyclesGlobal_(self):
        """updateMinDutyCyclesGlobal_(self)"""
        return _algorithms.SpatialPooler_updateMinDutyCyclesGlobal_(self)

    def updateMinDutyCyclesLocal_(self):
        """updateMinDutyCyclesLocal_(self)"""
        return _algorithms.SpatialPooler_updateMinDutyCyclesLocal_(self)

    def updateDutyCyclesHelper_(*args, **kwargs):
        """updateDutyCyclesHelper_(dutyCycles, newValues, period)"""
        return _algorithms.SpatialPooler_updateDutyCyclesHelper_(*args, **kwargs)

    updateDutyCyclesHelper_ = staticmethod(updateDutyCyclesHelper_)
    def updateBoostFactors_(self):
        """updateBoostFactors_(self)"""
        return _algorithms.SpatialPooler_updateBoostFactors_(self)

    def updateBoostFactorsLocal_(self):
        """updateBoostFactorsLocal_(self)"""
        return _algorithms.SpatialPooler_updateBoostFactorsLocal_(self)

    def updateBoostFactorsGlobal_(self):
        """updateBoostFactorsGlobal_(self)"""
        return _algorithms.SpatialPooler_updateBoostFactorsGlobal_(self)

    def updateBookeepingVars_(self, *args, **kwargs):
        """updateBookeepingVars_(self, learn)"""
        return _algorithms.SpatialPooler_updateBookeepingVars_(self, *args, **kwargs)

    def isUpdateRound_(self):
        """isUpdateRound_(self) -> bool"""
        return _algorithms.SpatialPooler_isUpdateRound_(self)

    def seed_(self, *args, **kwargs):
        """seed_(self, seed)"""
        return _algorithms.SpatialPooler_seed_(self, *args, **kwargs)

    def printState(self, *args):
        """
        printState(self, state)
        printState(self, state)
        """
        return _algorithms.SpatialPooler_printState(self, *args)

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


    def compute(self, *args):
        """
        compute(self, inputVector, learn, activeVector)
        compute(self, py_inputArray, learn, py_activeArray)
        """
        return _algorithms.SpatialPooler_compute(self, *args)

    def stripUnlearnedColumns(self, *args):
        """
        stripUnlearnedColumns(self, activeArray)
        stripUnlearnedColumns(self, py_x)
        """
        return _algorithms.SpatialPooler_stripUnlearnedColumns(self, *args)

    def loadFromString(self, *args, **kwargs):
        """loadFromString(self, inString)"""
        return _algorithms.SpatialPooler_loadFromString(self, *args, **kwargs)

    def getCState(self):
        """getCState(self) -> PyObject *"""
        return _algorithms.SpatialPooler_getCState(self)

    def setBoostFactors(self, *args):
        """
        setBoostFactors(self, boostFactors)
        setBoostFactors(self, py_x)
        """
        return _algorithms.SpatialPooler_setBoostFactors(self, *args)

    def getBoostFactors(self, *args):
        """
        getBoostFactors(self, boostFactors)
        getBoostFactors(self, py_x)
        """
        return _algorithms.SpatialPooler_getBoostFactors(self, *args)

    def setOverlapDutyCycles(self, *args):
        """
        setOverlapDutyCycles(self, overlapDutyCycles)
        setOverlapDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_setOverlapDutyCycles(self, *args)

    def getOverlapDutyCycles(self, *args):
        """
        getOverlapDutyCycles(self, overlapDutyCycles)
        getOverlapDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_getOverlapDutyCycles(self, *args)

    def setActiveDutyCycles(self, *args):
        """
        setActiveDutyCycles(self, activeDutyCycles)
        setActiveDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_setActiveDutyCycles(self, *args)

    def getActiveDutyCycles(self, *args):
        """
        getActiveDutyCycles(self, activeDutyCycles)
        getActiveDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_getActiveDutyCycles(self, *args)

    def setMinOverlapDutyCycles(self, *args):
        """
        setMinOverlapDutyCycles(self, minOverlapDutyCycles)
        setMinOverlapDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_setMinOverlapDutyCycles(self, *args)

    def getMinOverlapDutyCycles(self, *args):
        """
        getMinOverlapDutyCycles(self, minOverlapDutyCycles)
        getMinOverlapDutyCycles(self, py_x)
        """
        return _algorithms.SpatialPooler_getMinOverlapDutyCycles(self, *args)

    def setPotential(self, *args):
        """
        setPotential(self, column, potential)
        setPotential(self, column, py_x)
        """
        return _algorithms.SpatialPooler_setPotential(self, *args)

    def getPotential(self, *args):
        """
        getPotential(self, column, potential)
        getPotential(self, column, py_x)
        """
        return _algorithms.SpatialPooler_getPotential(self, *args)

    def setPermanence(self, *args):
        """
        setPermanence(self, column, permanence)
        setPermanence(self, column, py_x)
        """
        return _algorithms.SpatialPooler_setPermanence(self, *args)

    def getPermanence(self, *args):
        """
        getPermanence(self, column, permanence)
        getPermanence(self, column, py_x)
        """
        return _algorithms.SpatialPooler_getPermanence(self, *args)

    def getConnectedSynapses(self, *args):
        """
        getConnectedSynapses(self, column, connectedSynapses)
        getConnectedSynapses(self, column, py_x)
        """
        return _algorithms.SpatialPooler_getConnectedSynapses(self, *args)

    def getConnectedCounts(self, *args):
        """
        getConnectedCounts(self, connectedCounts)
        getConnectedCounts(self, py_x)
        """
        return _algorithms.SpatialPooler_getConnectedCounts(self, *args)

    def getOverlaps(self):
        """getOverlaps(self) -> PyObject *"""
        return _algorithms.SpatialPooler_getOverlaps(self)

    def getBoostedOverlaps(self):
        """getBoostedOverlaps(self) -> PyObject *"""
        return _algorithms.SpatialPooler_getBoostedOverlaps(self)

    def calculateOverlap_(self, *args):
        """
        calculateOverlap_(self, inputVector, overlap)
        calculateOverlap_(self, py_inputVector) -> PyObject *
        """
        return _algorithms.SpatialPooler_calculateOverlap_(self, *args)

    def inhibitColumns_(self, *args):
        """
        inhibitColumns_(self, overlaps, activeColumns)
        inhibitColumns_(self, py_overlaps) -> PyObject *
        """
        return _algorithms.SpatialPooler_inhibitColumns_(self, *args)

    def updatePermanencesForColumn_(self, *args):
        """
        updatePermanencesForColumn_(self, perm, column, raisePerm=True)
        updatePermanencesForColumn_(self, py_perm, column, raisePerm)
        """
        return _algorithms.SpatialPooler_updatePermanencesForColumn_(self, *args)

    def updateDutyCycles_(self, *args):
        """
        updateDutyCycles_(self, overlaps, activeArray)
        updateDutyCycles_(self, py_overlaps, py_activeArray)
        """
        return _algorithms.SpatialPooler_updateDutyCycles_(self, *args)

SpatialPooler_swigregister = _algorithms.SpatialPooler_swigregister
SpatialPooler_swigregister(SpatialPooler)

def SpatialPooler_updateDutyCyclesHelper_(*args, **kwargs):
  """SpatialPooler_updateDutyCyclesHelper_(dutyCycles, newValues, period)"""
  return _algorithms.SpatialPooler_updateDutyCyclesHelper_(*args, **kwargs)

class SDRClassifier(object):
    """Proxy of C++ nupic::algorithms::sdr_classifier::SDRClassifier class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> SDRClassifier
        __init__(self, steps, alpha, actValueAlpha, verbosity) -> SDRClassifier
        """
        this = _algorithms.new_SDRClassifier(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_SDRClassifier
    def compute(self, *args, **kwargs):
        """compute(self, recordNum, patternNZ, bucketIdxList, actValueList, category, learn, infer, result)"""
        return _algorithms.SDRClassifier_compute(self, *args, **kwargs)

    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.SDRClassifier_version(self)

    def getVerbosity(self):
        """getVerbosity(self) -> nupic::UInt"""
        return _algorithms.SDRClassifier_getVerbosity(self)

    def setVerbosity(self, *args, **kwargs):
        """setVerbosity(self, verbosity)"""
        return _algorithms.SDRClassifier_setVerbosity(self, *args, **kwargs)

    def getAlpha(self):
        """getAlpha(self) -> nupic::Real64"""
        return _algorithms.SDRClassifier_getAlpha(self)

    def persistentSize(self):
        """persistentSize(self) -> size_t"""
        return _algorithms.SDRClassifier_persistentSize(self)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.SDRClassifier_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.SDRClassifier_load(self, *args, **kwargs)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.SDRClassifier___eq__(self, *args, **kwargs)

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


    def loadFromString(self, *args, **kwargs):
        """loadFromString(self, inString)"""
        return _algorithms.SDRClassifier_loadFromString(self, *args, **kwargs)

    def getCState(self):
        """getCState(self) -> PyObject *"""
        return _algorithms.SDRClassifier_getCState(self)

SDRClassifier_swigregister = _algorithms.SDRClassifier_swigregister
SDRClassifier_swigregister(SDRClassifier)
sdrClassifierVersion = cvar.sdrClassifierVersion

import numpy

class ConnectionsSynapseVector(object):
    """Proxy of C++ vector<(nupic::algorithms::connections::Synapse)> class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def iterator(self):
        """iterator(self) -> SwigPyIterator"""
        return _algorithms.ConnectionsSynapseVector_iterator(self)

    def __iter__(self): return self.iterator()
    def __nonzero__(self):
        """__nonzero__(self) -> bool"""
        return _algorithms.ConnectionsSynapseVector___nonzero__(self)

    def __bool__(self):
        """__bool__(self) -> bool"""
        return _algorithms.ConnectionsSynapseVector___bool__(self)

    def __len__(self):
        """__len__(self) -> vector< nupic::algorithms::connections::Synapse >::size_type"""
        return _algorithms.ConnectionsSynapseVector___len__(self)

    def pop(self):
        """pop(self) -> ConnectionsSynapse"""
        return _algorithms.ConnectionsSynapseVector_pop(self)

    def __getslice__(self, *args, **kwargs):
        """__getslice__(self, i, j) -> std::vector< nupic::algorithms::connections::Synapse,std::allocator< nupic::algorithms::connections::Synapse > > *"""
        return _algorithms.ConnectionsSynapseVector___getslice__(self, *args, **kwargs)

    def __setslice__(self, *args, **kwargs):
        """__setslice__(self, i, j, v=std::vector< nupic::algorithms::connections::Synapse,std::allocator< nupic::algorithms::connections::Synapse > >())"""
        return _algorithms.ConnectionsSynapseVector___setslice__(self, *args, **kwargs)

    def __delslice__(self, *args, **kwargs):
        """__delslice__(self, i, j)"""
        return _algorithms.ConnectionsSynapseVector___delslice__(self, *args, **kwargs)

    def __delitem__(self, *args):
        """
        __delitem__(self, i)
        __delitem__(self, slice)
        """
        return _algorithms.ConnectionsSynapseVector___delitem__(self, *args)

    def __getitem__(self, *args):
        """
        __getitem__(self, slice) -> std::vector< nupic::algorithms::connections::Synapse,std::allocator< nupic::algorithms::connections::Synapse > >
        __getitem__(self, i) -> ConnectionsSynapse
        """
        return _algorithms.ConnectionsSynapseVector___getitem__(self, *args)

    def __setitem__(self, *args):
        """
        __setitem__(self, slice, v)
        __setitem__(self, slice)
        __setitem__(self, i, x)
        """
        return _algorithms.ConnectionsSynapseVector___setitem__(self, *args)

    def append(self, *args, **kwargs):
        """append(self, x)"""
        return _algorithms.ConnectionsSynapseVector_append(self, *args, **kwargs)

    def empty(self):
        """empty(self) -> bool"""
        return _algorithms.ConnectionsSynapseVector_empty(self)

    def size(self):
        """size(self) -> vector< nupic::algorithms::connections::Synapse >::size_type"""
        return _algorithms.ConnectionsSynapseVector_size(self)

    def clear(self):
        """clear(self)"""
        return _algorithms.ConnectionsSynapseVector_clear(self)

    def swap(self, *args, **kwargs):
        """swap(self, v)"""
        return _algorithms.ConnectionsSynapseVector_swap(self, *args, **kwargs)

    def get_allocator(self):
        """get_allocator(self) -> vector< nupic::algorithms::connections::Synapse >::allocator_type"""
        return _algorithms.ConnectionsSynapseVector_get_allocator(self)

    def begin(self):
        """begin(self) -> vector< nupic::algorithms::connections::Synapse >::iterator"""
        return _algorithms.ConnectionsSynapseVector_begin(self)

    def end(self):
        """end(self) -> vector< nupic::algorithms::connections::Synapse >::iterator"""
        return _algorithms.ConnectionsSynapseVector_end(self)

    def rbegin(self):
        """rbegin(self) -> vector< nupic::algorithms::connections::Synapse >::reverse_iterator"""
        return _algorithms.ConnectionsSynapseVector_rbegin(self)

    def rend(self):
        """rend(self) -> vector< nupic::algorithms::connections::Synapse >::reverse_iterator"""
        return _algorithms.ConnectionsSynapseVector_rend(self)

    def pop_back(self):
        """pop_back(self)"""
        return _algorithms.ConnectionsSynapseVector_pop_back(self)

    def erase(self, *args):
        """
        erase(self, pos) -> vector< nupic::algorithms::connections::Synapse >::iterator
        erase(self, first, last) -> vector< nupic::algorithms::connections::Synapse >::iterator
        """
        return _algorithms.ConnectionsSynapseVector_erase(self, *args)

    def __init__(self, *args): 
        """
        __init__(self) -> ConnectionsSynapseVector
        __init__(self, arg2) -> ConnectionsSynapseVector
        __init__(self, size) -> ConnectionsSynapseVector
        __init__(self, size, value) -> ConnectionsSynapseVector
        """
        this = _algorithms.new_ConnectionsSynapseVector(*args)
        try: self.this.append(this)
        except: self.this = this
    def push_back(self, *args, **kwargs):
        """push_back(self, x)"""
        return _algorithms.ConnectionsSynapseVector_push_back(self, *args, **kwargs)

    def front(self):
        """front(self) -> ConnectionsSynapse"""
        return _algorithms.ConnectionsSynapseVector_front(self)

    def back(self):
        """back(self) -> ConnectionsSynapse"""
        return _algorithms.ConnectionsSynapseVector_back(self)

    def assign(self, *args, **kwargs):
        """assign(self, n, x)"""
        return _algorithms.ConnectionsSynapseVector_assign(self, *args, **kwargs)

    def resize(self, *args):
        """
        resize(self, new_size)
        resize(self, new_size, x)
        """
        return _algorithms.ConnectionsSynapseVector_resize(self, *args)

    def insert(self, *args):
        """
        insert(self, pos, x) -> vector< nupic::algorithms::connections::Synapse >::iterator
        insert(self, pos, n, x)
        """
        return _algorithms.ConnectionsSynapseVector_insert(self, *args)

    def reserve(self, *args, **kwargs):
        """reserve(self, n)"""
        return _algorithms.ConnectionsSynapseVector_reserve(self, *args, **kwargs)

    def capacity(self):
        """capacity(self) -> vector< nupic::algorithms::connections::Synapse >::size_type"""
        return _algorithms.ConnectionsSynapseVector_capacity(self)

    __swig_destroy__ = _algorithms.delete_ConnectionsSynapseVector
ConnectionsSynapseVector_swigregister = _algorithms.ConnectionsSynapseVector_swigregister
ConnectionsSynapseVector_swigregister(ConnectionsSynapseVector)

class ConnectionsSynapse(object):
    """Proxy of C++ nupic::algorithms::connections::Synapse class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    flatIdx = _swig_property(_algorithms.ConnectionsSynapse_flatIdx_get, _algorithms.ConnectionsSynapse_flatIdx_set)
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


    def __init__(self): 
        """__init__(self) -> ConnectionsSynapse"""
        this = _algorithms.new_ConnectionsSynapse()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_ConnectionsSynapse
ConnectionsSynapse_swigregister = _algorithms.ConnectionsSynapse_swigregister
ConnectionsSynapse_swigregister(ConnectionsSynapse)

class SynapseData(object):
    """Proxy of C++ nupic::algorithms::connections::SynapseData class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    presynapticCell = _swig_property(_algorithms.SynapseData_presynapticCell_get, _algorithms.SynapseData_presynapticCell_set)
    permanence = _swig_property(_algorithms.SynapseData_permanence_get, _algorithms.SynapseData_permanence_set)
    segment = _swig_property(_algorithms.SynapseData_segment_get, _algorithms.SynapseData_segment_set)
    def __init__(self): 
        """__init__(self) -> SynapseData"""
        this = _algorithms.new_SynapseData()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_SynapseData
SynapseData_swigregister = _algorithms.SynapseData_swigregister
SynapseData_swigregister(SynapseData)

class SegmentData(object):
    """Proxy of C++ nupic::algorithms::connections::SegmentData class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    synapses = _swig_property(_algorithms.SegmentData_synapses_get, _algorithms.SegmentData_synapses_set)
    cell = _swig_property(_algorithms.SegmentData_cell_get, _algorithms.SegmentData_cell_set)
    def __init__(self): 
        """__init__(self) -> SegmentData"""
        this = _algorithms.new_SegmentData()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_SegmentData
SegmentData_swigregister = _algorithms.SegmentData_swigregister
SegmentData_swigregister(SegmentData)

class CellData(object):
    """Proxy of C++ nupic::algorithms::connections::CellData class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    segments = _swig_property(_algorithms.CellData_segments_get, _algorithms.CellData_segments_set)
    def __init__(self): 
        """__init__(self) -> CellData"""
        this = _algorithms.new_CellData()
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_CellData
CellData_swigregister = _algorithms.CellData_swigregister
CellData_swigregister(CellData)

class ConnectionsEventHandler(object):
    """Proxy of C++ nupic::algorithms::connections::ConnectionsEventHandler class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    __swig_destroy__ = _algorithms.delete_ConnectionsEventHandler
    def onCreateSegment(self, *args, **kwargs):
        """onCreateSegment(self, segment)"""
        return _algorithms.ConnectionsEventHandler_onCreateSegment(self, *args, **kwargs)

    def onDestroySegment(self, *args, **kwargs):
        """onDestroySegment(self, segment)"""
        return _algorithms.ConnectionsEventHandler_onDestroySegment(self, *args, **kwargs)

    def onCreateSynapse(self, *args, **kwargs):
        """onCreateSynapse(self, synapse)"""
        return _algorithms.ConnectionsEventHandler_onCreateSynapse(self, *args, **kwargs)

    def onDestroySynapse(self, *args, **kwargs):
        """onDestroySynapse(self, synapse)"""
        return _algorithms.ConnectionsEventHandler_onDestroySynapse(self, *args, **kwargs)

    def onUpdateSynapsePermanence(self, *args, **kwargs):
        """onUpdateSynapsePermanence(self, synapse, permanence)"""
        return _algorithms.ConnectionsEventHandler_onUpdateSynapsePermanence(self, *args, **kwargs)

    def __init__(self): 
        """__init__(self) -> ConnectionsEventHandler"""
        if self.__class__ == ConnectionsEventHandler:
            _self = None
        else:
            _self = self
        this = _algorithms.new_ConnectionsEventHandler(_self, )
        try: self.this.append(this)
        except: self.this = this
    def __disown__(self):
        self.this.disown()
        _algorithms.disown_ConnectionsEventHandler(self)
        return weakref_proxy(self)
ConnectionsEventHandler_swigregister = _algorithms.ConnectionsEventHandler_swigregister
ConnectionsEventHandler_swigregister(ConnectionsEventHandler)

class Connections(object):
    """Proxy of C++ nupic::algorithms::connections::Connections class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    VERSION = _algorithms.Connections_VERSION
    def __init__(self, *args): 
        """
        __init__(self) -> Connections
        __init__(self, numCells) -> Connections
        """
        this = _algorithms.new_Connections(*args)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _algorithms.delete_Connections
    def initialize(self, *args, **kwargs):
        """initialize(self, numCells)"""
        return _algorithms.Connections_initialize(self, *args, **kwargs)

    def createSegment(self, *args, **kwargs):
        """createSegment(self, cell) -> nupic::algorithms::connections::Segment"""
        return _algorithms.Connections_createSegment(self, *args, **kwargs)

    def createSynapse(self, *args, **kwargs):
        """createSynapse(self, segment, presynapticCell, permanence) -> ConnectionsSynapse"""
        return _algorithms.Connections_createSynapse(self, *args, **kwargs)

    def destroySegment(self, *args, **kwargs):
        """destroySegment(self, segment)"""
        return _algorithms.Connections_destroySegment(self, *args, **kwargs)

    def destroySynapse(self, *args, **kwargs):
        """destroySynapse(self, synapse)"""
        return _algorithms.Connections_destroySynapse(self, *args, **kwargs)

    def updateSynapsePermanence(self, *args, **kwargs):
        """updateSynapsePermanence(self, synapse, permanence)"""
        return _algorithms.Connections_updateSynapsePermanence(self, *args, **kwargs)

    def segmentsForCell(self, *args, **kwargs):
        """segmentsForCell(self, cell) -> VectorOfUInt32"""
        return _algorithms.Connections_segmentsForCell(self, *args, **kwargs)

    def synapsesForSegment(self, *args, **kwargs):
        """synapsesForSegment(self, segment) -> std::vector< nupic::algorithms::connections::Synapse,std::allocator< nupic::algorithms::connections::Synapse > > const &"""
        return _algorithms.Connections_synapsesForSegment(self, *args, **kwargs)

    def cellForSegment(self, *args, **kwargs):
        """cellForSegment(self, segment) -> nupic::algorithms::connections::CellIdx"""
        return _algorithms.Connections_cellForSegment(self, *args, **kwargs)

    def idxOnCellForSegment(self, *args, **kwargs):
        """idxOnCellForSegment(self, segment) -> nupic::algorithms::connections::SegmentIdx"""
        return _algorithms.Connections_idxOnCellForSegment(self, *args, **kwargs)

    def mapSegmentsToCells(self, *args, **kwargs):
        """mapSegmentsToCells(self, segments_begin, segments_end, cells_begin)"""
        return _algorithms.Connections_mapSegmentsToCells(self, *args, **kwargs)

    def segmentForSynapse(self, *args, **kwargs):
        """segmentForSynapse(self, synapse) -> nupic::algorithms::connections::Segment"""
        return _algorithms.Connections_segmentForSynapse(self, *args, **kwargs)

    def dataForSegment(self, *args, **kwargs):
        """dataForSegment(self, segment) -> SegmentData"""
        return _algorithms.Connections_dataForSegment(self, *args, **kwargs)

    def dataForSynapse(self, *args, **kwargs):
        """dataForSynapse(self, synapse) -> SynapseData"""
        return _algorithms.Connections_dataForSynapse(self, *args, **kwargs)

    def getSegment(self, *args, **kwargs):
        """getSegment(self, cell, idx) -> nupic::algorithms::connections::Segment"""
        return _algorithms.Connections_getSegment(self, *args, **kwargs)

    def segmentFlatListLength(self):
        """segmentFlatListLength(self) -> nupic::UInt32"""
        return _algorithms.Connections_segmentFlatListLength(self)

    def compareSegments(self, *args, **kwargs):
        """compareSegments(self, a, b) -> bool"""
        return _algorithms.Connections_compareSegments(self, *args, **kwargs)

    def synapsesForPresynapticCell(self, *args, **kwargs):
        """synapsesForPresynapticCell(self, presynapticCell) -> std::vector< nupic::algorithms::connections::Synapse,std::allocator< nupic::algorithms::connections::Synapse > >"""
        return _algorithms.Connections_synapsesForPresynapticCell(self, *args, **kwargs)

    def computeActivity(self, *args):
        """
        computeActivity(self, numActiveConnectedSynapsesForSegment, numActivePotentialSynapsesForSegment, activePresynapticCells, 
            connectedPermanence)
        computeActivity(self, numActiveConnectedSynapsesForSegment, numActivePotentialSynapsesForSegment, activePresynapticCell, 
            connectedPermanence)
        """
        return _algorithms.Connections_computeActivity(self, *args)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.Connections_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.Connections_load(self, *args, **kwargs)

    def numCells(self):
        """numCells(self) -> nupic::algorithms::connections::CellIdx"""
        return _algorithms.Connections_numCells(self)

    def numSegments(self, *args):
        """
        numSegments(self) -> nupic::UInt
        numSegments(self, cell) -> nupic::UInt
        """
        return _algorithms.Connections_numSegments(self, *args)

    def numSynapses(self, *args):
        """
        numSynapses(self) -> nupic::UInt
        numSynapses(self, segment) -> nupic::UInt
        """
        return _algorithms.Connections_numSynapses(self, *args)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.Connections___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.Connections___ne__(self, *args, **kwargs)

    def subscribe(self, *args, **kwargs):
        """subscribe(self, handler) -> nupic::UInt32"""
        return _algorithms.Connections_subscribe(self, *args, **kwargs)

    def unsubscribe(self, *args, **kwargs):
        """unsubscribe(self, token)"""
        return _algorithms.Connections_unsubscribe(self, *args, **kwargs)

    def __init__(self,
                 numCells,
                 maxSegmentsPerCell=255,
                 maxSynapsesPerSegment=255):
      self.this = _ALGORITHMS.new_Connections(numCells,
                                              maxSegmentsPerCell,
                                              maxSynapsesPerSegment)


    @classmethod
    ///No longer supporting capnprto
    ///def read(cls, proto):
    ///  instance = cls()
    ///  instance.convertedRead(proto)
    ///  return instance

    ///def write(self, pyBuilder):
    ///  """Serialize the Connections instance using capnp.

    ///  :param: Destination ConnectionsProto message builder
    ///  """
    ///  reader = ConnectionsProto.from_bytes(self._writeAsCapnpPyBytes(),
    ///                        traversal_limit_in_words=_TRAVERSAL_LIMIT_IN_WORDS)
    ///  pyBuilder.from_dict(reader.to_dict())  # copy


    /// def convertedRead(self, proto):
    ///   """Initialize the Connections instance from the given ConnectionsProto
    ///   reader.
    ///
    ///  :param proto: ConnectionsProto message reader containing data from a
    ///                previously serialized Connections instance.

    ///  """
    ///  self._initFromCapnpPyBytes(proto.as_builder().to_bytes()) # copy * 2


    def mapSegmentsToCells(self, segments):
      segments = numpy.asarray(segments, dtype="uint32")
      return self._mapSegmentsToCells(segments)

    def _mapSegmentsToCells(self, *args, **kwargs):
        """_mapSegmentsToCells(self, py_segments) -> PyObject *"""
        return _algorithms.Connections__mapSegmentsToCells(self, *args, **kwargs)

Connections_swigregister = _algorithms.Connections_swigregister
Connections_swigregister(Connections)

class TemporalMemory(object):
    """Proxy of C++ nupic::algorithms::temporal_memory::TemporalMemory class"""
    thisown = _swig_property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    def __init__(self, *args): 
        """
        __init__(self) -> TemporalMemory
        __init__(self, columnDimensions, cellsPerColumn=32, activationThreshold=13, initialPermanence=0.21, 
            connectedPermanence=0.50, minThreshold=10, maxNewSynapseCount=20, permanenceIncrement=0.10, 
            permanenceDecrement=0.10, predictedSegmentDecrement=0.0, 
            seed=42, maxSegmentsPerCell=255, maxSynapsesPerSegment=255, checkInputs=True) -> TemporalMemory
        """
        this = _algorithms.new_TemporalMemory(*args)
        try: self.this.append(this)
        except: self.this = this
    def initialize(self, *args, **kwargs):
        """
        initialize(self, columnDimensions={2048}, cellsPerColumn=32, activationThreshold=13, initialPermanence=0.21, 
            connectedPermanence=0.50, minThreshold=10, maxNewSynapseCount=20, 
            permanenceIncrement=0.10, permanenceDecrement=0.10, predictedSegmentDecrement=0.0, 
            seed=42, maxSegmentsPerCell=255, maxSynapsesPerSegment=255, checkInputs=True)
        """
        return _algorithms.TemporalMemory_initialize(self, *args, **kwargs)

    __swig_destroy__ = _algorithms.delete_TemporalMemory
    def version(self):
        """version(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_version(self)

    def seed_(self, *args, **kwargs):
        """seed_(self, seed)"""
        return _algorithms.TemporalMemory_seed_(self, *args, **kwargs)

    def reset(self):
        """reset(self)"""
        return _algorithms.TemporalMemory_reset(self)

    def activateCells(self, *args, **kwargs):
        """activateCells(self, activeColumnsSize, activeColumns, learn=True)"""
        return _algorithms.TemporalMemory_activateCells(self, *args, **kwargs)

    def activateDendrites(self, learn=True):
        """activateDendrites(self, learn=True)"""
        return _algorithms.TemporalMemory_activateDendrites(self, learn)

    def compute(self, *args, **kwargs):
        """compute(self, activeColumnsSize, activeColumns, learn=True)"""
        return _algorithms.TemporalMemory_compute(self, *args, **kwargs)

    def createSegment(self, *args, **kwargs):
        """createSegment(self, cell) -> nupic::algorithms::connections::Segment"""
        return _algorithms.TemporalMemory_createSegment(self, *args, **kwargs)

    def numberOfCells(self):
        """numberOfCells(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_numberOfCells(self)

    def getColumnDimensions(self):
        """getColumnDimensions(self) -> VectorOfUInt32"""
        return _algorithms.TemporalMemory_getColumnDimensions(self)

    def numberOfColumns(self):
        """numberOfColumns(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_numberOfColumns(self)

    def getCellsPerColumn(self):
        """getCellsPerColumn(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getCellsPerColumn(self)

    def getActivationThreshold(self):
        """getActivationThreshold(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getActivationThreshold(self)

    def setActivationThreshold(self, *args, **kwargs):
        """setActivationThreshold(self, arg2)"""
        return _algorithms.TemporalMemory_setActivationThreshold(self, *args, **kwargs)

    def getInitialPermanence(self):
        """getInitialPermanence(self) -> nupic::algorithms::connections::Permanence"""
        return _algorithms.TemporalMemory_getInitialPermanence(self)

    def setInitialPermanence(self, *args, **kwargs):
        """setInitialPermanence(self, arg2)"""
        return _algorithms.TemporalMemory_setInitialPermanence(self, *args, **kwargs)

    def getConnectedPermanence(self):
        """getConnectedPermanence(self) -> nupic::algorithms::connections::Permanence"""
        return _algorithms.TemporalMemory_getConnectedPermanence(self)

    def setConnectedPermanence(self, *args, **kwargs):
        """setConnectedPermanence(self, arg2)"""
        return _algorithms.TemporalMemory_setConnectedPermanence(self, *args, **kwargs)

    def getMinThreshold(self):
        """getMinThreshold(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getMinThreshold(self)

    def setMinThreshold(self, *args, **kwargs):
        """setMinThreshold(self, arg2)"""
        return _algorithms.TemporalMemory_setMinThreshold(self, *args, **kwargs)

    def getMaxNewSynapseCount(self):
        """getMaxNewSynapseCount(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getMaxNewSynapseCount(self)

    def setMaxNewSynapseCount(self, *args, **kwargs):
        """setMaxNewSynapseCount(self, arg2)"""
        return _algorithms.TemporalMemory_setMaxNewSynapseCount(self, *args, **kwargs)

    def getCheckInputs(self):
        """getCheckInputs(self) -> bool"""
        return _algorithms.TemporalMemory_getCheckInputs(self)

    def setCheckInputs(self, *args, **kwargs):
        """setCheckInputs(self, arg2)"""
        return _algorithms.TemporalMemory_setCheckInputs(self, *args, **kwargs)

    def getPermanenceIncrement(self):
        """getPermanenceIncrement(self) -> nupic::algorithms::connections::Permanence"""
        return _algorithms.TemporalMemory_getPermanenceIncrement(self)

    def setPermanenceIncrement(self, *args, **kwargs):
        """setPermanenceIncrement(self, arg2)"""
        return _algorithms.TemporalMemory_setPermanenceIncrement(self, *args, **kwargs)

    def getPermanenceDecrement(self):
        """getPermanenceDecrement(self) -> nupic::algorithms::connections::Permanence"""
        return _algorithms.TemporalMemory_getPermanenceDecrement(self)

    def setPermanenceDecrement(self, *args, **kwargs):
        """setPermanenceDecrement(self, arg2)"""
        return _algorithms.TemporalMemory_setPermanenceDecrement(self, *args, **kwargs)

    def getPredictedSegmentDecrement(self):
        """getPredictedSegmentDecrement(self) -> nupic::algorithms::connections::Permanence"""
        return _algorithms.TemporalMemory_getPredictedSegmentDecrement(self)

    def setPredictedSegmentDecrement(self, *args, **kwargs):
        """setPredictedSegmentDecrement(self, arg2)"""
        return _algorithms.TemporalMemory_setPredictedSegmentDecrement(self, *args, **kwargs)

    def getMaxSegmentsPerCell(self):
        """getMaxSegmentsPerCell(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getMaxSegmentsPerCell(self)

    def getMaxSynapsesPerSegment(self):
        """getMaxSynapsesPerSegment(self) -> nupic::UInt"""
        return _algorithms.TemporalMemory_getMaxSynapsesPerSegment(self)

    def _validateCell(self, *args, **kwargs):
        """_validateCell(self, cell) -> bool"""
        return _algorithms.TemporalMemory__validateCell(self, *args, **kwargs)

    def save(self, *args, **kwargs):
        """save(self, outStream)"""
        return _algorithms.TemporalMemory_save(self, *args, **kwargs)

    def load(self, *args, **kwargs):
        """load(self, inStream)"""
        return _algorithms.TemporalMemory_load(self, *args, **kwargs)

    def persistentSize(self):
        """persistentSize(self) -> size_t"""
        return _algorithms.TemporalMemory_persistentSize(self)

    def __eq__(self, *args, **kwargs):
        """__eq__(self, other) -> bool"""
        return _algorithms.TemporalMemory___eq__(self, *args, **kwargs)

    def __ne__(self, *args, **kwargs):
        """__ne__(self, other) -> bool"""
        return _algorithms.TemporalMemory___ne__(self, *args, **kwargs)

    def printParameters(self):
        """printParameters(self)"""
        return _algorithms.TemporalMemory_printParameters(self)

    def columnForCell(self, *args, **kwargs):
        """columnForCell(self, cell) -> nupic::UInt"""
        return _algorithms.TemporalMemory_columnForCell(self, *args, **kwargs)

    def printState(self, *args):
        """
        printState(self, state)
        printState(self, state)
        """
        return _algorithms.TemporalMemory_printState(self, *args)

    connections = _swig_property(_algorithms.TemporalMemory_connections_get, _algorithms.TemporalMemory_connections_set)
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



    def getActiveCells(self, *args):
        """
        getActiveCells(self) -> VectorOfUInt32
        getActiveCells(self) -> PyObject *
        """
        return _algorithms.TemporalMemory_getActiveCells(self, *args)

    def getPredictiveCells(self, *args):
        """
        getPredictiveCells(self) -> VectorOfUInt32
        getPredictiveCells(self) -> PyObject *
        """
        return _algorithms.TemporalMemory_getPredictiveCells(self, *args)

    def getWinnerCells(self, *args):
        """
        getWinnerCells(self) -> VectorOfUInt32
        getWinnerCells(self) -> PyObject *
        """
        return _algorithms.TemporalMemory_getWinnerCells(self, *args)

    def getActiveSegments(self, *args):
        """
        getActiveSegments(self) -> VectorOfUInt32
        getActiveSegments(self) -> PyObject *
        """
        return _algorithms.TemporalMemory_getActiveSegments(self, *args)

    def getMatchingSegments(self, *args):
        """
        getMatchingSegments(self) -> VectorOfUInt32
        getMatchingSegments(self) -> PyObject *
        """
        return _algorithms.TemporalMemory_getMatchingSegments(self, *args)

    def cellsForColumn(self, *args):
        """
        cellsForColumn(self, column) -> VectorOfUInt32
        cellsForColumn(self, columnIdx) -> PyObject *
        """
        return _algorithms.TemporalMemory_cellsForColumn(self, *args)

    def convertedActivateCells(self, *args, **kwargs):
        """convertedActivateCells(self, py_activeColumns, learn)"""
        return _algorithms.TemporalMemory_convertedActivateCells(self, *args, **kwargs)

    def convertedCompute(self, *args, **kwargs):
        """convertedCompute(self, py_activeColumns, learn)"""
        return _algorithms.TemporalMemory_convertedCompute(self, *args, **kwargs)

    def loadFromString(self, *args, **kwargs):
        """loadFromString(self, inString)"""
        return _algorithms.TemporalMemory_loadFromString(self, *args, **kwargs)

    def getCState(self):
        """getCState(self) -> PyObject *"""
        return _algorithms.TemporalMemory_getCState(self)

TemporalMemory_swigregister = _algorithms.TemporalMemory_swigregister
TemporalMemory_swigregister(TemporalMemory)



