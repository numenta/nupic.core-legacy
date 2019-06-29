# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2015, Numenta, Inc.
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
# ----------------------------------------------------------------------

import numpy

from htm.bindings.regions.PyRegion import PyRegion


class TestNode(PyRegion):


  @classmethod
  def getSpec(cls):
    if hasattr(TestNode, '_failIngetSpec'):
      assert False, 'Failing in TestNode.getSpec() as requested'
    result = dict(
      description='The node spec of the NuPIC 2 Python TestNode',
      singleNodeOnly=False,
      inputs=dict(
        bottomUpIn=dict(
          description='Primary input for the node',
          dataType='Real64',
          count=0,
          required=True,
          regionLevel=False,
          isDefaultInput=True,
          requireSplitterMap=False
          )
        ),
      outputs=dict(
        bottomUpOut=dict(
          description='Primary output for the node',
          dataType='Real64',
          count=0,
          regionLevel=False,
          isDefaultOutput=True
          )
        ),
      parameters=dict(
	    count=dict(
		  description='size of output buffer',
		  dataType='UInt32',
		  count=1,
		  constraints='',
		  defaultValue='64',
		  accessMode='ReadWrite'
		),
        int32Param=dict(
          description='Int32 scalar parameter',
          dataType='Int32',
          count=1,
          constraints='',
          defaultValue='32',
          accessMode='ReadWrite'
        ),
        uint32Param=dict(
          description='UInt32 scalar parameter',
          dataType='UInt32',
          count=1,
          constraints='',
          defaultValue='33',
          accessMode='ReadWrite'
        ),
        int64Param=dict(
          description='Int64 scalar parameter',
          dataType='Int64',
          count=1,
          constraints='',
          defaultValue='64',
          accessMode='ReadWrite'
        ),
        uint64Param=dict(
          description='UInt64 scalar parameter',
          dataType='UInt64',
          count=1,
          constraints='',
          defaultValue='65',
          accessMode='ReadWrite'
        ),
        real32Param=dict(
          description='Real32 scalar parameter',
          dataType='Real32',
          count=1,
          constraints='',
          defaultValue='32.1',
          accessMode='ReadWrite'
        ),
        real64Param=dict(
          description='Real64 scalar parameter',
          dataType='Real64',
          count=1,
          constraints='',
          defaultValue='64.1',
          accessMode='ReadWrite'
        ),
        boolParam=dict(
          description='bool parameter',
          dataType='Bool',
          count=1,
          constraints='',
          defaultValue='false',
          accessMode='ReadWrite'
        ),
        real32arrayParam=dict(
          description='Real32 array parameter',
          dataType='Real32',
          count=0, # array
          constraints='',
          defaultValue='',
          accessMode='ReadWrite'
        ),
        int64arrayParam=dict(
          description='Int64 array parameter',
          dataType='Int64',
          count=0, # array
          constraints='',
          defaultValue='',
          accessMode='ReadWrite'
        ),
        boolArrayParam=dict(
          description='bool array parameter',
          dataType='Bool',
          count=0, # array
          constraints='',
          defaultValue='',
          accessMode='ReadWrite'
        ),
        stringParam=dict(
          description='String parameter',
          dataType='Byte',
          count=0, # string is conventionally Byte/0
          constraints='',
          defaultValue='nodespec value',
          accessMode='ReadWrite'
        ),
        failInInit=dict(
          description='For testing failure in __init__()',
          dataType='Int32',
          count=1,
          constraints='',
          defaultValue='0',
          accessMode='ReadWrite'
        ),
        failInCompute=dict(
          description='For testing failure in compute()',
          dataType='Int32',
          count=1,
          constraints='',
          defaultValue='0',
          accessMode='ReadWrite'
        ),
      ),
      commands=dict()
    )

    print(result)
    return result


  def __init__(self, *args, **kwargs):
    """ """
    # Facilitate failing in __init__ to test error handling
    if 'failInInit' in kwargs:
      assert False, 'TestNode.__init__() Failing on purpose as requested'

    # Check if should fail in compute to test error handling
    self._failInCompute = kwargs.pop('failInCompute', False)

    # set these to a bunch of incorrect values, just to make
    # sure they are set correctly by the nodespec.
    self.parameters = dict(
      count=64,
      int32Param=32,
      uint32Param=33,
      int64Param=64,
      uint64Param=65,
      real32Param=32.1,
      real64Param=64.1,
      boolParam=False,
      real32ArrayParam=numpy.arange(10).astype('float32'),
      real64ArrayParam=numpy.arange(10).astype('float64'),
      # Construct int64 array in the same way as in C++
      int64ArrayParam=numpy.arange(4).astype('int64'),
      boolArrayParam=numpy.array([False]*4),
      stringParam="nodespec value")

    for key in kwargs:
      if not key in self.parameters:
        raise Exception("TestNode found keyword %s but there is no parameter with that name" % key)
      self.parameters[key] = kwargs[key]

    self.outputElementCount = 2 # used for computation
    self._delta = 1
    self._iter = 0
    for i in range(0,4):
      self.parameters["int64ArrayParam"][i] = i*64


  def getParameter(self, name, index):
    assert name in self.parameters
    return self.parameters[name]


  def setParameter(self, name, index, value):
    assert name in self.parameters
    self.parameters[name] = value


  def initialize(self):
    print('TestNode.initialize() here.')


  def compute(self, inputs, outputs):
    if self._failInCompute:
      assert False, 'TestNode.compute() Failing on purpose as requested'


  def getOutputElementCount(self, name):
    assert name == 'bottomUpOut'
    return self.outputElementCount


  def getParameterArrayCount(self, name, index):
    assert name.endswith('ArrayParam')
    print('len(self.parameters[%s]) = %d' % (name, len(self.parameters[name])))
    return len(self.parameters[name])


  def getParameterArray(self, name, index, array):
    assert name.endswith('ArrayParam')
    assert name in self.parameters
    v = self.parameters[name]
    assert len(array) == len(v)
    assert array.dtype == v.dtype
    array[:] = v


  def setParameterArray(self, name, index, array):
    assert name.endswith('ArrayParam')
    assert name in self.parameters
    assert array.dtype == self.parameters[name].dtype
    self.parameters[name] = numpy.array(array)


  def writeArray(self, regionImpl, name, dtype, castFn):
    count = self.getParameterArrayCount(name, 0)
    param = numpy.zeros(count, dtype=dtype)
    self.getParameterArray(name, 0, param)
    field = regionImpl.init(name, count)
    for i in range(count):
      field[i] = castFn(param[i])


  def readArray(self, regionImpl, name, dtype):
    field = getattr(regionImpl, name)
    count = len(field)
    param = numpy.zeros(count, dtype=dtype)
    for i in range(count):
      param[i] = field[i]
    self.setParameter(name, 0, param)

