#!/usr/bin/env python
# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2017, Numenta, Inc.
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

"""serialization performance test that involves a network that contains
a simple PyRegion which in turn contains an extension-based Random instance.
"""

import json
import time

import htm.bindings.engine_internal as engine
from htm.bindings.tools.serialization_test_py_region import \
     SerializationTestPyRegion



_SERIALIZATION_LOOPS = 100000
_DESERIALIZATION_LOOPS = 100000



def _runTest():
  net = engine.Network()
  net.addRegion(SerializationTestPyRegion.__name__,
                "py." + SerializationTestPyRegion.__name__,
                json.dumps({
                  "dataWidth": 128,
                  "randomSeed": 99,
                }))

  # Measure serialization
  startSerializationTime = time.time()

  # serialize 100000 times to a file.
  for i in range(_SERIALIZATION_LOOPS):
    net.saveToFile("SerializationTest.stream")

  elapsedSerializationTime = time.time() - startSerializationTime

  # Measure deserialization
  startDeserializationTime = time.time()

  deserializationCount = 0
  for i in range(_DESERIALIZATION_LOOPS):
    net = engine.Network()
    net.loadFromFile("SerializationTest.stream")

  elapsedDeserializationTime = time.time() - startDeserializationTime

  # Print report
  print(_SERIALIZATION_LOOPS, "Serialization loops in", \
        elapsedSerializationTime, "seconds.")
  print("\t", elapsedSerializationTime/_SERIALIZATION_LOOPS, "seconds per loop.")

  print(deserializationCount, "Deserialization loops in", \
        elapsedDeserializationTime, "seconds.")
  print("\t", elapsedDeserializationTime/deserializationCount, "seconds per loop.")



def main():
  """Measure serialization performance of a network containing a simple
  python region that in-turn contains a Random instance.
  """
  engine.Network.registerPyRegion(__name__,
                                    SerializationTestPyRegion.__name__)

  try:
    _runTest()
  finally:
    engine.Network.unregisterPyRegion(SerializationTestPyRegion.__name__)

if __name__ == "__main__":
  main()
