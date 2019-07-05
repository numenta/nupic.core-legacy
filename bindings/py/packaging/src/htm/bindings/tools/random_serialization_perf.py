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

"""Serialization performance test of the extension-based Random class.
"""

import time

from htm.bindings.math import Random



_SERIALIZATION_LOOPS = 100000
_DESERIALIZATION_LOOPS = 100000


def main():
  """Measure serialization performance of Random
  """
  r = Random(42)

  # Measure serialization
  startSerializationTime = time.time()

  for i in range(_SERIALIZATION_LOOPS):
    r.saveToFile("RandonSerialization.stream")

  elapsedSerializationTime = time.time() - startSerializationTime

  # Measure deserialization
  startDeserializationTime = time.time()

  for _ in range(_DESERIALIZATION_LOOPS):
    r.loadFromFile("RandonSerialization.stream")

  elapsedDeserializationTime = time.time() - startDeserializationTime

  # Print report
  print(_SERIALIZATION_LOOPS, "Serialization loops in", \
        elapsedSerializationTime, "seconds.")
  print("\t", elapsedSerializationTime/_SERIALIZATION_LOOPS, "seconds per loop.")

  print(deserializationCount, "Deserialization loops in", \
        elapsedDeserializationTime, "seconds.")
  print("\t", elapsedDeserializationTime/deserializationCount, "seconds per loop.")

if __name__ == "__main__":
  main()
