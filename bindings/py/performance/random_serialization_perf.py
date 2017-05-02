#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""NuPIC random module tests."""

import time

# NOTE need to import capnp first to activate the magic necessary for
# RandomProto_capnp, etc.
import capnp
from nupic.proto.RandomProto_capnp import RandomProto

from nupic.bindings.math import Random

_SERIALIZATION_LOOPS = 100000
_DESERIALIZATION_LOOPS = 100000


def main():
  """Measure capnp serialization performance of Random
  """

  # Simple test: make sure that dumping / loading works...
  r = Random(42)

  # Measure serialization
  startSerializationTime = time.time()

  builderProto = RandomProto.new_message()
  for i in xrange(_SERIALIZATION_LOOPS):
    r.write(builderProto)

  elapsedSerializationTime = time.time() - startSerializationTime


  readerProto = RandomProto.from_bytes(builderProto.to_bytes())


  # Measure deserialization
  startDeserializationTime = time.time()

  for i in xrange(_DESERIALIZATION_LOOPS):
    print r.read(readerProto)

  elapsedDeserializationTime = time.time() - startDeserializationTime

  # Print report
  print _SERIALIZATION_LOOPS, "Serialization loops in", \
        elapsedSerializationTime, "seconds."
  print "\t", elapsedSerializationTime/_SERIALIZATION_LOOPS, "seconds per loop."

  print _DESERIALIZATION_LOOPS, "Deserialization loops in", \
        elapsedDeserializationTime, "seconds."
  print "\t", elapsedDeserializationTime/_DESERIALIZATION_LOOPS, "seconds per loop."

if __name__ == "__main__":
  main()
