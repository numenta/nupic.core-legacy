# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2014-2015, Numenta, Inc.
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

import unittest
import pytest

try:
    import cPickle as pickle # For python 2
except ImportError:
    import pickle # For python 3

class TemporalMemoryBindingsTest(unittest.TestCase):
  @pytest.mark.skip(reason="Calling arguments on compute()...another PR")
  @staticmethod
  def testIssue807():
    # The following should silently pass.  Previous versions segfaulted.
    # See https://github.com/numenta/nupic.core/issues/807 for context
    from htm.bindings.algorithms import TemporalMemory

    tm = TemporalMemory()
    tm.compute(set(), True)

  def testNupicTemporalMemoryPickling(self):
    """Test pickling / unpickling of NuPIC TemporalMemory."""
    from htm.bindings.algorithms import TemporalMemory

    # Simple test: make sure that dumping / loading works...
    tm = TemporalMemory(columnDimensions=(16,))
    pickledTm = pickle.dumps(tm)

    tm2 = pickle.loads(pickledTm)

    self.assertEqual(tm.numberOfCells(), tm2.numberOfCells(),
                     "Simple NuPIC TemporalMemory pickle/unpickle failed.")
