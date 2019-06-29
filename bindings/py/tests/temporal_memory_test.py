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

class TemporalMemoryBindingsTest(unittest.TestCase):
  @pytest.mark.skip(reason="Calling arguments on compute()...another PR")
  @staticmethod
  def testIssue807():
    # The following should silently pass.  Previous versions segfaulted.
    # See https://github.com/numenta/nupic.core/issues/807 for context
    from htm.bindings.algorithms import TemporalMemory

    tm = TemporalMemory()
    tm.compute(set(), True)
