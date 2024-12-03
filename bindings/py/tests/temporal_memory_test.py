# Copyright 2014-2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

class TemporalMemoryBindingsTest(unittest.TestCase):
  @staticmethod
  def testIssue807():
    # The following should silently pass.  Previous versions segfaulted.
    # See https://github.com/numenta/nupic.core/issues/807 for context
    from nupic.bindings.algorithms import TemporalMemory

    tm = TemporalMemory()
    tm.compute(set(), True)
