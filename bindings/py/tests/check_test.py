# Copyright 2015 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest
import nupic.bindings.check as check



class LoadBindingsTest(unittest.TestCase):


  def testImportBindingsInstalled(self):
    """Test that we can import nupic.bindings"""
    check.checkImportBindingsInstalled()


  def testImportBindingsExtensions(self):
    """Test that we can load C extensions under nupic.binding"""
    check.checkImportBindingsExtensions()
