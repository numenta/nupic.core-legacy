# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Frederick C. Rotbart 
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
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import unittest

from htm.advanced.regions import extractList, asBool

class RegionInitTests(unittest.TestCase):

    def testExtractListInt(self):
        """
        Test that extractList extracts ints.
        """
        list_string = '[0, 1, 2, 3, 42]'
        extracted_list = extractList(list_string, int)
        self.assertEqual([0,1,2,3,42], extracted_list)
        list_string = '[ 0, 1, 2, 3, 42]'
        extracted_list = extractList(list_string, int)
        self.assertEqual([0,1,2,3,42], extracted_list)

    def testExtractListFloat(self):
        """
        Test that extractList extracts floats.
        """
        list_string = '[0.0, 1.0, 2.0, 3.1, 4.2]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0,1.0,2.0,3.1,4.2], extracted_list)
        list_string = '[ 0.0, 1.0, 2.0, 3.1, 4.2]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0,1.0,2.0,3.1,4.2], extracted_list)

    def testExtractListNoType(self):
        """
        Test that extractList extracts without type.
        """
        list_string = '[0.0, 1.0, 2.0, 3.1, 4.2]'
        extracted_list = extractList(list_string)
        self.assertEqual([0.0,1.0,2.0,3.1,4.2], extracted_list)

        list_string = '[0, 1, 2, 3, 42]'
        extracted_list = extractList(list_string)
        self.assertEqual([0,1,2,3,42], extracted_list)

    def testExtractArrayInt(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[0 1 2 3 4]'
        extracted_list = extractList(list_string, int)
        self.assertEqual([0, 1, 2, 3, 4], extracted_list)
        list_string = '[ 0 1 2 3 4]'
        extracted_list = extractList(list_string, int)
        self.assertEqual([0, 1, 2, 3, 4], extracted_list)

    def testExtractArrayFloat(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[0.0 1.0 2.0 3.1 4.2]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0, 1.0, 2.0, 3.1, 4.2], extracted_list)
        list_string = '[ 0.0 1.0 2.0 3.1 4.2]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0, 1.0, 2.0, 3.1, 4.2], extracted_list)

    def testExtractNegativeFloat(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[-20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, -20.0], extracted_list)
        list_string = '[ -20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, -20.0], extracted_list)
        list_string = '[-20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, -20.0], extracted_list)
        list_string = '[ -20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, -20.0], extracted_list)

    def testExtractArrayMixedSignFloat(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([20.0, -20.0], extracted_list)
        list_string = '[ 20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        list_string = '[  20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        list_string = '[   20.0, -20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([20.0, -20.0], extracted_list)
        list_string = '[-20.0,20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, 20.0], extracted_list)
        list_string = '[-20.0, 20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, 20.0], extracted_list)
        list_string = '[-20.0,  20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, 20.0], extracted_list)
        list_string = '[ -20.0,20.0]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20.0, 20.0], extracted_list)

    def testExtractArrayMixedSignInt(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[20, -20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([20, -20], extracted_list)
        list_string = '[ 20, -20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([20, -20], extracted_list)
        list_string = '[-20, 20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, 20], extracted_list)
        list_string = '[ -20, 20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, 20], extracted_list)

    def testExtractArrayNegativeInt(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[-20, -20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, -20], extracted_list)
        list_string = '[ -20, -20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, -20], extracted_list)
        list_string = '[-20,-20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, -20], extracted_list)
        list_string = '[ -20,-20]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([-20, -20], extracted_list)

    def testExtractArrayFloatNoTrailingZero(self):
        """
        Test that extractList extracts floats from np.array string.
        """
        list_string = '[0. 1. 2.]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0, 1.0, 2.0], extracted_list)
        list_string = '[ 0. 1. 2.]'
        extracted_list = extractList(list_string, float)
        self.assertEqual([0.0, 1.0, 2.0], extracted_list)

    def testExtractListEmpty(self):
        """
        Test that accepts empty lists.
        """
        list_string = ''
        extracted_list = extractList(list_string, float)
        self.assertEqual([], extracted_list)

    def testExtractNestedLists(self):
        """
        Test that extractList extracts nested lists.
        """
        list_string = '[0, 1, 2, 3, 42, [67, 89], "Hello World"]'
        extracted_list = extractList(list_string)
        self.assertEqual([0, 1, 2, 3, 42, [67, 89], "Hello World"], extracted_list)
        list_string = '[ 0, 1, 2, 3, 42, [67, 89], "Hello World"]'
        extracted_list = extractList(list_string)
        self.assertEqual([0, 1, 2, 3, 42, [67, 89], "Hello World"], extracted_list)
        list_string = '[ 0, 1, 2, 3, 42, [ 67, 89], "Hello World"]'
        extracted_list = extractList(list_string)
        self.assertEqual([0, 1, 2, 3, 42, [ 67, 89], "Hello World"], extracted_list)
        list_string = '[0, 1, 2, 3, 42, [ 67, 89], "Hello World"]'
        extracted_list = extractList(list_string)
        self.assertEqual([0, 1, 2, 3, 42, [ 67, 89], "Hello World"], extracted_list)

    def testAsBool(self):
        """"
        Check that casting to a bool gives the correct result for supported variants.
        """
        self.assertTrue(asBool(True))
        
        self.assertFalse(asBool(False))

        self.assertTrue(asBool('True'))
        
        self.assertFalse(asBool('False'))
        
        self.assertTrue(asBool(1))
        
        self.assertFalse(asBool(0))
        
        self.assertTrue(asBool('1'))
        
        self.assertFalse(asBool('0'))
        
