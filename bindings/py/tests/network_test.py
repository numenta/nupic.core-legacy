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

import unittest

import nupic.bindings.engine_internal as engine



class NetworkTest(unittest.TestCase):

  def testSimpleTwoRegionNetworkIntrospection(self):
    # Create Network instance
    network = engine.Network()

    # Add two TestNode regions to network
    network.addRegion("region1", "TestNode", "")
    network.addRegion("region2", "TestNode", "")

    # Set dimensions on first region
    region1 = network.getRegions().getByName("region1")
    region1.setDimensions(engine.Dimensions([1, 1]))

    # Link region1 and region2
    network.link("region1", "region2", "UniformLink", "")

    # Initialize network
    network.initialize()

    for linkName, link in network.getLinks():
      # Compare Link API to what we know about the network
      self.assertEqual(link.toString(), linkName)
      self.assertEqual(link.getDestRegionName(), "region2")
      self.assertEqual(link.getSrcRegionName(), "region1")
      self.assertEqual(link.getLinkType(), "UniformLink")
      self.assertEqual(link.getDestInputName(), "bottomUpIn")
      self.assertEqual(link.getSrcOutputName(), "bottomUpOut")
      break
    else:
      self.fail("Unable to iterate network links.")


