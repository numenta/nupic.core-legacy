'''
Created on 6 Nov 2019

@author: fred
'''
import unittest

import pickle
import sys
import numpy as np

from htm.advanced.support.register_regions import registerAllAdvancedRegions
from htm.bindings.engine_internal import Network
from htm.advanced.frameworks.location.location_network_creation import createL4L6aLocationColumn

cortical_params = {
  # Column parameters
    
  # L2 Parameters
  # Adapted from htmresearch.frameworks.layers.l2_l4_inference.L4L2Experiment#getDefaultL2Params
  # L4 Parameters
    "l4_cellsPerColumn": 24,
    "l4_columnCount": 16,
    "l4_connectedPermanence": 0.6,
    "l4_permanenceIncrement": 0.1,
    "l4_permanenceDecrement": 0.02,
    "l4_apicalPredictedSegmentDecrement": 0.0,
    "l4_basalPredictedSegmentDecrement": 0.0,
    "l4_initialPermanence": 1.0,
    "l4_activationThreshold": 8,
    "l4_minThreshold": 8,
    "l4_reducedBasalThreshold": 8,
    "l4_sampleSize": 10,
    "l4_implementation": "ApicalTiebreak",

  # L6a Parameters
    "l6a_moduleCount": 10,
    "l6a_dimensions": 2,
    "l6a_connectedPermanence": 0.5,
    "l6a_permanenceIncrement": 0.1,
    "l6a_permanenceDecrement": 0.0,
    "l6a_initialPermanence": 1.0,
    "l6a_activationThreshold": 8,
    "l6a_initialPermanence": 1.0,
    "l6a_learningThreshold": 8,
    "l6a_sampleSize": 10,
    "l6a_cellsPerAxis": 10,
    "l6a_scale": 10,
    "l6a_orientation": 60,
    "l6a_bumpOverlapMethod": "probabilistic"
    }

class TestSimpleSPTMNetwork(unittest.TestCase):

    def _create_network(self, L4Params, L6aParams):
        """
        Constructor.
        """
        network = Network()
        
        # Create network
        network = createL4L6aLocationColumn(network=network,
                                            L4Params=L4Params,
                                            L6aParams=L6aParams,
                                            inverseReadoutResolution=None,
                                            baselineCellsPerAxis=6,
                                            suffix="")

        network.initialize()
        return network
        
    def setUp(self):
        registerAllAdvancedRegions()

        self._params = cortical_params
        
        L4Params = {param.split("_")[1]:value for param, value in self._params.items() if param.startswith("l4")}
        L6aParams = {param.split("_")[1]:value for param, value in self._params.items() if param.startswith("l6a")}
        # Configure L6a self._htm_parameters
        numModules = L6aParams["moduleCount"]
        L6aParams["scale"] = [L6aParams["scale"]] * numModules
        angle = L6aParams["orientation"] // numModules
        orientation = list(range(angle // 2, angle * numModules, angle))
        L6aParams["orientation"] = np.radians(orientation).tolist()
                
        self.network = self._create_network(L4Params, L6aParams)
        
    def tearDown(self):
        self.network = None
        
    def _run_network(self, network):
        """
        Run the network with fixed data.
        """
        motorInput = network.getRegion("motorInput")
        sensorInput = network.getRegion("sensorInput")
        motorInput.executeCommand('addDataToQueue', [0,0])
        sensorInput.executeCommand('addDataToQueue', [1,2,3], False, 0)
        
        network.run(1)
        
        L4Region = network.getRegion("L4")
        activeL4Cells = np.array(L4Region.getOutputArray("activeCells")).nonzero()[0]
        
        return activeL4Cells
        
    def testAL246aCorticalColumnPickle(self):
        """
        Test that L246aCorticalColumn can be pickled.
        """
        if sys.version_info[0] >= 3:
            proto = 3
        else:
            proto = 2
        # Simple test: make sure that dumping / loading works...
        pickledColumn = pickle.dumps(self.network, proto)
        network2 = pickle.loads(pickledColumn)
        s1 = self._run_network(self.network)
        s2 = self._run_network(network2)
        self.assertTrue(np.array_equal(s1, s2))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()