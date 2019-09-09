"""Unit tests for temporal memory."""

# disable pylint warning: "Access to a protected member xxxxx of a client class"
# pylint: disable=W0212

from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR
import unittest

import mock
from unittest import TestCase as TestCaseBase

debugPrint = False

parameters1 = {

 'sp': {'boostStrength': 3.0,
        'columnCount': 1638,
        'localAreaDensity': 0.04395604395604396,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1},
}
  
class TemporalMemoryClassTest(TestCaseBase):
  """Tests the high-level Temporal Memory class"""

  def testPredictiveCells(self):
    """
    This tests that we don't get empty predicitve cells
    """
    
    tm = TemporalMemory(
        columnDimensions=(parameters1["sp"]["columnCount"],),
        cellsPerColumn=parameters1["tm"]["cellsPerColumn"],
        activationThreshold=parameters1["tm"]["activationThreshold"],
        initialPermanence=parameters1["tm"]["initialPerm"],
        connectedPermanence=parameters1["sp"]["synPermConnected"],
        minThreshold=parameters1["tm"]["minThreshold"],
        maxNewSynapseCount=parameters1["tm"]["newSynapseCount"],
        permanenceIncrement=parameters1["tm"]["permanenceInc"],
        permanenceDecrement=parameters1["tm"]["permanenceDec"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=parameters1["tm"]["maxSegmentsPerCell"],
        maxSynapsesPerSegment=parameters1["tm"]["maxSynapsesPerSegment"],
    )
    
    activeColumnsA = SDR(parameters1["sp"]["columnCount"])
    activeColumnsB = SDR(parameters1["sp"]["columnCount"])
    
    activeColumnsA.randomize(sparsity=0.4,seed=1)
    activeColumnsB.randomize(sparsity=0.4,seed=1)
    
    # give pattern A - bursting
    # give pattern B - bursting
    # give pattern A - should be predicting
    
    tm.activateDendrites(True)
    predictiveCellsSDR = tm.getPredictiveCells()
    tm.activateCells(activeColumnsA,True)
    _print("\nColumnsA")
    _print("activeCols:"+str(len(activeColumnsA.sparse)))
    _print("activeCells:"+str(len(tm.getActiveCells().sparse)))
    _print("predictiveCells:"+str(len(predictiveCellsSDR.sparse)))
    
    
    tm.activateDendrites(True)
    predictiveCellsSDR = tm.getPredictiveCells()
    tm.activateCells(activeColumnsB,True)

    _print("\nColumnsB")
    _print("activeCols:"+str(len(activeColumnsB.sparse)))
    _print("activeCells:"+str(len(tm.getActiveCells().sparse)))
    _print("predictiveCells:"+str(len(predictiveCellsSDR.sparse)))
    
    tm.activateDendrites(True)
    predictiveCellsSDR = tm.getPredictiveCells()
    tm.activateCells(activeColumnsA,True)
    _print("\nColumnsA")
    _print("activeCols:"+str(len(activeColumnsA.sparse)))
    _print("activeCells:"+str(len(tm.getActiveCells().sparse)))
    _print("predictiveCells:"+str(len(predictiveCellsSDR.sparse)))
    
   
    self.assertTrue(len(predictiveCellsSDR.sparse)>0)

def _print(txt):
    if debugPrint:
        print(txt)
if __name__ == "__main__":
  unittest.main()