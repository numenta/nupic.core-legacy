import csv
import datetime
import os
import numpy as np
import random

from nupic.bindings.algorithms import SpatialPooler
from nupic.bindings.algorithms import TemporalMemory
from nupic.bindings.sdr import SDR, Metrics
from nupic.encoders.rdse import RDSE, RDSE_Parameters
from nupic.encoders.date import DateEncoder
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, "gymdata.csv")

default_parameters = {
 'enc': {'resolution': 0.9551003024002529,
         'size': 692,
         'sparsity': 0.02349173387867304},
 'sp': {'boostStrength': 2.331600954004477,
        'columnCount': 1905,
        'numActiveColumnsPerInhArea': 34,
        'potentialPct': 0.9616584936945282,
        'synPermActiveInc': 0.034755234293843695,
        'synPermConnected': 0.0709642810371415,
        'synPermInactiveDec': 0.007365910271986643},
 'time': {'timeOfDay': (30, 2), 'weekend': 29},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 27,
        'initialPerm': 0.20991245171830633,
        'maxSegmentsPerCell': 152,
        'maxSynapsesPerSegment': 25,
        'minThreshold': 7,
        'newSynapseCount': 20,
        'permanenceDec': 0.09335662142342668,
        'permanenceInc': 0.08673586267418514}}


def main(parameters=default_parameters, argv=None, verbose=True):

  timeOfDayEncoder = DateEncoder(parameters["time"]["timeOfDay"])
  weekendEncoder   = DateEncoder(parameters["time"]["weekend"])

  rdseParams = RDSE_Parameters()
  rdseParams.size       = parameters["enc"]["size"]
  rdseParams.sparsity   = parameters["enc"]["sparsity"]
  rdseParams.resolution = parameters["enc"]["resolution"]
  scalarEncoder = RDSE( rdseParams )

  encodingWidth = (timeOfDayEncoder.size
                   + weekendEncoder.size
                   + scalarEncoder.size)
  enc_info = Metrics( [encodingWidth], 999999999 )

  spParams = parameters["sp"]
  sp = SpatialPooler(
    inputDimensions            = (encodingWidth,),
    columnDimensions           = (spParams["columnCount"],),
    potentialPct               = spParams["potentialPct"],
    potentialRadius            = encodingWidth,
    globalInhibition           = True,
    localAreaDensity           = -1,
    numActiveColumnsPerInhArea = spParams["numActiveColumnsPerInhArea"],
    synPermInactiveDec         = spParams["synPermInactiveDec"],
    synPermActiveInc           = spParams["synPermActiveInc"],
    synPermConnected           = spParams["synPermConnected"],
    boostStrength              = spParams["boostStrength"],
    wrapAround                 = True
  )
  sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

  tmParams = parameters["tm"]
  tm = TemporalMemory(
    columnDimensions          = (spParams["columnCount"],),
    cellsPerColumn            = tmParams["cellsPerColumn"],
    activationThreshold       = tmParams["activationThreshold"],
    initialPermanence         = tmParams["initialPerm"],
    connectedPermanence       = spParams["synPermConnected"],
    minThreshold              = tmParams["minThreshold"],
    maxNewSynapseCount        = tmParams["newSynapseCount"],
    permanenceIncrement       = tmParams["permanenceInc"],
    permanenceDecrement       = tmParams["permanenceDec"],
    predictedSegmentDecrement = 0.0,
    maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
  )
  tm_info = Metrics( [tm.numberOfCells()], 999999999 )

  anomaly_history = AnomalyLikelihood()

  # Read the input file.
  records = []
  with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    headers = next(reader)
    next(reader)
    next(reader)
    for record in reader:
      records.append(record)
  for i in range(100):
    records.append( random.choice(records) )

  inputs  = []
  anomaly = []
  for record in records:

    # Convert data string into Python date object.
    dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
    # Convert data value string into float.
    consumption = float(record[1])

    # Now we call the encoders to create bit representations for each value.
    timeOfDayBits   = timeOfDayEncoder.encode(dateString)
    weekendBits     = weekendEncoder.encode(dateString)
    consumptionBits = scalarEncoder.encode(consumption)

    # Concatenate all these encodings into one large encoding for Spatial
    # Pooling.
    encoding = SDR( encodingWidth ).concatenate(
                                [consumptionBits, timeOfDayBits, weekendBits])
    enc_info.addData( encoding )

    # Create an SDR to represent active columns, This will be populated by the
    # compute method below. It must have the same dimensions as the Spatial
    # Pooler.
    activeColumns = SDR( sp.getColumnDimensions() )

    # Execute Spatial Pooling algorithm over input space.
    sp.compute(encoding, True, activeColumns)
    sp_info.addData( activeColumns )

    # Execute Temporal Memory algorithm over active mini-columns.
    tm.compute(activeColumns, learn=True)
    tm_info.addData( tm.getActiveCells().flatten() )

    anomalyLikelihood = anomaly_history.anomalyProbability( consumption, tm.anomaly )

    inputs.append( consumption )
    anomaly.append( tm.anomaly )
    # anomaly.append( anomalyLikelihood )

  if verbose:
    print("Encoder", enc_info)
    print("")
    print("Spatial Pooler", sp_info)
    print("")
    print("Temporal Memory", tm_info)
    import matplotlib.pyplot as plt
    inputs = np.array(inputs) / max(inputs)
    plt.plot(np.arange(len(inputs)), inputs, 'blue',
             np.arange(len(inputs)), anomaly, 'red')
    plt.show()

  anom_low  = np.mean(anomaly[-300:-100])
  anom_high = np.mean(anomaly[-100:])
  return anom_high - anom_low


if __name__ == "__main__":
  main()
