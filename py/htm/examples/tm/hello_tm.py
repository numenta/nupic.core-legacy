# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2013, Numenta, Inc.
#               2019, David McDougall
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

__doc__ = """
This program shows how to access the Temporal Memory algorithm directly.  This
program demonstrates how to create a TM instance, train it, get predictions and
anomalies, and inspect the state.

The code here runs a very simple version of sequence learning, with one
cell per column. The TM is trained with the simple sequence A->B->C->D->E
"""

from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

# Utility routine for printing an SDR in a particular way.
def formatBits(sdr):
  s = ''
  for c in range(sdr.size):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(sdr.dense.flatten()[c])
  s += ' '
  return s

def printStateTM( tm ):
    # Useful for tracing internal states
    print("Active cells     " + formatBits(tm.getActiveCells()))
    print("Winner cells     " + formatBits(tm.getWinnerCells()))
    tm.activateDendrites(True)
    print("Predictive cells " + formatBits(tm.getPredictiveCells()))
    print("Anomaly", tm.anomaly * 100, "%")
    print("")


print("################################################################################")
print(__doc__)
print("################################################################################")
print("")
print("Creating the Temporal Memory")
tm = TM(columnDimensions = (50,),
        cellsPerColumn=1,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=8,
        )
tm.printParameters()

print("""
Creating inputs to feed to the temporal memory. Each input is an SDR
representing the active mini-columns.  Here we create a simple sequence of 5
SDRs representing the sequence A -> B -> C -> D -> E """)
dataset = { inp : SDR( tm.numberOfColumns() ) for inp in "ABCDE" }
dataset['A'].dense[0:10]  = 1   # Input SDR representing "A", corresponding to mini-columns 0-9
dataset['B'].dense[10:20] = 1   # Input SDR representing "B", corresponding to mini-columns 10-19
dataset['C'].dense[20:30] = 1   # Input SDR representing "C", corresponding to mini-columns 20-29
dataset['D'].dense[30:40] = 1   # Input SDR representing "D", corresponding to mini-columns 30-39
dataset['E'].dense[40:50] = 1   # Input SDR representing "E", corresponding to mini-columns 40-49
# Notify the SDR object that we've updated its dense data in-place.
for z in dataset.values():
  z.dense = z.dense
for inp in "ABCDE":
  print("Input:", inp, " Bits:", formatBits( dataset[inp]) )
print("")

print("################################################################################")
print("")
print("""Send this simple sequence to the temporal memory for learning.""")
print("""
The compute method performs one step of learning and/or inference. Note: here
we just perform learning but you can perform prediction/inference and learning
in the same step if you want (online learning).
""")
for inp in "ABCDE": # Send each letter in the sequence in order
  print("Input:", inp)
  activeColumns = dataset[inp]

  print(">>> tm.compute()")
  tm.compute(activeColumns, learn = True)

  printStateTM(tm)

print("""The reset command tells the TM that a sequence just ended and essentially
zeros out all the states. It is not strictly necessary but it's a bit
messier without resets, and the TM learns quicker with resets.
""")
print(">>> tm.reset()")
print("")
tm.reset()


print("################################################################################")
print("")
print("""Send the same sequence of vectors and look at predictions made by
temporal memory.

The following prints out the active cells, predictive cells, active segments and
winner cells.

What you should notice is that the mini-columns where active state is 1
represent the SDR for the current input pattern and the columns where predicted
state is 1 represent the SDR for the next expected pattern.
""")
for inp in "ABCDE":
  print("Input:", inp)
  activeColumns = dataset[inp]

  print(">>> tm.compute()")
  tm.compute(activeColumns, learn = False)

  printStateTM(tm)
