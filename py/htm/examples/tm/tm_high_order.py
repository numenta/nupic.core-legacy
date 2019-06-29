# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc.
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
A tutorial that shows some features of the Temporal Memory.

This program demonstrates some basic properties of the
Temporal Memory, in particular how it handles high-order sequences.
"""

import numpy as np
import random
random.seed(1)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM

print("--------------------------------------------------")
print(__doc__)
print("--------------------------------------------------")
print("")
print("Creating the Temporal Memory")
tm = TM(
  columnDimensions = (2048,),
  cellsPerColumn=8,
  initialPermanence=0.21,
  connectedPermanence=0.3,
  minThreshold=15,
  maxNewSynapseCount=40,
  permanenceIncrement=0.1,
  permanenceDecrement=0.1,
  activationThreshold=15,
  predictedSegmentDecrement=0.01,
  )
tm.printParameters()

print("""
We will create a sparse representation of characters A, B, C, D, X, and Y.
In this particular example we manually construct them, but usually you would
use the spatial pooler to build these.""")
sparsity   = 0.02
sparseCols = int(tm.numberOfColumns() * sparsity)
dataset    = {inp : SDR( tm.numberOfColumns() ) for inp in "ABCDXY"}
for i, inp in enumerate("ABCDXY"):
  dataset[inp].dense[ i * sparseCols : (i + 1) * sparseCols ] = 1
  dataset[inp].dense = dataset[inp].dense # This line notifies the SDR that it's dense data has changed in-place.
  print("Input", inp, "is bits at indices: [",  i * sparseCols, '-', (i + 1) * sparseCols, ')')

seq1 = "ABCD"
seq2 = "XBCY"
seqT = "ABCDXY"


def trainTM(sequence, iterations, noiseLevel):
  """
  Trains the TM with given sequence for a given number of time steps and level
  of input corruption

  Argument sequence   (string) Sequence of input characters.
  Argument iterations (int)    Number of time TM will be presented with sequence.
  Argument noiseLevel (float)  Amount of noise to be applied on the characters in the sequence.

  Returns x, y
      x is list of timestamps / step numbers
      y is list of prediction accuracy at each step
  """
  ts = 0  
  x = []
  y = []
  for t in range(iterations):
    tm.reset()
    for inp in sequence:
      v = SDR(dataset[inp]).addNoise( noiseLevel )
      tm.compute( v, learn=True)
      x.append(ts)
      y.append( 1 - tm.anomaly )
      ts += 1
  return x, y


def showPredictions():
  """
  Shows predictions of the TM when presented with the characters A, B, C, D, X, and
  Y without any contextual information, that is, not embedded within a sequence.
  """
  for inp in sorted(dataset.keys()):
    print("--- " + inp + " ---")
    sdr = dataset[inp]
    tm.reset()
    tm.compute( sdr, learn=False)
    tm.activateDendrites(learn=False)
    activeColumnsIndices   = [tm.columnForCell(i) for i in tm.getActiveCells().sparse]
    predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells().sparse]
    print("Active cols: " + str(sorted(set(activeColumnsIndices))))
    print("Predicted cols: " + str(sorted(set(predictedColumnIndices))))
    print("")


print("")
print("--------------------------------------------------")
print("Part 1. We present the sequence ABCD to the TM. The TM will eventually")
print("will learn the sequence and predict the upcoming characters. This can be")
print("measured by the prediction accuracy in Fig 1.")
print("N.B. In-between sequences the prediction accuracy is 0.0 as the TM does not")
print("output any prediction.")
print("--------------------------------------------------")
print("")

x, y = trainTM(seq1, iterations=10, noiseLevel=0.0)

plt.ylim([-0.1,1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 1: TM learns sequence ABCD")
plt.savefig("figure_1")
plt.close()

print("")
print("--------------------------------------------------")
print("Once the TM has learned the sequence ABCD, we will present the individual")
print("characters to the TM to know its prediction. The TM outputs the columns")
print("that become active upon the presentation of a particular character as well")
print("as the columns predicted in the next time step. Here, you should see that")
print("A predicts B, B predicts C, C predicts D, and D does not output any")
print("prediction.")
print("N.B. Here, we are presenting individual characters, that is, a character")
print("deprived of context in a sequence. There is no prediction for characters")
print("X and Y as we have not presented them to the TM in any sequence.")
print("--------------------------------------------------")
print("")

showPredictions()

print("")
print("--------------------------------------------------")
print("Part 2. We now present the sequence XBCY to the TM. As expected, the accuracy will")
print("drop until the TM learns the new sequence (Fig 2). What would be the prediction of")
print("the TM if presented with the sequence BC? This would depend on what character")
print("anteceding B. This is an important feature of high-order sequences.")
print("--------------------------------------------------")
print("")

x, y = trainTM(seq2, iterations=10, noiseLevel=0.0)

# In this figure you can see how the TM starts making good predictions for particular
# characters (spikes in the plot). Then, it will get half of its predictions right, which
# correspond to the times in which is presented with character C. After some time, it
# will learn correctly the sequence XBCY, and predict its characters accordingly.
plt.ylim([-0.1,1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 2: TM learns new sequence XBCY")
plt.savefig("figure_2")
plt.close()

print("")
print("--------------------------------------------------")
print("We will present again each of the characters individually to the TM, that is,")
print("not within any of the two sequences. When presented with character A the TM")
print("predicts B, B predicts C, but this time C outputs a simultaneous prediction of")
print("both D and Y. In order to disambiguate, the TM would require to know if the")
print("preceding characters were AB or XB. When presented with character X the TM")
print("predicts B, whereas Y and D yield no prediction.")
print("--------------------------------------------------")
print("")

showPredictions()

print("")
print("--------------------------------------------------")
print("""Part 3. Now we will present noisy inputs to the TM. We would like to see how the
TM responds to the presence of noise and how it recovers from it. We will add
noise to the sequence XBCY by corrupting 30% of the bits in the SDR encoding
each character. We would expect to see a decrease in prediction accuracy as the
TM is unable to learn the random noise in the input (Fig 3). However, this
decrease is not significant.""")
print("--------------------------------------------------")
print("")

x, y = trainTM(seq2, iterations=50, noiseLevel=0.3)

plt.ylim([-0.1,1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 3: Accuracy in TM with 30% noise in input")
plt.savefig("figure_3")
plt.close()

print("")
print("--------------------------------------------------")
print("Let's have a look again at the output of the TM when presented with noisy")
print("input (30%). Here, the noise is low enough that the TM is not affected by it,")
print("which would be the case if we saw 'noisy' columns being predicted when")
print("presented with individual characters. Thus, we could say that the TM exhibits")
print("resilience to noise in its input.")
print("--------------------------------------------------")
print("")

showPredictions()

print("")
print("--------------------------------------------------")
print("Now, we will increase the noise to 60% of the bits in the characters.")
print("As expected, the predictive accuracy decreases (Fig 4) and 'noisy' columns are")
print("predicted by the TM.")
print("--------------------------------------------------")
print("")

x, y = trainTM(seq2, iterations=50, noiseLevel=0.6)

plt.ylim([-0.1,1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 4: Accuracy in TM with 60% noise in input")
plt.savefig("figure_4")
plt.close()

# Will the TM be able to forget the 'noisy' columns learned in the previous step?
# We will present the TM with the original sequence XBCY so it forgets the 'noisy'.
# columns.

print("")
print("--------------------------------------------------")
print("""After presenting the uncorrupted sequence XBCY to the TM, we would expect to see
the predicted noisy columns from the previous step disappear and the prediction
accuracy return to normal. (Fig 5.)""")
print("--------------------------------------------------")
print("")

x, y = trainTM(seq2, iterations=10, noiseLevel=0.0)

plt.ylim([-0.1,1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 5: When noise is suspended, accuracy is restored")
plt.savefig("figure_5")
plt.close()

print("")
print("--------------------------------------------------")
print("""Part 4. We will present both sequences ABCD and XBCY randomly to the TM.
Here, we might observe simultaneous predictions occurring when the TM is
presented with characters D, Y, and C. For this purpose we will use a
blank TM
NB. Here we will not reset the TM after presenting each sequence with the
purpose of making the TM learn different predictions for D and Y.""")
print("--------------------------------------------------")
print("")

tm = TM(columnDimensions = (2048,),
  cellsPerColumn=8,
  initialPermanence=0.21,
  connectedPermanence=0.3,
  minThreshold=15,
  maxNewSynapseCount=40,
  permanenceIncrement=0.1,
  permanenceDecrement=0.1,
  activationThreshold=15,
  predictedSegmentDecrement=0.01,
  )

for t in range(75):
  seq = random.choice([ seq1, seq2 ])
  for inp in seq:
    tm.compute( dataset[inp], learn=True)

print("")
print("--------------------------------------------------")
print("We now have a look at the output of the TM when presented with the individual")
print("characters A, B, C, D, X, and Y. We should observe simultaneous predictions when")
print("presented with character D (predicting A and X), character Y (predicting A and X),")
print("and when presented with character C (predicting D and Y).")
print("--------------------------------------------------")
print("")

showPredictions()

print("")
print("All images generated by this script are saved in your current working directory.")
print("")
