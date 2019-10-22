# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2018, Numenta, Inc.
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

"""
A simple tutorial that shows some features of the Spatial Pooler.

The following program has the purpose of presenting some
basic properties of the Spatial Pooler. It reproduces Figs.
5, 7 and 9 from this paper: http://arxiv.org/abs/1505.02142
To learn more about the Spatial Pooler have a look at BAMI:
http://numenta.com/biological-and-machine-intelligence/
or at its class reference in the NuPIC documentation:
http://numenta.org/docs/nupic/classnupic_1_1research_1_1spatial__pooler_1_1_spatial_pooler.html
The purpose of the Spatial Pooler is to create a sparse representation
of its inputs in such a way that similar inputs will be mapped to similar
sparse representations. Thus, the Spatial Pooler should exhibit some resilience
to noise in its input.
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from htm.bindings.sdr import SDR
from htm.algorithms import SpatialPooler as SP


def percentOverlap(x1, x2):
  """
  Computes the percentage of overlap between SDRs x1 and x2.

  Argument x1 is an SDR
  Argument x2 is an SDR

  Returns percentOverlap (float) percentage overlap between x1 and x2
  """
  minX1X2 = min(x1.getSum(), x2.getSum())
  percentOverlap = 0
  if minX1X2 > 0:
    percentOverlap = float( x1.getOverlap( x2 )) / minX1X2
  return percentOverlap


def corruptSDR(sdr, noiseLevel):
  """
  Corrupts a binary vector by inverting noiseLevel percent of its bits.

  Argument vector     (array) binary vector to be corrupted
  Argument noiseLevel (float) amount of noise to be applied on the vector.
  """
  vector = sdr.flatten().dense
  for i in range(sdr.size):
    rnd = random.random()
    if rnd < noiseLevel:
      if vector[i] == 1:
        vector[i] = 0
      else:
        vector[i] = 1
  sdr.dense = vector


inputSDR  = SDR( dimensions = (1000,1) ).randomize( .50 )
outputSDR = SDR( dimensions = (2048,1) )

sp = SP(inputSDR.dimensions,
        outputSDR.dimensions,
        potentialRadius = int(0.5 * inputSDR.size),
        localAreaDensity = .02,
        globalInhibition = True,
        seed = 0,
        synPermActiveInc = 0.01,
        synPermInactiveDec = 0.008)

# Part 1:
# -------
# A column connects to a subset of the input vector (specified
# by both the potentialRadius and potentialPct). The overlap score
# for a column is the number of connections to the input that become
# active when presented with a vector. When learning is 'on' in the SP,
# the active connections are reinforced, whereas those inactive are
# depressed (according to parameters synPermActiveInc and synPermInactiveDec.
# In order for the SP to create a sparse representation of the input, it
# will select a small percentage (usually 2%) of its most active columns,
# ie. columns with the largest overlap score.
# In this first part, we will create a histogram showing the overlap scores
# of the Spatial Pooler (SP) after feeding it with a random binary
# input. As well, the histogram will show the scores of those columns
# that are chosen to build the sparse representation of the input.

overlaps = sp.compute(inputSDR, False, outputSDR)
activeColsScores = []
for i in outputSDR.sparse:
  activeColsScores.append(overlaps[i])

print("")
print("---------------------------------")
print("Figure 1 shows a histogram of the overlap scores")
print("from all the columns in the spatial pooler, as well as the")
print("overlap scores of those columns that were selected to build a")
print("sparse representation of the input (shown in green).")
print("The SP chooses 2% of the columns with the largest overlap score")
print("to make such sparse representation.")
print("---------------------------------")
print("")

bins = np.linspace(min(overlaps), max(overlaps), 28)
plt.hist(overlaps, bins, alpha=0.5, label="All cols")
plt.hist(activeColsScores, bins, alpha=0.5, label="Active cols")
plt.legend(loc="upper right")
plt.xlabel("Overlap scores")
plt.ylabel("Frequency")
plt.title("Figure 1: Column overlap of a SP with random input.")
plt.savefig("figure_1")
plt.close()

# Part 2a:
# -------
# The input overlap between two binary vectors is defined as their dot product.
# In order to normalize this value we divide by the minimum number of active
# inputs (in either vector). This means we are considering the sparser vector as
# reference. Two identical binary vectors will have an input overlap of 1,
# whereas two completely different vectors (one is the logical NOT of the other)
# will yield an overlap of 0. In this section we will see how the input overlap
# of two binary vectors decrease as we add noise to one of them.

inputX1 = SDR( inputSDR.size ).randomize( .50 )
inputX2 = SDR( inputSDR.size )
outputX1 = SDR( outputSDR.size )
outputX2 = SDR( outputSDR.size )

x = []
y = []
for noiseLevel in np.arange(0, 1.1, 0.1):
  inputX2.setSDR( inputX1 )
  corruptSDR(inputX2, noiseLevel)
  x.append(noiseLevel)
  y.append(percentOverlap(inputX1, inputX2))

print("")
print("---------------------------------")
print("Figure 2 shows the input overlap between 2 identical binary vectors in")
print("function of the noise applied to one of them.")
print("0 noise level means that the vector remains the same, whereas")
print("1 means that the vector is the logical negation of the original vector. ")
print("The relationship between overlap and noise level is practically linear ")
print("and monotonically decreasing.")
print("---------------------------------")
print("")

plt.plot(x, y)
plt.xlabel("Noise level")
plt.ylabel("Input overlap")
plt.title("Figure 2: Input overlap between 2 identical vectors in function of noiseLevel.")
plt.savefig("figure_2")
plt.close()

# Part 2b:
# -------
# The output overlap between two binary input vectors is the overlap of the
# columns that become active once they are fed to the SP. In this part we
# turn learning off, and observe the output of the SP as we input two binary
# input vectors with varying level of noise.
# Starting from two identical vectors (that yield the same active columns)
# we would expect that as we add noise to one of them their output overlap
# decreases.
# In this part we will show how the output overlap behaves in function of the
# input overlap between two vectors.
# Even with an untrained spatial pooler, we see some noise resilience.
# Note that due to the non-linear properties of high dimensional SDRs, overlaps
# greater than 10 bits, or 25% in this example, are considered significant.

x = []
y = []
for noiseLevel in np.arange(0, 1.1, 0.1):
  inputX2.setSDR( inputX1 )
  corruptSDR(inputX2, noiseLevel)

  sp.compute(inputX1, False, outputX1)
  sp.compute(inputX2, False, outputX2)

  x.append(percentOverlap(inputX1, inputX2))
  y.append(percentOverlap(outputX1, outputX2))

print("")
print("---------------------------------")
print("Figure 3 shows the output overlap between two sparse representations")
print("in function of their input overlap. Starting from two identical binary ")
print("vectors (which yield the same active columns) we add noise two one of ")
print("them, feed it to the SP, and estimate the output overlap between the two")
print("representations in terms of the common active columns between them.")
print("As expected, as the input overlap decreases, so does the output overlap.")
print("---------------------------------")
print("")

plt.plot(x, y)
plt.xlabel("Input overlap")
plt.ylabel("Output overlap")
plt.title("Figure 3: Output overlap in function of input overlap in a SP "
          "without training")
plt.savefig("figure_3")
plt.close()

# Part 3:
# -------
# After training, a SP can become less sensitive to noise. For this purpose, we
# train the SP by turning learning on, and by exposing it to a variety of random
# binary vectors. We will expose the SP to a repetition of input patterns in
# order to make it learn and distinguish them once learning is over. This will
# result in robustness to noise in the inputs. In this section we will reproduce
# the plot in the last section after the SP has learned a series of inputs. Here
# we will see how the SP exhibits increased resilience to noise after learning.

# We will present 10 random vectors to the SP, and repeat this 30 times.
# Later you can try changing the number of times we do this to see how it
# changes the last plot. Then, you could also modify the number of examples to
# see how the SP behaves. Is there a relationship between the number of examples
# and the number of times that we expose them to the SP?

numExamples   = 10
inputVectors  = [SDR(inputSDR.size).randomize( .50 ) for _ in range(numExamples)]
outputColumns = [SDR(outputSDR.size) for _ in range(numExamples)]

# This is the number of times that we will present the input vectors to the SP
epochs = 30

overlapsUntrained = overlaps

for _ in range(epochs):
  for i in range(numExamples):
    # Feed the examples to the SP
    overlaps = sp.compute(inputVectors[i], True, outputColumns[i])

print("")
print("---------------------------------")
print("Figure 4a shows the sorted overlap scores of all columns in a spatial")
print("pooler with random input, before and after learning. The top 2% of ")
print("columns with the largest overlap scores, comprising the active columns ")
print("of the output sparse representation, are highlighted in green.")
print("---------------------------------")
print("")

plt.plot(sorted(overlapsUntrained)[::-1], label="Before learning")
plt.plot(sorted(overlaps)[::-1], label="After learning")
plt.axvspan(0, len(activeColsScores), facecolor="g", alpha=0.3, label="Active columns")
plt.legend(loc="upper right")
plt.xlabel("Columns")
plt.ylabel("Overlap scores")
plt.title("Figure 4a: Sorted column overlaps of a SP with random input.")
plt.savefig("figure_4a")
plt.close()


inputCorrupted  = SDR( inputSDR.dimensions )
outputCorrupted = SDR( outputSDR.dimensions )

x = []
y = []
# We will repeat the experiment in the last section for only one input vector
# in the set of input vectors
for noiseLevel in np.arange(0, 1.1, 0.1):
  inputCorrupted.setSDR( inputVectors[0] )
  corruptSDR(inputCorrupted, noiseLevel)

  sp.compute(inputVectors[0], False, outputColumns[0])
  sp.compute(inputCorrupted, False, outputCorrupted)

  x.append(percentOverlap(inputVectors[0], inputCorrupted))
  y.append(percentOverlap(outputColumns[0], outputCorrupted))

print("")
print("---------------------------------")
print("How robust is the SP to noise after learning?")
print("Figure 4 shows again the output overlap between two binary vectors in ")
print("function of their input overlap. After training, the SP exhibits more ")
print("robustness to noise in its input, resulting in a -almost- sigmoid curve.")
print("This implies that even if a previous input is presented again with a ")
print("certain amount of noise its sparse representation still resembles its ")
print("original.")
print("---------------------------------")
print("")

plt.plot(x, y)
plt.xlabel("Input overlap")
plt.ylabel("Output overlap")
plt.title("Figure 4: Output overlap in function of input overlap in a SP after "
          "training")
plt.savefig("figure_4")
plt.close()

print("")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(" All images generated by this script will be saved")
print(" in your current working directory.")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print("")
