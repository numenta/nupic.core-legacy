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

""" A simple program that demonstrates the workings of the spatial pooler. """

from htm.bindings.sdr import SDR
from htm.algorithms import SpatialPooler as SP

# Create the Spatial Pooler, and the SDR data structures needed to work with it.
inputSDR  = SDR( dimensions = (32, 32) )
activeSDR = SDR( dimensions = (64, 64) )
sp = SP(inputDimensions  = inputSDR.dimensions,
        columnDimensions = activeSDR.dimensions,
        localAreaDensity = 0.02,
        globalInhibition = True,
        seed             = 1,
        synPermActiveInc   = 0.01,
        synPermInactiveDec = 0.008)

def run():
    print("Running the Spatial Pooler ...")
    print("")
    sp.compute(inputSDR, True, activeSDR)
    print("Active Outputs " + str(activeSDR))
    print("")


# Lesson 1, Trying random inputs.
print("")
print("Hello Spatial Pooler.")
print("")
print("")
print("Lesson 1) Different inputs give different outputs.")
print("    Will now generate 3 random Sparse Distributed Representations (SDRs) and run each")
print("    through the spatial pooler.  Observe that the output activity is different each time.")
print("")

for i in range(3):
    print("----------------------------------------------------------------------")
    inputSDR.randomize( .02 )
    print("Random Input " + str(inputSDR))
    print("")
    run()


# Lesson 2, Trying identical inputs.
print("=" * 70)
print("")
print("")
print("Lesson 2) Identical inputs give identical outputs.")
print("    The input SDR is the same as was used for the previous run of the spatial pooler.")
print("")
print("Input " + str(inputSDR))
print("")
run()


# Lesson 3, Trying similar inputs.
print("=" * 70)
print("")
print("")
print("Lesson 3) Similar inputs give similar outputs.")
print("          Now we are changing the input SDR slightly.")
print("          We change a small percentage of 1s to 0s and 0s to 1s.")
print("          The resulting SDRs are similar, but not identical to the original SDR")
print("")

print("Adding 10% noise to the input SDR from the previous run.")
inputSDR.addNoise(0.10)
print("Input " + str(inputSDR))
print("")
run()
print("Notice how the output SDR hardly changed at all.")
print("")
print("")
print("Adding another 20% noise to the input SDR.")
inputSDR.addNoise(0.2)
print("Input " + str(inputSDR))
print("")
run()
print("The output SDR now differs considerably from that of the previous output.")
print("However there are still similarities between the outputs.")
print("")
print("End.")
