# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, David McDougall
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
Simple program to examine the ScalarEncoder.

For help using this program run:
$ python -m nupic.examples.rf_view_ScalarEncoder --help
"""

from nupic.bindings.encoders import ScalarEncoderParameters, ScalarEncoder
from nupic.bindings.sdr import SDR, Metrics
import numpy as np
import argparse
import matplotlib.pyplot as plt

#
# Gather input from the user.
#
parser = argparse.ArgumentParser(
    description = "Simple program to examine the ScalarEncoder. /\\/\\/\\ " +
    ScalarEncoder.__doc__ + ' /\\/\\/\\ ' + ScalarEncoderParameters.__doc__)

parser.add_argument('--activeBits', type=int, default=0,
                    help=ScalarEncoderParameters.activeBits.__doc__)

parser.add_argument('--minimum', type=float, default=0,
                    help=ScalarEncoderParameters.minimum.__doc__)

parser.add_argument('--maximum', type=float, default=0,
                    help=ScalarEncoderParameters.maximum.__doc__)

parser.add_argument('--periodic', action='store_true', default=False,
                    help=ScalarEncoderParameters.periodic.__doc__)

parser.add_argument('--radius', type=float, default=0,
                    help=ScalarEncoderParameters.radius.__doc__)

parser.add_argument('--resolution', type=float, default=0,
                    help=ScalarEncoderParameters.resolution.__doc__)

parser.add_argument('--size', type=int, default=0,
                    help=ScalarEncoderParameters.size.__doc__)

parser.add_argument('--sparsity', type=float, default=0,
                    help=ScalarEncoderParameters.sparsity.__doc__)

args = parser.parse_args()

#
# Setup the encoder.  First copy the command line arguments into the parameter structure.
#
parameters = ScalarEncoderParameters()
parameters.activeBits = args.activeBits
parameters.clipInput  = False # This script will never encode values outside of the range [min, max]
parameters.minimum    = args.minimum
parameters.maximum    = args.maximum
parameters.periodic   = args.periodic
parameters.radius     = args.radius
parameters.resolution = args.resolution
parameters.size       = args.size
parameters.sparsity   = args.sparsity

enc = ScalarEncoder( parameters )

#
# Run the encoder and measure some statistics about its output.
#
sdrs = []
n_samples = (enc.parameters.maximum - enc.parameters.minimum) / enc.parameters.resolution
n_samples = int(round( 5 * n_samples ))
for i in np.linspace(enc.parameters.minimum, enc.parameters.maximum, n_samples):
  sdrs.append( enc.encode( i ) )

M = Metrics( [enc.size], len(sdrs) + 1 )
for s in sdrs:
    M.addData(s)
print("Statistics:")
print("Encoded %d inputs."%len(sdrs))
print("Output " + str(M))

#
# Plot the Receptive Field of each bit in the encoder.
#
rf = np.zeros([ enc.size, len(sdrs) ], dtype=np.uint8)
for i in range(len(sdrs)):
    rf[ :, i ] = sdrs[i].dense
plt.imshow(rf, interpolation='nearest')
plt.title( "ScalarEncoder Receptive Fields")
plt.ylabel("Cell Number")
plt.xlabel("Input Value")
n_ticks = 11
plt.xticks( np.linspace(0, len(sdrs)-1, n_ticks),
            np.linspace(enc.parameters.minimum, enc.parameters.maximum, n_ticks))
plt.show()
