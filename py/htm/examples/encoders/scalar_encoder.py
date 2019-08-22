# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero Public License version 3 as published by the Free
# Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

import htm.bindings.encoders
ScalarEncoder           = htm.bindings.encoders.ScalarEncoder
ScalarEncoderParameters = htm.bindings.encoders.ScalarEncoderParameters


if __name__ == '__main__':
    """
    Simple program to examine the ScalarEncoder.

    For help using this program run:
    $ python -m htm.examples.encoders.scalar_encoder --help
    """
    import numpy as np
    from htm.bindings.sdr import SDR, Metrics
    import argparse
    import textwrap
    from sys import exit, modules

    #
    # Gather input from the user.
    #
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "Simple program to examine the ScalarEncoder.\n\n" +
                        textwrap.dedent(ScalarEncoder.__doc__ + "\n\n" +
                                        ScalarEncoderParameters.__doc__))
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
    parser.add_argument('--category', action='store_true',
                        help=ScalarEncoderParameters.category.__doc__)
    args = parser.parse_args()

    #
    # Setup the encoder.  First copy the command line arguments into the parameter structure.
    #
    parameters = ScalarEncoderParameters()
    parameters.activeBits = args.activeBits
    parameters.clipInput  = False # This script does not encode values outside of the range [min, max]
    parameters.minimum    = args.minimum
    parameters.maximum    = args.maximum
    parameters.periodic   = args.periodic
    parameters.radius     = args.radius
    parameters.resolution = args.resolution
    parameters.size       = args.size
    parameters.sparsity   = args.sparsity
    parameters.category   = args.category

    # Try initializing the encoder.
    try:
        enc = ScalarEncoder( parameters )
    except RuntimeError as error:
        print("")
        print(error)
        print("")
        parser.print_usage()
        exit()

    #
    # Run the encoder and measure some statistics about its output.
    #
    if args.category:
        n_samples = int(enc.parameters.maximum - enc.parameters.minimum + 1)
    else:
        n_samples = (enc.parameters.maximum - enc.parameters.minimum) / enc.parameters.resolution
        oversample = 2 # Use more samples than needed to avoid aliasing & artifacts.
        n_samples  = int(round( oversample * n_samples ))
    assert n_samples > 0

    sdrs = []
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
    import matplotlib.pyplot as plt
    if 'matplotlib.pyplot' in modules:
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

