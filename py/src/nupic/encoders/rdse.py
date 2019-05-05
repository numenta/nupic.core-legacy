# ------------------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
#
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

import nupic.bindings.encoders
RDSE            = nupic.bindings.encoders.RDSE
RDSE_Parameters = nupic.bindings.encoders.RDSE_Parameters

__all__ = ['RDSE', 'RDSE_Parameters']

if __name__ == '__main__':
    """
    This module is also a program to examine the Random Distributed Scalar
    Encoder (RDSE).

    For help using this program run:
    $ python -m nupic.encoders.scalar_encoder --help
    """
    import numpy as np
    from nupic.bindings.sdr import SDR, Metrics
    import argparse
    import textwrap
    import matplotlib.pyplot as plt
    from sys import exit

    #
    # Gather input from the user.
    #
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = "Examine the Random Distributed Scalar Encoder (RDSE).\n\n" +
                        textwrap.dedent(RDSE.__doc__ + "\n\n" +
                                        RDSE_Parameters.__doc__))
    parser.add_argument('--activeBits', type=int, default=0,
                        help=RDSE_Parameters.activeBits.__doc__)
    parser.add_argument('--minimum', type=float, required=True,
                        help="TODO")
    parser.add_argument('--maximum', type=float, required=True,
                        help="TODO")
    parser.add_argument('--radius', type=float, default=0,
                        help=RDSE_Parameters.radius.__doc__)
    parser.add_argument('--resolution', type=float, default=0,
                        help=RDSE_Parameters.resolution.__doc__)
    parser.add_argument('--size', type=int, required=True,
                        help=RDSE_Parameters.size.__doc__)
    parser.add_argument('--sparsity', type=float, default=0,
                        help=RDSE_Parameters.sparsity.__doc__)
    parser.add_argument('--category', action='store_true',
                        help="Assume the inputs are enumerated categories.\n" +
                             "Only encodes integers, forces radius=1.")
    args = parser.parse_args()

    #
    # Setup the encoder.  First copy the command line arguments into the parameter structure.
    #
    parameters = RDSE_Parameters()
    parameters.activeBits = args.activeBits
    parameters.radius     = args.radius
    parameters.resolution = args.resolution
    parameters.size       = args.size
    parameters.sparsity   = args.sparsity

    if args.category:
        if parameters.radius == 0:
            parameters.radius = 1
            print("Category encoder requested, setting radius to 1.")
        elif parameters.radius != 1:
            print("Category encoder requested, but radius is not equal to 1!")
            exit()

    # Try initializing the encoder.
    try:
        enc = RDSE( parameters )
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
        n_samples = int(args.maximum - args.minimum + 1)
    else:
        n_samples = (args.maximum - args.minimum) / enc.parameters.resolution
        n_samples = int(round( 5 * n_samples ))
    sdrs = []
    for i in np.linspace(args.minimum, args.maximum, n_samples):
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
    plt.title( "RDSE Receptive Fields")
    plt.ylabel("Cell Number")
    plt.xlabel("Input Value")
    n_ticks = 11
    plt.xticks( np.linspace(0, len(sdrs)-1, n_ticks),
                np.linspace(args.minimum, args.maximum, n_ticks))
    plt.show()
