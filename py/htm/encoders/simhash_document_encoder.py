# ------------------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2016, Numenta, Inc. https://numenta.com
#               2019, Brev Patterson, Lux Rota LLC, https://luxrota.com
#               2019, David McDougall
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero Public License version 3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero Public License for
# more details.
#
# You should have received a copy of the GNU Affero Public License along with
# this program.  If not, see http://www.gnu.org/licenses.
# ------------------------------------------------------------------------------

import htm.bindings.encoders
SimHashDocumentEncoder = htm.bindings.encoders.SimHashDocumentEncoder
SimHashDocumentEncoderParameters = htm.bindings.encoders.SimHashDocumentEncoderParameters

__all__ = ['SimHashDocumentEncoder', 'SimHashDocumentEncoderParameters']


if __name__ == '__main__':
  """
  Simple program to examine the SimHashDocumentEncoder.

  For help using this program run:
  $ python -m htm.encoders.simhash_document_encoder --help
  """
  import argparse
  # import numpy as np
  # import matplotlib.pyplot as plt
  import textwrap
  from htm.bindings.sdr import SDR, Metrics
  from sys import exit

  # Gather input from the user.
  parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Simple program to examine the SimHashDocumentEncoder.\n\n" +
    textwrap.dedent(SimHashDocumentEncoder.__doc__ + "\n\n" +
    SimHashDocumentEncoderParameters.__doc__))
  parser.add_argument('--activeBits', type=int, default=0,
    help=SimHashDocumentEncoderParameters.activeBits.__doc__)
  parser.add_argument('--size', type=int, default=0,
    help=SimHashDocumentEncoderParameters.size.__doc__)
  parser.add_argument('--sparsity', type=float, default=0,
    help=SimHashDocumentEncoderParameters.sparsity.__doc__)
  parser.add_argument('--tokenSimilarity', action='store_true', default=False,
    help=SimHashDocumentEncoderParameters.tokenSimilarity.__doc__)
  args = parser.parse_args()

  # Copy the command line arguments into the parameter structure.
  parameters = SimHashDocumentEncoderParameters()
  parameters.activeBits = args.activeBits
  parameters.size = args.size
  parameters.sparsity = args.sparsity
  parameters.tokenSimilarity = args.tokenSimilarity

  # Try initializing the encoder.
  try:
    encoder = SimHashDocumentEncoder(parameters)
  except RuntimeError as error:
    print("")
    print(error)
    print("")
    parser.print_usage()
    exit()

  # Run the encoder and measure some statistics about its output.
  # @TODO
