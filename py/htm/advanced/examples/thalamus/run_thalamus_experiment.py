# ----------------------------------------------------------------------
# HTM Community Edition of NuPIC
# Copyright (C) 2019, Numenta, Inc. 
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
# along with this program.    If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import ndimage as ndi

from htm.advanced.algorithms.thalamus import Thalamus
from thalamus_utils import  createLocationEncoder, encodeLocation, getUnionLocations, trainThalamusLocations
from skimage.filters import gabor_kernel
from htm.bindings.sdr import SDR


# TODO: implement an overlap matrix to show that the location codes are overlapping

# TODO: implement feature filtering. Can have R, G, B ganglion inputs segregated
# into different relay cells. Only the R ones will burst, the rest are tonic.

# TODO: change color scheme so that grey is nothing and blue is tonic.

# TODO: implement a filling in mechanism.

# TODO: fan-out from ganglion cells to relay cells are not currently implemented.
# We should have ganglion cells also on the dendrites.


def loadImage(t, filename="cajal.jpg"):
    """
    Load the given gray scale image. Threshold it to black and white and crop it
    to be the dimensions of the FF input for the thalamus.    Return a binary numpy
    matrix where 1 corresponds to black, and 0 corresponds to white.
    """
    image = Image.open(filename).convert("1")
    image.load()
    box = (0, 0, t.inputWidth, t.inputHeight)
    image = image.crop(box)

    # Here a will be a binary numpy array where True is white. Convert to floating
    # point numpy array where white is 0.0
    a = np.asarray(image)
    im = np.ones((t.inputWidth, t.inputHeight))
    im[a] = 0

    return im


def plotActivity(activity, filename, title="", vmin=0.0, vmax=2.0, cmap="Greys"):
    plt.imshow(activity, vmin=vmin, vmax=vmax, origin="upper", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.savefig(os.path.join("images", filename))
    plt.close()


def locationsTest():
    """
    Test with square and blocky A
    """
    print('Running locations test experiment')
    print("Initializing thalamus")
    t = Thalamus()

    encoder = createLocationEncoder(t)

    trainThalamusLocations(t, encoder)

    output = SDR(encoder.getWidth())
    ff = np.zeros((32, 32))
    for x in range(10,20):
        print("Testing with x=", x)
        ff[:] = 0
        ff[10:20, 10:20] = 1
        plotActivity(ff, "square_ff_input.jpg", title="Feed forward input")
        ffOutput = inferThalamus(t, encodeLocation(encoder, x, x, output), ff)
        plotActivity(ffOutput, "square_relay_output_" + str(x) + ".jpg", title="Relay cell activity", cmap="coolwarm")

    # Show attention with an A
    ff = np.zeros((32, 32))
    for x in range(10,20):
        print("Testing with x=", x)
        ff[:] = 0
        ff[10, 10:20] = 1
        ff[15, 10:20] = 1
        ff[10:20, 10] = 1
        ff[10:20, 20] = 1
        plotActivity(ff, "A_ff_input.jpg", title="Feed forward input")
        ffOutput = inferThalamus(t, encodeLocation(encoder, x, x, output), ff)
        plotActivity(t.burstReadyCells, "relay_burstReady_" + str(x) + ".jpg", title="Burst-ready cells (x,y)=({},{})".format(x, x))
        plotActivity(ffOutput, "A_relay_output_" + str(x) + ".jpg", title="Relay cell activity", cmap="coolwarm")


def largeThalamus(w=250):
    """
    Test a moving bursting area.
    """
    print('Running large thalamus experiment')
    print("Initializing thalamus")
    t = Thalamus(
        trnCellShape=(w, w),
        relayCellShape=(w, w),
        inputShape=(w, w),
        l6CellCount=128*128,
        trnThreshold=15,
    )

    encoder = createLocationEncoder(t, w=17)
    trainThalamusLocations(t, encoder)

    print("Loading image")
    ff = loadImage(t)
    plotActivity(ff, "cajal_input.jpg", title="Feed forward input")

    for x in range(w//2-60,w//2+60,40):
        print("Testing with x=", x)
        ff = loadImage(t)
        l6Code = getUnionLocations(encoder, x, x, 20)
        print("Num active cells in L6 union:", len(l6Code.sparse),"out of", t.l6CellCount)
        ffOutput = inferThalamus(t, l6Code, ff)
        plotActivity(t.burstReadyCells, "relay_burstReady_" + str(x) + ".jpg", title="Burst-ready cells (x,y)=({},{})".format(x, x))
        plotActivity(ffOutput, "cajal_relay_output_" + str(x) + ".jpg", title="Relay cell activity", cmap="coolwarm")

    # The eye
    x=150
    y=110
    print("Testing with x,y=", x, y)
    ff = loadImage(t)
    l6Code = getUnionLocations(encoder, x, y, 20)
    print("Num active cells in L6 union:", len(l6Code.sparse),"out of", t.l6CellCount)
    ffOutput = inferThalamus(t, l6Code, ff)
    plotActivity(t.burstReadyCells, "relay_burstReady_eye.jpg", title="Burst-ready cells (x,y)=({},{})".format(x, y), )
    plotActivity(ffOutput, "cajal_relay_output_eye.jpg", title="Filtered activity", cmap="Greys")

    # The ear
    x=25
    y=150
    print("Testing with x,y=", x, y)
    ff = loadImage(t)
    l6Code = getUnionLocations(encoder, x, y, 20)
    print("Num active cells in L6 union:", len(l6Code.sparse),"out of", t.l6CellCount)
    ffOutput = inferThalamus(t, l6Code, ff)
    plotActivity(t.burstReadyCells, "relay_burstReady_ear.jpg", title="Burst-ready cells (x,y)=({},{})".format(x, y), )
    plotActivity(ffOutput, "cajal_relay_output_ear.jpg", title="Filtered activity", cmap="Greys")

    return t


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 + ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)


def filtered(w=250):
    """
    In this example we filter the image into several channels using gabor filters. L6 activity is used to select
    one of those channels. Only activity selected by those channels burst.
    """
    
    print('Running filtered experiment')
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    print("Initializing thalamus")

    t = Thalamus(
        trnCellShape=(w, w),
        relayCellShape=(w, w),
        inputShape=(w, w),
        l6CellCount=128*128,
        trnThreshold=15,
        )

    ff = loadImage(t)

    for i,k in enumerate(kernels):
        plotActivity(k, "kernel"+str(i)+".jpg", "Filter kernel", vmax=k.max(), vmin=k.min())
        filtered0 = power(ff, k)
        ft = np.zeros((w, w))
        ft[filtered0 > filtered0.mean() + filtered0.std()] = 1.0
        plotActivity(ft, "filtered"+str(i)+".jpg", "Filtered image", vmax=1.0)

    encoder = createLocationEncoder(t, w=17)
    trainThalamusLocations(t, encoder)

    filtered0 = power(ff, kernels[3])
    ft = np.zeros((w, w))
    ft[filtered0 > filtered0.mean() + filtered0.std()] = 1.0

    # Get a salt and pepper burst ready image
    print("Getting unions")
    l6Code = getUnionLocations(encoder, 125, 125, 150, step=10)
    print("Num active cells in L6 union:", len(l6Code.sparse),"out of", t.l6CellCount)
    ffOutput = inferThalamus(t, l6Code, ft)
    plotActivity(t.burstReadyCells, "relay_burstReady_filtered.jpg", title="Burst-ready cells")
    plotActivity(ffOutput, "cajal_relay_output_filtered.jpg", title="Filtered activity", cmap="Greys")

    # Get a more detailed filtered image
    print("Getting unions")
    l6Code = getUnionLocations(encoder, 125, 125, 150, step=3)
    print("Num active cells in L6 union:", len(l6Code.sparse),"out of", t.l6CellCount)
    ffOutput_all = inferThalamus(t, l6Code, ff)
    ffOutput_filtered = inferThalamus(t, l6Code, ft)
    ffOutput3 = ffOutput_all*0.4 + ffOutput_filtered
    plotActivity(t.burstReadyCells, "relay_burstReady_all.jpg", title="Burst-ready cells")
    plotActivity(ffOutput3, "cajal_relay_output_filtered2.jpg", title="Filtered activity", cmap="Greys")


def inferThalamus(t, l6Input, ffInput):
    """
    Compute the effect of this feed forward input given the specific L6 input.

    :param t: instance of Thalamus
    :param l6Input:
    :param ffInput: a numpy array of 0's and 1's
    :return:
    """
    print("\n-----------")
    t.reset()
    t.deInactivateCells(l6Input)
    ffOutput = t.computeFeedForwardActivity(ffInput)
    # print("L6 input:", l6Input)
    # print("Active TRN cells: ", t.activeTRNCellIndices)
    # print("Burst ready relay cells: ", t.burstReadyCellIndices)
    return ffOutput


if __name__ == '__main__':

    if not os.path.exists('images'):
        os.makedirs('images')
    
    locationsTest()  
    largeThalamus(250)
    filtered(250)
