/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

#ifndef NTA_SCANNING_HPP
#define NTA_SCANNING_HPP

// Performs the time-intensive steps of ScanControlNode.getAlpha
void computeAlpha(size_t xstep, size_t ystep, size_t widthS, size_t heightS,
                  size_t imageWidth, size_t imageHeight, size_t xcount,
                  size_t ycount, size_t weightWidth, float sharpness,
                  float *data, float *values, float *counts, float *weights) {
  size_t y0, y1, x0, x1, i, j, m, n;
  float coefficient = 0, minval = 0, maxval = 0;
  float *d, *v, *c, *w;

  if (sharpness < 1) {
    // Calculate coefficient for sigmoid, used to scale values in range [0, 1]
    // (If sharpness is 1, the results are simply thresholded)
    coefficient = -1 / (1 - sharpness) + 1;
    if (coefficient != 0) {
      minval = 1 / (1 + exp(coefficient * (-0.5f)));
      maxval = 1 / (1 + exp(coefficient * (0.5f)));
    }
  }

  // Sum Gaussian-modulated scan results
  // For each window position (each entry in data), increment both "values" and
  // "counts" for all the pixels in the window
  // But they are not incremented evenly; they are multiplied by a Gaussian
  // (the weights matrix)
  d = data;
  for (i = 0; i < ycount; i++) {
    y0 = i * ystep;
    i == ycount - 1 ? y1 = imageHeight : y1 = y0 + heightS;
    for (j = 0; j < xcount; j++) {
      x0 = j * xstep;
      j == xcount - 1 ? x1 = imageWidth : x1 = x0 + widthS;
      for (m = 0; m < (y1 - y0); m++) {
        v = values + (m + y0) * imageWidth + x0;
        c = counts + (m + y0) * imageWidth + x0;
        w = weights + m * weightWidth;
        for (n = x1 - x0; n > 0; n--, v++, c++, w++) {
          *v += *d * *w;
          *c += *w;
        }
      }
      d++;
    }
  }

  // Post-process the results by normalizing and then applying sigmoid or
  // threshold
  v = values;
  c = counts;
  for (i = imageWidth * imageHeight; i > 0; i--, v++, c++) {
    // Normalize
    *v /= *c;
    if (sharpness == 1) {
      // Simple threshold
      *v >= 0.5 ? *v = 1 : *v = 0;
    } else if (coefficient != 0) {
      // Sigmoid (coefficient was calculated from value of "sharpness")
      *v = (1 / (1 + exp(coefficient * (*v - 0.5f))) - minval) /
           (maxval - minval);
    }
  }
}

#endif // NTA_SCANNING_HPP
