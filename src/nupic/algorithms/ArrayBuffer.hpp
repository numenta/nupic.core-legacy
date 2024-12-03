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

/** @file
 *  This header file defines the data structures used for
 *  facilitating the passing of numpy arrays from python
 *  code to C code.
 */

#ifndef NTA_ARRAY_BUFFER_HPP
#define NTA_ARRAY_BUFFER_HPP

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Structure that wraps the essential elements of
// a numpy array object.
typedef struct _NUMPY_ARRAY {
  int nNumDims;
  const int *pnDimensions;
  const int *pnStrides;
  const char *pData;
} NUMPY_ARRAY;

// Bounding box
typedef struct _BBOX {
  int nLeft;
  int nRight;
  int nTop;
  int nBottom;
} BBOX;

// Macros for clipping boxes
#ifndef MIN
#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#endif // MIN
#ifndef MAX
#define MAX(x, y) ((x) <= (y) ? (y) : (x))
#endif // MAX

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // NTA_ARRAY_BUFFER_HPP
