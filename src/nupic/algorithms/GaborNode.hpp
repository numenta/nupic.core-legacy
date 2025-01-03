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
 *  This header file defines the API for performing efficient
 *  Gabor processing.
 */

#ifndef NTA_GABOR_NODE_HPP
#define NTA_GABOR_NODE_HPP

#include <nupic/utils/Log.hpp>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include "ArrayBuffer.hpp"
#include <nupic/types/Types.h>

// Number of bits that our gabor filter coefficients are
// shifted (to the left) for scaling purposes, when
// using GABOR_METHOD_INTEGER8
#define GABOR_SCALING_SHIFT 12

// For reasons of efficiency and simplicity, we'll store our
// responses statistics (used for automated normalization) in
// a static buffer of fixed size.  Because of this, we need to
// impose a constraint on the maximum number of filters;
// 64 should be enough for anyone...  :-)
#define MAXNUM_FILTERS 64

// Enumeration that specifies how we handle boundary
// effects
typedef enum _EDGE_MODE {
  EDGE_MODE_CONSTRAINED = 0,
  EDGE_MODE_SWEEPOFF,
  EDGE_MODE__LAST
} EDGE_MODE;

// Enumeration that specifies whether we compute responses
// for "dual" phases (180 degrees shifted)
typedef enum _PHASE_MODE {
  PHASE_MODE_SINGLE = 0,
  PHASE_MODE_DUAL,
  PHASE_MODE__LAST
} PHASE_MODE;

// Enumeration that specifies the method we will use for
// automatically normalizing our filter responses.
typedef enum _NORMALIZE_METHOD {
  NORMALIZE_METHOD_FIXED = 0,
  NORMALIZE_METHOD_MAX,
  NORMALIZE_METHOD_MEAN,
  NORMALIZE_METHOD_MAXPOWER,
  NORMALIZE_METHOD_MEANPOWER,
  NORMALIZE_METHOD__LAST
} NORMALIZE_METHOD;

// Enumeration that specifies whether our chosen normalization
// statistic will be computed over all scales and/or orientations.
typedef enum _NORMALIZE_MODE {
  NORMALIZE_MODE_GLOBAL = 0,
  NORMALIZE_MODE_PERORIENT,
  NORMALIZE_MODE__LAST
} NORMALIZE_MODE;

typedef enum _PHASENORM_MODE {
  PHASENORM_MODE_COMBO = 0, // phases are normalized together
  PHASENORM_MODE_INDIV,     // phases are normalized individually
  PHASENORM_MODE__LAST
} PHASENORM_MODE;

// Enumeration that specifies the method by which we will
// perform final post-processing upon the normalized responses.
typedef enum _POSTPROC_METHOD {
  POSTPROC_METHOD_RAW = 0,
  POSTPROC_METHOD_SIGMOID,
  POSTPROC_METHOD_THRESHOLD,
  POSTPROC_METHOD__LAST
} POSTPROC_METHOD;

// FUNCTION: gaborCompute()
// PURPOSE: Implements efficient Gabor filtering.
NTA_EXPORT
int gaborCompute(const NUMPY_ARRAY *psGaborBank, const NUMPY_ARRAY *psInput,
                 const NUMPY_ARRAY *psAlpha, const NUMPY_ARRAY *psBBox,
                 const NUMPY_ARRAY *psImageBox, const NUMPY_ARRAY *psOutput,
                 float fGainConstant, EDGE_MODE eEdgeMode,
                 float fOffImageFillValue, PHASE_MODE ePhaseMode,
                 NORMALIZE_METHOD eNormalizeMethod,
                 NORMALIZE_MODE eNormalizeMode, PHASENORM_MODE ePhaseNormMode,
                 POSTPROC_METHOD ePostProcMethod, float fPostProcSlope,
                 float fPostProcMidpoint, float fPostProcMin,
                 float fPostProcMax, const NUMPY_ARRAY *psBufferIn,
                 const NUMPY_ARRAY *psBufferOut,
                 const NUMPY_ARRAY *psPostProcLUT, float fPostProcScalar);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // NTA_GABOR_NODE_HPP
