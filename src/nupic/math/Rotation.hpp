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
 * Declarations for 2D matrix rotation by 45 degrees.
 */

#ifndef NTA_ROTATION_HPP
#define NTA_ROTATION_HPP

/*
 * Used in GaborFilter
 */

#define cos45 0.70710678118654746f // cos(pi/4) = 1/sqrt(2)

template <typename T> struct Rotation45 {
  typedef size_t size_type;
  typedef T value_type;
  int srow_;
  int scol_;
  size_t offset_;

  inline T round(T x) { return floor(x + 0.5); }

  /**
   * Rotate counter-clockwise by 45 degrees.
   * Fill in pixels in the larger, rotated version of the image.
   */
  inline void rotate(T *original, T *rotated, size_t nrows, size_t ncols,
                     size_t z) {
    offset_ = size_t(T(ncols) * cos45); // Vertical offset
    for (int j = -1 * offset_; j != int(z - offset_); j++) {
      for (int i = 0; i != int(z); i++) {
        // Compute the nearest source pixel for this destination pixel
        // Multiply the destination pixel by the rotation matrix
        srow_ = int(round(cos45 * T(j) + cos45 * T(i)));
        scol_ = int(round(-1 * cos45 * T(j) + cos45 * T(i)));
        if (0 <= srow_ && srow_ < int(nrows) && 0 <= scol_ &&
            scol_ < int(ncols)) {
          // Copy the source pixel to the destination pixel
          rotated[size_t(j + offset_) * z + i] =
              original[srow_ * ncols + scol_];
        }
      }
    }
  }

  /**
   * Rotate clockwise by 45 degrees.
   * Start with the larger, rotated image, and fill in the smaller image
   * of the original size.
   */
  inline void unrotate(T *unrotated, T *rotated, size_t nrows, size_t ncols,
                       size_t z) {
    offset_ = size_t(T(ncols) * cos45); // Vertical offset
    for (size_t j = 0; j != nrows; j++) {
      for (size_t i = 0; i != ncols; i++) {
        // Compute the nearest source pixel for this destination pixel
        // Multiply the destination pixel by the rotation matrix
        srow_ = int(round(cos45 * T(j) + -1 * cos45 * T(i)));
        scol_ = int(round(cos45 * T(j) + cos45 * T(i)));
        srow_ += int(offset_);
        if (0 <= srow_ && srow_ < int(z) && 0 <= scol_ && scol_ < int(z)) {
          // Copy the source pixel to the destination pixel
          unrotated[j * ncols + i] = rotated[srow_ * z + scol_];
        }
      }
    }
  }
};

#endif // NTA_ROTATION_HPP
