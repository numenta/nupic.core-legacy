//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

// NOTES:
// Note about MurmurHash3
//    The upstream repository for MurmurHash3 is not active. It is not merging fixes, even ones which
//    pass all of thier unit tests. The maintainer appears to be only interested in the algorithms,
//    and not the implementations.
//
//    @ctrl-z-9000-times modified the upstream code as follows:
//      - to not use switch case statements with implicit fallthroughs because our build
//        settings disallow that.
//      - removed all the other functions which we dont need.
//      - added namespace htm.
//
//    The seed argument type and the return type must match type UInt32.  This algorithm
//    only works for 32bit integers.

#include <cstdint>

namespace htm {

  uint32_t MurmurHash3_x86_32( const void * key, int len, uint32_t seed );

}      // End namespace htm
#endif // _MURMURHASH3_H_
