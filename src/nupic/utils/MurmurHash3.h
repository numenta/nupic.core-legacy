//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

#include <nupic/types/Types.hpp>

namespace nupic {

  UInt32 MurmurHash3_x86_32( const void * key, int len, UInt32 seed );

}      // End namespace nupic
#endif // _MURMURHASH3_H_
