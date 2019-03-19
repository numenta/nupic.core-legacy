//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

#include "MurmurHash3.h"

namespace nupic {

/*
 * Don't worry about the technically undefined behavior when r >= 32, since this
 * is only used with a hardcoded r
 */
inline UInt32 rotl32 ( UInt32 x, char r )
{
  return (x << r) | (x >> (32 - r));
}

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here
inline UInt32 getblock32( const UInt32 * p, int i )
{
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return p[i];
  #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32( p[i] );
  #else
    #error "weird byte order"
  #endif
}

//-----------------------------------------------------------------------------
// Finalization mix - force all bits of a hash block to avalanche
inline UInt32 fmix32 ( UInt32 h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

//-----------------------------------------------------------------------------
UInt32 MurmurHash3_x86_32( const void * key, int len, UInt32 seed )
{
  const unsigned char * data = (const unsigned char *) key;
  const int nblocks = len / 4;

  UInt32 h1 = seed;

  const UInt32 c1 = 0xcc9e2d51;
  const UInt32 c2 = 0x1b873593;

  //----------
  // body

  const UInt32 * blocks = (const UInt32 *)(data + nblocks * 4);

  for(int i = -nblocks; i; i++)
  {
    UInt32 k1 = getblock32(blocks,i);

    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1,13);
    h1 = h1 * 5 + 0xe6546b64;
  }

  //----------
  // tail

  const unsigned char * tail = (const unsigned char*)(data + nblocks * 4);

  UInt32 k1 = 0;

  const auto tail_switch = len & 3;
  if( tail_switch == 3 )
  {
    k1 ^= tail[2] << 16;
  }
  if( tail_switch >= 2 )
  {
    k1 ^= tail[1] << 8;
  }
  if( tail_switch >= 1 )
  {
    k1 ^= tail[0];
    k1 *= c1;
    k1  = rotl32(k1,15);
    k1 *= c2;
    h1 ^= k1;
  }

  //----------
  // finalization

  h1 ^= len;

  h1 = fmix32(h1);

  return h1;
}
} // End namespace nupic
