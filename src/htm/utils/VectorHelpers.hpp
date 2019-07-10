#ifndef NTA_UTILS_VECTORHELPERS
#define NTA_UTILS_VECTORHELPERS

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <algorithm>

#include <htm/types/Types.hpp>
#include <htm/utils/Log.hpp> 


namespace htm {

class VectorHelpers
{
public:

  /**
   * Convert sparse to binary representation.
   */
  template<typename T>
  static std::vector<T> sparseToBinary(const std::vector<UInt>& sparseVector, UInt width)
  {
    std::vector<T> binary(width);
    for (auto sparseIdx: sparseVector) {
      NTA_ASSERT(sparseIdx < binary.size()) << "attemping to insert out of bounds element! " << sparseIdx;
      binary[sparseIdx] = (T)1;
    }
    return binary;
  }
};

} // end namespace htm
#endif // end NTA_UTILS_VECTORHELPERS
