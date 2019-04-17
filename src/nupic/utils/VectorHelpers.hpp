#ifndef NTA_UTILS_VECTORHELPERS
#define NTA_UTILS_VECTORHELPERS

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <algorithm>

#include <nupic/types/Types.hpp>
#include <nupic/utils/Log.hpp> 


namespace nupic {
namespace utils {

class VectorHelpers
{
public:

  /**
   * Convert binary to sparse representation.
   */
  template<typename T>
  static std::vector<UInt> binaryToSparse(const std::vector<T>& binaryVector)
  {
    std::vector<UInt> sparse;
    for (UInt index = 0; index < binaryVector.size(); index++) {
      if (binaryVector[index] == (T)1) {
        sparse.push_back(index);
      }
    }
    return sparse;
  }

  /**
   * Convert sparse to binary representation.
   */
  template<typename T>
  static std::vector<T> sparseToBinary(const std::vector<UInt>& sparseVector, UInt width)
  {
    std::vector<T> binary(width);
    for (auto sparseIdx: sparseVector) {
      binary[sparseIdx] = (T)1;
    }
    return binary;
  }
};

} // end namespace utils
} // end namespace nupic
#endif // end NTA_UTILS_VECTORHELPERS
