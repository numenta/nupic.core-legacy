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
   * Pretty print a vector, with separator string, to output stream.
   *
   * @param v      Vector to print.
   * @param sep    String to separate each element of vector.
   * @param prefix String to print before vector data.
   * @param os     std::ostream to write vector to, default: cout.
   */
  template<typename T>
  static void print_vector( const std::vector<T>& v,
                            std::string sep="",
                            std::string prefix="",
                            std::ostream& os=std::cout)
  {
    os << prefix;
    for (auto it=v.cbegin(); it != v.cend(); ++it) {
      os << *it;
      if ( ((it + 1) != v.cend()) &&  (sep != "") ) {
        os << sep;
      }
    }
    os << std::endl;
  }

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
