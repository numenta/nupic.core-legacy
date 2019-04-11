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
   * Cast a vector to a different (compatible) type.
   */
  template<typename T1, typename T2>
  static std::vector<T2> castVectorType(const std::vector<T1>& orig)
  {
    std::vector<T2> dest;
    std::transform(orig.cbegin(), orig.cend(), std::back_inserter(dest),
                   [](const T1& elem) { return static_cast<T2>(elem); });
    return dest;
  }

  static std::vector<Real> stringToFloatVector(const std::vector<std::string>& orig)
  {
    std::vector<Real> dest;
    for(auto it=orig.cbegin(); it != orig.cend(); ++it) {
      dest.push_back(std::stof(*it));
    }
    return dest;
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


  /**
   * Create a Union of two vectors (An OR of the two).
   * The values are assumed to be sorted, sparse indexes.
   */
  template<typename T>
  static void unionOfVectors(std::vector<T>& out, 
                             const std::vector<T>& v1, 
                             const std::vector<T>& v2) {
    out.resize(v1.size() + v2.size());
    const auto it = std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), out.begin());
    out.resize(it - out.begin());
  }

};

} // end namespace utils
} // end namespace nupic
#endif // end NTA_UTILS_VECTORHELPERS
