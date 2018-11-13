#ifndef NTA_UTILS_VECTORHELPERS
#define NTA_UTILS_VECTORHELPERS

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include <algorithm>

#include <nupic/types/Types.hpp>


namespace nupic {
namespace utils {

class VectorHelpers
{
    public:
        /**
        * pretty print a vector v, with separator string sep, to output stream os (default cout)
        */
        template<typename T>
        static void print_vector(const std::vector<T>& v, std::string sep="", std::string prefix="", std::ostream& os=std::cout)
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
        * cast a vector to a different (compatible) type
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
        * convert binary to sparse representation
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
        * convert sparse to binary representation
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
        * convert representation of active cells(binary vector) to active columns(binary vector). 
        * If any cell of a column is active (1), the column is considered active. See TP for details.
        */
        static std::vector<UInt> cellsToColumns(const std::vector<UInt>& cellsBinary, const UInt cellsPerColumn)
{
  std::vector<UInt> activeColumns;
  for (UInt i = 0; i < cellsBinary.size(); i+= cellsPerColumn) { // loop over the whole (active) cells array
    UInt active = 0;
    for (UInt inColumn = 0; inColumn < cellsPerColumn; inColumn++) { // loop over cells in 1 column
      active |= cellsBinary[i + inColumn];
    }
    activeColumns.push_back(active);
  }
  return activeColumns;
}

};

}}

#endif
