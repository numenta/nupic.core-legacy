#ifndef NTA_UTILS_VECTORHELPERS_HPP
#define NTA_UTILS_VECTORHELPERS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

namespace nupic {
namespace utils {

class VectorHelpers
{
    public:
        /**
        * pretty print a vector v, with separator string sep, to output stream os (default cout)
        */
  template<typename T>
  static std::string printVector(const std::vector<T>& vec, std::string sep=",")
  {
    std::stringstream ss;
    for(auto v: vec) {
      ss << v << sep;
    }
  return ss.str();
  }

};

}}
#endif
