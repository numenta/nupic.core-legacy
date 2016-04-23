#ifndef NTA_UTILS_CSVHELPERS
#define NTA_UTILS_CSVHELPERS

#include <iostream>
#include <boost/tokenizer.hpp>
#include <vector>
#include <string>
#include <memory>
#include <fstream>

#include <nupic/types/Types.hpp>
//#include "nupic/utils/VectorHelpers.hpp"

namespace nupic {
namespace utils {
/** 
* defines I/O classes for CSV - CSVReader & CSVWriter
*/
namespace csv {

//using namespace nupic::utils;
// read
template<class T>
class CSVReader
{
    public:
        CSVReader(std::string filename, UInt skipNHeaderLines=0)
{
    this->filename_ = filename;
    reset();
    if (!this->lineIterator_->is_open()) {
        throw "Invalid file";
    }
    // skip N header rows
    for(UInt i=0; i< skipNHeaderLines; i++) {
      std::string line;
      getline(*(this->lineIterator_), line); // why "" does not work, while 'line' does?
    }
}


        /**
        * read a line/row as a vector of separate column elements
        */
        std::vector<std::string> getLine()
{
    if(this->eof()) {
      return std::vector<std::string>();
    }
    // Read the CSV file into a couple of vectors
    std::string line;
    getline(*(this->lineIterator_), line);

    // Use a boost tokenizer to parse a CSV line
    boost::tokenizer<boost::escaped_list_separator<char>> tok(line);
    std::vector<std::string> tokens(tok.begin(), tok.end());

//    std::vector<float> stof = VectorHelpers::stringToFloatVector(tokens); //force string format
    return tokens; //VectorHelpers::castVectorType<float, T>(stof);
}


        /**
        * read a whole column of the CSV file
        */
        std::vector<std::string> readColumn(UInt index)
{
  std::vector<std::string> column;
  while (!this->eof()) {
    auto row = this->getLine();
    if (row.size() > 0) {
      column.push_back(row[index]);
    }
  }
  reset();
  return column;
}


        /**
        * return min/max value in the CSV for given column
        */
        const T maxValue(UInt column)
{
  std::vector<T> col = this->readColumn(column);
  return *std::max_element(col.begin(), col.end());
}


        const T minValue(UInt column)
{
  std::vector<T> col = this->readColumn(column);
  return *std::min_element(col.begin(), col.end());
}

    private:
        std::unique_ptr<std::ifstream> lineIterator_;
        const std::string filename_;
        /**
        * end of file
        */
        bool eof()
{
    return (*(this->lineIterator_)).eof();
}
        /** reset **/
        void reset()
{
  lineIterator_.reset(new std::ifstream(filename_.c_str()));
}

};

//write
template<class T>
class CSVWriter
{
    public:
        CSVWriter(const std::string& filename, std::string header="")
{
  this->fp_.reset(new std::ofstream(filename));
  *(this->fp_) << header << std::endl;
}


        void writeLine(const std::vector<T>& cells, std::string separator=",")
{
  //VectorHelpers::print_vector(cells, separator, "", *(this->fp_)); FIXME later
}


        ~CSVWriter()
{
  (*(this->fp_)).close();
}


    private:
        std::unique_ptr<std::ofstream> fp_;
};

}}} //namespaces

#endif
