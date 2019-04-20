/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

/** @file
 * Definitions for the base Serializable class in C++
 *
 * CapnProto serialization has been removed and replaced with Cereal binary streams.
 * See: https://github.com/USCiLab/cereal
 *    dkeeney 8/15/2018
 */

#ifndef NTA_SERIALIZABLE_HPP
#define NTA_SERIALIZABLE_HPP


#include <iostream>
#include <fstream>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>
#include <nupic/os/ImportFilesystem.hpp>

#define CEREAL_SAVE_FUNCTION_NAME save_ar
#define CEREAL_LOAD_FUNCTION_NAME load_ar
#define CEREAL_SERIALIZE_FUNCTION_NAME serialize_ar
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

// TODO: The RapidJson distribution (Used by Cereal 1.2.2) had a problem with these warnings.
// It is being fixed in next RapidJson release. Do not know which Cereal release that will be in.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexceptions"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
  #if ((__GNUC__ * 100) + __GNUC_MINOR__) >= 800    // gcc 8
    #pragma GCC diagnostic ignored "-Wclass-memaccess"
  #endif
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

#include <cereal/archives/json.hpp>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <cereal/types/memory.hpp>  // for serializing smart pointers
#include <cereal/types/vector.hpp>  // for serializing std::vector
#include <cereal/types/string.hpp>  // for serializing std::string
#include <cereal/types/map.hpp>     // for serializing std::map
#include <cereal/types/set.hpp>     // for serializing std::set
#include <cereal/types/deque.hpp>   // for serializing std::deque

#define SERIALIZABLE_VERSION 3


namespace nupic {


typedef enum {BINARY, PORTABLE, JSON, XML} SerializableFormat;

// Design explanation:
//
// Archive?
// see: http://uscilab.github.io/cereal/serialization_archives.html
// The Cereal package passes the varients of the Archive package around to act sort of
// like ostream and such.  Archive acts like a base class but in fact it is nothing 
// but a typename in the template and is a placeholder for BinaryOutputArchive,
// JSONInputArchive, etc. which determine the format and stream used by the output.
//
// These Archive variants seem to indirectly subclass off of iostream because function
// overloads like this will give a compile error saying they are ambiguous.
//    void save(ostream& f);
//    void save(BinaryOutputArchive& f);
// And like iostream there is no default constructors and no assignment operators.
// In other words it cannot be copied. They cannot be dynamically allocated and
// they cannot be a member of a class unless it is initialized at instantiation time
// with its stream argument. But we can make a pointer or reference to it.
//
// What's this CerealAdapter macro?
// What we would like to do is have functions in this base class that can call the 
// save(Archive& ar ) and load(Archive& ar ) in the target class.  Normally we 
// would just use inheritance and setup a virtual method and matching functions 
// in the derived class that override it.  But these are templated functions so will
// not work. We have to do this indirectly.  The macro CerealAdapter adds some 
// non-templeted helper functions to the derived class that can use inheritance.  
// We wrap the Archive in the ArWrapper class and pass it to these helper classes.  
// The helper classes can then call save(Archive& ar ) and load(Archive& ar ) with 
// the templates since they are in the same class.
//
// Note that care must be taken to be sure that the selected Archive object (the thing
// being pointed to) remains in scope while the save(Archive& ar ) and load(Archive& ar ) 
// are called but that they must go out of scope as soon as they return.  This causes the
// serialization to be flushed to the underlining stream.
//
// Why the ArWrapper?
// Archive is not really a type but rather a template typename and it resolves
// to the right real type at compile time.  But we are using our CerealAdapter to
// resolve out type a run time.  So, our ArWrapper needs to contain all possible
// variations of Archive, only one of which we will be actually using. The Archive 
// variants have no default constructor so they must be pointers.  A union would 
// nice but C++ rejects any use of a union in this situation.
//
// Since we are doing our selection at run time, we need an enum in the ArWrapper
// to select the right type that matches the template when calling save(Archive& ar ) 
// and load(Archive& ar ).  So that is why we have the switch(fmt) selection in
// the CerealAdapter macro.
//
// Now we can use saveToFile(filename, fmt) and saveToStream(stream, fmt) and 
// their corresponding load pairs and whatever else we might need in the future
// without requiring all Serialization derived classes to independently implement
// them. The Cereal package does not need to be exposed in the API.
//
// When setting up a new class that should be serialized, do the following:
// 1) subclass it from Serialization.
// 2) Add CerealAdapter macro and save(Archive& ar) and load(Archive& ar) methods
//    to the class as shown at the end of this file.
// 3) Implement the save and load functions as described in the Cereal documentation
//    see http://uscilab.github.io/cereal/index.html
//    - Note that you can directly serialize:
//      a) Any fundemental data type (numeric values).
//      b) A std::string
//      c) Most of the STL containers.
//      d) Any other class that is subclassed from Serialize.
//      e) Any smart pointer (std::shared_ptr) if contains above types, non-array.
//    - It will not serialize pointers or arrays directly.
//    - You can use the syntax of 
//            ar( arg1, arg2, arg3 );
//      or
//            ar << arg1 << arg2 << arg3;
//    - Use the macro CEREAL_NVP(sdr); or ar(cereal::make_nvp("SDR", sdr));
//      to give names to variables in the JSON serialization.
//    - Serialize a raw array, start with length followed by serialization of each element.
//      the sizeTag starts a sequence in JSON.
//            ar(cereal::make_size_tag(static_cast<cereal::size_type>(count_)));
//            for (size_t i = 0; i < count_; i++)
//              ar( array[i] );
//    - Serialize an std::pair
//            ar(cereal::make_map_item(it->first, it->second));
//    - Extra attention is needed if a variable is in a base class. See docs.
//    - Extra attention may be needed for some private variables. See docs.
//

class ArWrapper {
public:
  ArWrapper() { }
  ArWrapper(cereal::BinaryOutputArchive* ar)         { fmt = SerializableFormat::BINARY; binary_out = ar; }
  ArWrapper(cereal::PortableBinaryOutputArchive* ar) { fmt = SerializableFormat::PORTABLE; portable_out = ar; }
  ArWrapper(cereal::JSONOutputArchive* ar)           { fmt = SerializableFormat::JSON; json_out = ar; }
  ArWrapper(cereal::XMLOutputArchive* ar)            { fmt = SerializableFormat::XML; xml_out = ar; }
  ArWrapper(cereal::BinaryInputArchive* ar)          { fmt = SerializableFormat::BINARY; binary_in = ar; }
  ArWrapper(cereal::PortableBinaryInputArchive* ar)  { fmt = SerializableFormat::PORTABLE; portable_in = ar; }
  ArWrapper(cereal::JSONInputArchive* ar)            { fmt = SerializableFormat::JSON; json_in = ar; }
  ArWrapper(cereal::XMLInputArchive* ar)             { fmt = SerializableFormat::XML; xml_in = ar; }

  SerializableFormat fmt;
  cereal::BinaryOutputArchive* binary_out;
  cereal::PortableBinaryOutputArchive* portable_out;
  cereal::JSONOutputArchive* json_out;
  cereal::XMLOutputArchive* xml_out;
  cereal::BinaryInputArchive* binary_in;
  cereal::PortableBinaryInputArchive* portable_in;
  cereal::JSONInputArchive* json_in;
  cereal::XMLInputArchive* xml_in;

};


/**
 * Base Serializable class that any serializable class
 * must inherit from.
 */
class Serializable {
public:
  Serializable() {}
  virtual inline int getSerializableVersion() const { return SERIALIZABLE_VERSION; }

// TODO:Cereal- To be removed after Cereal is in place.
  virtual inline void saveToFile(std::string filePath) const {
    std::string dirPath = Path::getParent(filePath);
	  Directory::create(dirPath, true, true);
	  std::ofstream out(filePath, std::ios_base::out | std::ios_base::binary);
	  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	  out.precision(std::numeric_limits<double>::digits10 + 1);
	  out.precision(std::numeric_limits<float>::digits10 + 1);
	  save(out);
	  out.close();
  }

// TODO:Cereal- To be removed after Cereal is in place.
  virtual inline void loadFromFile(std::string filePath) {
    std::ifstream in(filePath, std::ios_base::in | std::ios_base::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    load(in);
    in.close();
  }
	// TODO:Cereal- after all serialization using Cereal is complete, 
  //       remove save() and load() pairs from all derived classes
	//       change all save_ar() load_ar() pairs to be save() load().
	//       Remove the following two lines.

  // These must be implemented by the subclass.
  virtual void save(std::ostream &stream) const {};
  virtual void load(std::istream &stream) {};


  virtual inline void saveToFile_ar(std::string filePath, SerializableFormat fmt=SerializableFormat::BINARY) const {
      std::string dirPath = Path::getParent(filePath);
	  Directory::create(dirPath, true, true);
	  std::ofstream out(filePath, std::ios_base::out | std::ios_base::binary);
	  out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	  out.precision(std::numeric_limits<double>::digits10 + 1);
	  out.precision(std::numeric_limits<float>::digits10 + 1);
		saveToStream_ar(out, fmt);
		out.close();
	}
  virtual inline void saveToStream_ar(std::ostream &out, SerializableFormat fmt=SerializableFormat::BINARY) const {
    ArWrapper arw;
    arw.fmt = fmt;
		switch(fmt) {
		case SerializableFormat::BINARY:  { 
      cereal::BinaryOutputArchive ar(out);
      arw.binary_out = &ar;
      cereal_adapter_save(arw); 
    } break;
		case SerializableFormat::PORTABLE: {
      cereal::PortableBinaryOutputArchive ar( out, cereal::PortableBinaryOutputArchive::Options::Default() ); 
      arw.portable_out = &ar;
      cereal_adapter_save(arw); 
    } break;
		case SerializableFormat::JSON: {
      cereal::JSONOutputArchive ar(out);
      arw.json_out = &ar;
      cereal_adapter_save(arw);
    } break;
		case SerializableFormat::XML: {
      cereal::XMLOutputArchive ar(out);
      arw.xml_out = &ar;
      cereal_adapter_save(arw);
    } break;
		default: NTA_THROW << "unknown serialization format.";
		}
  }
  
  virtual inline void loadFromFile_ar(std::string filePath, SerializableFormat fmt=SerializableFormat::BINARY) {
    std::ifstream in(filePath, std::ios_base::in | std::ios_base::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		loadFromStream_ar(in, fmt);
		in.close();
	}
	virtual inline void loadFromStream_ar(std::istream &in,  SerializableFormat fmt=SerializableFormat::BINARY) {
    ArWrapper arw;
    arw.fmt = fmt;
		switch(fmt) {
		case SerializableFormat::BINARY: {
      cereal::BinaryInputArchive ar(in);
      arw.binary_in = &ar;
      cereal_adapter_load(arw);
    } break;
		case SerializableFormat::PORTABLE: {
      cereal::PortableBinaryInputArchive ar(in, cereal::PortableBinaryInputArchive::Options::Default());
      arw.portable_in = &ar;
      cereal_adapter_load(arw);
    } break;
		case SerializableFormat::JSON: {
      cereal::JSONInputArchive ar(in);
      arw.json_in = &ar;
      cereal_adapter_load(arw);
    } break;
		case SerializableFormat::XML: {
      cereal::XMLInputArchive ar(in);
      arw.xml_in = &ar;
      cereal_adapter_load(arw);
    } break;
		default: NTA_THROW << "unknown serialization format.";
		}
  }

    


  // Note: if you get a compile error saying this is not defined,
  //       or that "cannot instantiate abstract class"
  //       add the macro 'CerealAdapter' in the derived class.
  // TODO:Cereal- make these pure virtual when Cereal is included everywhere.
  virtual void cereal_adapter_save(ArWrapper& a) const {};
  virtual void cereal_adapter_load(ArWrapper& a) {};


	

  virtual ~Serializable() {}

};

// A macro that adds two helper classes to each derived class.
#define CerealAdapter \
  void cereal_adapter_save(ArWrapper& a) const override {                 \
    switch(a.fmt) {                                                       \
    case SerializableFormat::BINARY:   save_ar(*a.binary_out); break;     \
    case SerializableFormat::PORTABLE: save_ar(*a.portable_out); break;   \
		case SerializableFormat::JSON:     save_ar(*a.json_out); break;       \
		case SerializableFormat::XML:      save_ar(*a.xml_out); break;        \
		default: NTA_THROW << "unknown serialization format.";                \
		}                                                                     \
  }                                                                       \
  void cereal_adapter_load(ArWrapper& a) override {                       \
    switch(a.fmt) {                                                       \
    case SerializableFormat::BINARY:   load_ar(*a.binary_in); break;      \
    case SerializableFormat::PORTABLE: load_ar(*a.portable_in); break;    \
		case SerializableFormat::JSON:     load_ar(*a.json_in); break;        \
		case SerializableFormat::XML:      load_ar(*a.xml_in); break;         \
		default: NTA_THROW << "unknown serialization format.";                \
		}                                                                     \
  }                             


/**** Example of derived Serializable class
class B : public Serializable {
public:
  int y;

  CerealAdapter;                 <=== don't forget this.
  template<class Archive>
  void save_ar(Archive& ar) {
    ar( y );
  }
  template<class Archive>
  void load_ar(Archive& ar) {
    ar( y );
  }
};
*****/

} // end namespace nupic

#endif // NTA_SERIALIZABLE_HPP

