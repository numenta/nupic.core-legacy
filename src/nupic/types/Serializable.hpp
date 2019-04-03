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
 * CapnProto serialization has been removed and replaced with binary streams.
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
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

// The RapidJson distribution (Used by Cereal) had a problem with these warnings.
// It is being fixed in next release.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexceptions"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

#include <cereal/archives/json.hpp>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 800)
#pragma GCC diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>

#define SERIALIZABLE_VERSION 3


namespace nupic {


typedef enum {BINARY, PORTABLE, JSON, XML} SerializableFormat;

// The Archives don't have default constructors or assignment operators so we need to pass as pointers.

class arWrapper {
public:
  arWrapper() { }
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
 * should inherit from.
 */
class Serializable {
public:
  Serializable() {}
  virtual inline int getSerializableVersion() const { return SERIALIZABLE_VERSION; }

// TODO: To be removed after Cereal is in place.
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

// TODO: To be removed after Cereal is in place.
  virtual inline void loadFromFile(std::string filePath) {
    std::ifstream in(filePath, std::ios_base::in | std::ios_base::binary);
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    load(in);
    in.close();
  }
	// TODO: after all serialization using Cereal is complete, 
	//       change all save_ar() load_ar() pairs to be save() load().
	//       Remove the following two lines.

  // These must be implemented by the subclass.
  virtual void save(std::ostream &stream) const = 0;
  virtual void load(std::istream &stream) = 0;


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
    arWrapper arw;
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
    arWrapper arw;
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
  virtual void cereal_adapter_save(arWrapper& a) const {};
  virtual void cereal_adapter_load(arWrapper& a) {};
	

  virtual ~Serializable() {}

};


#define CerealAdapter \
  void cereal_adapter_save(arWrapper& a) const override {                 \
    switch(a.fmt) {                                                       \
    case SerializableFormat::BINARY:   save_ar(*a.binary_out); break;     \
    case SerializableFormat::PORTABLE: save_ar(*a.portable_out); break;   \
		case SerializableFormat::JSON:     save_ar(*a.json_out); break;       \
		case SerializableFormat::XML:      save_ar(*a.xml_out); break;        \
		default: NTA_THROW << "unknown serialization format.";                \
		}                                                                     \
  }                                                                       \
  void cereal_adapter_load(arWrapper& a) override {                       \
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

