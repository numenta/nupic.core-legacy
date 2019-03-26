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
#ifndef CEREAL_C_ARRAY_HPP_
#define CEREAL_C_ARRAY_HPP_

#include "nupic/utils/Log.hpp"
#include "nupic/types/BasicType.hpp"

using namespace cereal;
using namespace cereal::traits;
namespace nupic {

/**
  * Serialize/Deserialize a C array of any type.
  * When deserializing, buf must point to an already allocated array space so
  * the array size must already be known. The size should be serialized/deserialized
  * just prior to calling this method using a SizeTag:
  *     	ar( cereal::make_size_tag( static_cast<size_t>(count_) ) ); 
  * It is this SizeTag that tells JSON to start an array.  Otherwise it expects
  * to find name value pairs.
  *
  */
// FOR Binary 
template <class Archive, class T,
  EnableIf<is_same_archive<Archive, BinaryOutputArchive>::value
           && std::is_arithmetic<T>::value> = sfinae>
inline void save_c_array(Archive &ar, T *buf, size_t count) {
		ar(binary_data(buf, count * sizeof(T)));
}
template <class Archive, class T, 
  EnableIf<is_same_archive<Archive, BinaryInputArchive>::value
           && std::is_arithmetic<T>::value> = sfinae>
inline void load_c_array( Archive & ar,  T *buf, size_t count) {
  ar( binary_data( buf, count * sizeof(T) ) );
}

// FOR Text
template <class Archive, class T, 
  DisableIf<is_same_archive<Archive, BinaryOutputArchive>::value
           && std::is_arithmetic<T>::value> = sfinae>
inline void save_c_array( Archive & ar,  T *buf, size_t count) {
    for (size_t i = 0; i < count; i++) {
		  ar(buf[i]); 
		}
}
template <class Archive, class T, 
  DisableIf<is_same_archive<Archive, BinaryInputArchive>::value
           && std::is_arithmetic<T>::value> = sfinae>
inline void load_c_array( Archive & ar,  T *buf, size_t count) {
    for (size_t i = 0; i < count; i++) {
		  ar(buf[i]); 
		}
}

// FOR Boolean.  bool takes some special handling
template <class Archive> 
inline void save_c_array_bool( Archive & ar, bool *buf, size_t count) {
  ar( make_size_tag( static_cast<size_type>(count) ) ); // number of elements
  for (size_t i = 0; i < count; i++) {
    ar( static_cast<bool>(buf[i]) );
  }
}
 
template <class Archive> 
inline void load_c_array_bool( Archive & ar, bool *buf, size_t count) {
  ar( make_size_tag( count ) );
  for (size_t i = 0; i < count; i++) {
    bool b;
    ar( b );
    buf[i] = b;
  }
}


template <class Archive>
inline void save_c_array(Archive & ar, nupic::NTA_BasicType type, void* ptr, size_t count) {
  if (count == 0) return;
	switch (type) {
  case NTA_BasicType_Byte:   save_c_array(ar, (Byte *)ptr, count);  break;
	case NTA_BasicType_Int16:  save_c_array(ar, (Int16*)ptr, count);  break;
	case NTA_BasicType_UInt16: save_c_array(ar, (UInt16*)ptr, count); break;
	case NTA_BasicType_Int32:  save_c_array(ar, (Int32*)ptr, count);  break;
	case NTA_BasicType_UInt32: save_c_array(ar, (UInt32*)ptr, count); break;
	case NTA_BasicType_Int64:  save_c_array(ar, (Int64*)ptr, count);  break;
	case NTA_BasicType_UInt64: save_c_array(ar, (UInt64*)ptr, count); break;
	case NTA_BasicType_Real32: save_c_array(ar, (Real32*)ptr, count); break;
	case NTA_BasicType_Real64: save_c_array(ar, (Real64*)ptr, count); break;
	case NTA_BasicType_Bool:   save_c_array_bool(ar, (bool*)ptr, count);break;
	default:
	  NTA_THROW << "Unexpected Element Type: " << type;
	  break;
	}
}

template <class Archive>
inline void load_c_array(Archive & ar, nupic::NTA_BasicType type, void* ptr, size_t count) {
  if (count == 0) return;
	switch (type) {
	case NTA_BasicType_Byte:   load_c_array(ar, (Byte*)ptr, count);  break;
	case NTA_BasicType_Int16:  load_c_array(ar, (Int16*)ptr, count);  break;
	case NTA_BasicType_UInt16: load_c_array(ar, (UInt16*)ptr, count); break;
	case NTA_BasicType_Int32:  load_c_array(ar, (Int32*)ptr, count);  break;
	case NTA_BasicType_UInt32: load_c_array(ar, (UInt32*)ptr, count); break;
	case NTA_BasicType_Int64:  load_c_array(ar, (Int64*)ptr, count);  break;
	case NTA_BasicType_UInt64: load_c_array(ar, (UInt64*)ptr, count); break;
	case NTA_BasicType_Real32: load_c_array(ar, (Real32*)ptr, count); break;
	case NTA_BasicType_Real64: load_c_array(ar, (Real64*)ptr, count); break;
	case NTA_BasicType_Bool:   load_c_array_bool(ar, (bool*)ptr, count);break;
	default:
	  NTA_THROW << "Unexpected Element Type: " << type;
	  break;
	}
}


} // end cereal namespace

#endif // CEREAL_C_ARRAY_HPP_