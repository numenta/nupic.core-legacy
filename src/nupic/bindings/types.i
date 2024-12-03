/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */


%include <nupic/types/Types.h>
%include <nupic/types/Types.hpp>

///////////////////////////////////////////////////////////////////
///  Bring in SWIG typemaps for base types and stl.
///////////////////////////////////////////////////////////////////
%include <typemaps.i>
%include <stl.i>
%include <std_list.i>
%include <std_set.i>

///////////////////////////////////////////////////////////////////
///  Instantiate templates that we will use.
///////////////////////////////////////////////////////////////////

%template(VectorOfInt32) std::vector<NTA_Int32>;
%template(VectorOfInt64) std::vector<NTA_Int64>;
%template(VectorOfUInt32) std::vector<NTA_UInt32>;
%template(VectorOfUInt64) std::vector<NTA_UInt64>;

%template(FloatVector) std::vector<NTA_Real32>;
%template(DoubleVector) std::vector<NTA_Real64>;

%template(StringVector) std::vector<std::string>;
%template(StringList) std::list<std::string>;
%template(StringSet) std::set<std::string>;
%template(StringMap) std::map<std::string, std::string>;

%template(StringStringPair) std::pair<std::string, std::string>;
%template(StringStringList) std::vector< std::pair<std::string, std::string> >;
%template(StringMapList) std::vector< std::map<std::string, std::string> >;
%template(StringIntPair) std::pair<std::string, NTA_Int32>;

%template(PairOfUInt32) std::pair<nupic::UInt32, nupic::UInt32>;
%template(VectorOfPairsOfUInt32) std::vector<std::pair<nupic::UInt32,nupic::UInt32> >;
%template(VectorOfVectorsOfPairsOfUInt32) std::vector<std::vector<std::pair<nupic::UInt32,nupic::UInt32> > >;

%template(PairUInt32Real32) std::pair<nupic::UInt32,nupic::Real32>;
%template(PairUInt32Real64) std::pair<nupic::UInt32,nupic::Real64>;
%template(VectorOfPairsUInt32Real32) std::vector<std::pair<nupic::UInt32,nupic::Real32> >;
%template(VectorOfPairsUInt32Real64) std::vector<std::pair<nupic::UInt32,nupic::Real64> >;




