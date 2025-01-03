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

#include <nupic/algorithms/Cell.hpp>
#include <nupic/proto/Cell.capnp.h>

using namespace nupic::algorithms::Cells4;
using namespace nupic;

Cell::Cell() : _segments(0), _freeSegments(0) {}

//------------------------------------------------------------------------------
/**
 * This variable is global to the Cell class and controls whether we
 * want to match python's segment ordering. If we are not matching Python's
 * segment order, we reuse segment slots in getFreeSegment.
 * Matching Python's segment order takes up a bit more memory in this
 * implementation, and is potentially a bit slower. In addition
 * some subtle differences show up between the Python and CPP implementations.
 * For example, in getBestMatchingCell if the two segments have activity equal
 * the max activity, different segments can get chosen.
 * The variable has no functional impact as far as accuracy is concerned.
 */
static bool cellMatchPythonSegOrder = false;

void Cell::setSegmentOrder(bool matchPythonOrder) {
  if (matchPythonOrder) {
    std::cout << "*** Python segment match turned on for Cells4\n";
  }
  cellMatchPythonSegOrder = matchPythonOrder;
}

//------------------------------------------------------------------------------
/**
 * Returns an empty segment to use, either from list of already
 * allocated ones that have been previously "freed" (but we kept
 * the memory allocated), or by allocating a new one.
 */
UInt Cell::getFreeSegment(const Segment::InSynapses &synapses,
                          Real initFrequency, bool sequenceSegmentFlag,
                          Real permConnected, UInt iteration) {
  NTA_ASSERT(!synapses.empty());

  UInt segIdx = 0;

  if (cellMatchPythonSegOrder) {
    // for unit tests where segment order matters

    segIdx = _segments.size();
    _segments.resize(_segments.size() + 1);

  } else {

    // Reuse segment slots, but that causes some unit tests
    // to fail, for example when 2 segments are in a different order
    // between C++ and Python, and they happen to have the same activity
    // level: both C++ and Python will compute the same update, but apply
    // it to 2 different segments!

    if (_freeSegments.empty()) {
      segIdx = _segments.size();
      // TODO: Should we grow by larger amounts here?
      _segments.resize(_segments.size() + 1);
    } else {
      segIdx = _freeSegments.back();
      _freeSegments.pop_back();
    }
  }

  NTA_ASSERT(segIdx < _segments.size());
  NTA_ASSERT(not_in(segIdx, _freeSegments));
  NTA_ASSERT(_segments[segIdx].empty()); // important in case we push_back

  _segments[segIdx] = Segment(synapses, initFrequency, sequenceSegmentFlag,
                              permConnected, iteration);

  return segIdx;
}

//--------------------------------------------------------------------------------
/**
 * Update the duty cycle of each segment in this cell
 */
void Cell::updateDutyCycle(UInt iterations) {
  for (UInt i = 0; i != _segments.size(); ++i) {
    if (!_segments[i].empty()) {
      _segments[i].dutyCycle(iterations, false, false);
    }
  }
}

//-----------------------------------------------------------------------------
void Cell::write(CellProto::Builder &proto) const {
  auto segmentsProto = proto.initSegments(_segments.size());
  for (UInt i = 0; i < _segments.size(); ++i) {
    auto segProto = segmentsProto[i];
    _segments[i].write(segProto);
  }
}

//-----------------------------------------------------------------------------
void Cell::read(CellProto::Reader &proto) {
  auto segmentsProto = proto.getSegments();
  _segments.resize(segmentsProto.size());
  _freeSegments.resize(0);
  for (UInt i = 0; i < segmentsProto.size(); ++i) {
    auto segProto = segmentsProto[i];
    _segments[i].read(segProto);
    if (_segments[i].empty()) {
      _freeSegments.push_back(i);
    }
  }
}

//-----------------------------------------------------------------------------
void Cell::save(std::ostream &outStream) const {
  outStream << _segments.size() << ' ';
  // TODO: save only non-empty segments
  for (UInt i = 0; i != _segments.size(); ++i) {
    _segments[i].save(outStream);
    outStream << ' ';
  }
}

//----------------------------------------------------------------------------
void Cell::load(std::istream &inStream) {
  UInt n = 0;

  inStream >> n;

  _segments.resize(n);
  _freeSegments.resize(0);

  for (UInt i = 0; i != (UInt)n; ++i) {
    _segments[i].load(inStream);
    if (_segments[i].empty())
      _freeSegments.push_back(i);
  }
}
bool Cell::operator==(const Cell &other) const {
  if (_freeSegments != other._freeSegments) {
    return false;
  }
  return _segments == other._segments;
}
