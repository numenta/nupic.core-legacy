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

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/SegmentUpdate.hpp>
#include <nupic/utils/Log.hpp>

using namespace nupic::algorithms::Cells4;

SegmentUpdate::SegmentUpdate()
    : _sequenceSegment(false), _cellIdx((UInt)-1), _segIdx((UInt)-1),
      _timeStamp((UInt)-1), _synapses(), _phase1Flag(false),
      _weaklyPredicting(false) {}

SegmentUpdate::SegmentUpdate(UInt cellIdx, UInt segIdx, bool sequenceSegment,
                             UInt timeStamp, std::vector<UInt> synapses,
                             bool phase1Flag, bool weaklyPredicting,
                             Cells4 *cells)
    : _sequenceSegment(sequenceSegment), _cellIdx(cellIdx), _segIdx(segIdx),
      _timeStamp(timeStamp), _synapses(std::move(synapses)),
      _phase1Flag(phase1Flag), _weaklyPredicting(weaklyPredicting) {
  NTA_ASSERT(invariants(cells));
}

//--------------------------------------------------------------------------------
SegmentUpdate::SegmentUpdate(const SegmentUpdate &o) {
  _cellIdx = o._cellIdx;
  _segIdx = o._segIdx;
  _sequenceSegment = o._sequenceSegment;
  _synapses = o._synapses;
  _timeStamp = o._timeStamp;
  _phase1Flag = o._phase1Flag;
  _weaklyPredicting = o._weaklyPredicting;
  NTA_ASSERT(invariants());
}

bool SegmentUpdate::invariants(Cells4 *cells) const {
  bool ok = true;

  if (cells) {

    ok &= _cellIdx < cells->nCells();
    if (_segIdx != (UInt)-1)
      ok &= _segIdx < cells->__nSegmentsOnCell(_cellIdx);

    if (!_synapses.empty()) {
      for (UInt i = 0; i != _synapses.size(); ++i)
        ok &= _synapses[i] < cells->nCells();
      ok &= is_sorted(_synapses, true, true);
    }
  }

  return ok;
}

bool SegmentUpdate::operator==(const SegmentUpdate &o) const {

  if (_cellIdx != o._cellIdx || _segIdx != o._segIdx ||
      _sequenceSegment != o._sequenceSegment || _timeStamp != o._timeStamp ||
      _phase1Flag != o._phase1Flag ||
      _weaklyPredicting != o._weaklyPredicting) {
    return false;
  }
  return _synapses == o._synapses;
}
