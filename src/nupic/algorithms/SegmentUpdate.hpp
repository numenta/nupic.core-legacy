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

#ifndef NTA_SEGMENTUPDATE_HPP
#define NTA_SEGMENTUPDATE_HPP

#include <nupic/proto/SegmentUpdate.capnp.h>
#include <nupic/types/Serializable.hpp>
#include <nupic/types/Types.hpp>
#include <vector>
using namespace nupic;

namespace nupic {
namespace algorithms {
namespace Cells4 {

class Cells4;

//------------------------------------------------------------------------
//------------------------------------------------------------------------
/**
 * SegmentUpdate stores information to update segments by creating, removing
 * or updating synapses.
 * SegmentUpdates are applied to the segment they target on a different
 * iteration than the iteration they were created in. SegmentUpdates have a
 * timeStamp, and they are discarded without being applied if they become
 * 'stale'.
 */
class SegmentUpdate : Serializable<SegmentUpdateProto> {
public:
  typedef std::vector<UInt>::const_iterator const_iterator;

private:
  bool _sequenceSegment;       // used when creating a new segment
  UInt _cellIdx;               // the index of the target cell
  UInt _segIdx;                // (UInt) -1 if creating new segment
  UInt _timeStamp;             // controls obsolescence of update
  std::vector<UInt> _synapses; // contains source cell indices
  bool _phase1Flag;            // If true, this update was created
                               // during Phase 1 of compute
  bool _weaklyPredicting;      // Set true if segment only reaches
                               // activationThreshold when including
                               // unconnected synapses.

public:
  SegmentUpdate();

  //----------------------------------------------------------------------
  SegmentUpdate(UInt cellIdx, UInt segIdx, bool sequenceSegment, UInt timeStamp,
                std::vector<UInt> synapses = std::vector<UInt>(),
                bool phase1Flag = false, bool weaklyPredicting = false,
                Cells4 *cells = nullptr);

  //----------------------------------------------------------------------
  SegmentUpdate(const SegmentUpdate &o);

  //----------------------------------------------------------------------
  SegmentUpdate &operator=(const SegmentUpdate &o) {
    _cellIdx = o._cellIdx;
    _segIdx = o._segIdx;
    _sequenceSegment = o._sequenceSegment;
    _synapses = o._synapses;
    _timeStamp = o._timeStamp;
    _phase1Flag = o._phase1Flag;
    _weaklyPredicting = o._weaklyPredicting;
    NTA_ASSERT(invariants());
    return *this;
  }
  //---------------------------------------------------------------------
  bool operator==(const SegmentUpdate &other) const;
  inline bool operator!=(const SegmentUpdate &other) const {
    return !operator==(other);
  }

  //---------------------------------------------------------------------
  bool isSequenceSegment() const { return _sequenceSegment; }
  UInt cellIdx() const { return _cellIdx; }
  UInt segIdx() const { return _segIdx; }
  UInt timeStamp() const { return _timeStamp; }
  UInt operator[](UInt idx) const { return _synapses[idx]; }
  const_iterator begin() const { return _synapses.begin(); }
  const_iterator end() const { return _synapses.end(); }
  UInt size() const { return _synapses.size(); }
  bool empty() const { return _synapses.empty(); }
  bool isNewSegment() const { return _segIdx == (UInt)-1; }
  bool isPhase1Segment() const { return _phase1Flag; }
  bool isWeaklyPredicting() const { return _weaklyPredicting; }

  //---------------------------------------------------------------------
  /**
   * Checks that all indices are in range, and that the synapse src cell indices
   * are unique and sorted.
   */
  bool invariants(Cells4 *cells = nullptr) const;

  //---------------------------------------------------------------------
  using Serializable::write;
  void write(SegmentUpdateProto::Builder &proto) const override {
    proto.setSequenceSegment(_sequenceSegment);
    proto.setCellIdx(_cellIdx);
    proto.setSegIdx(_segIdx);
    proto.setTimestamp(_timeStamp);
    auto synapsesProto = proto.initSynapses(_synapses.size());
    for (UInt i = 0; i < _synapses.size(); ++i) {
      synapsesProto.set(i, _synapses[i]);
    }
    proto.setPhase1Flag(_phase1Flag);
    proto.setWeaklyPredicting(_weaklyPredicting);
  }

  //---------------------------------------------------------------------
  using Serializable::read;
  void read(SegmentUpdateProto::Reader &proto) override {
    _sequenceSegment = proto.getSequenceSegment();
    _cellIdx = proto.getCellIdx();
    _segIdx = proto.getSegIdx();
    _timeStamp = proto.getTimestamp();
    auto synapsesProto = proto.getSynapses();
    _synapses.resize(synapsesProto.size());
    for (UInt i = 0; i < synapsesProto.size(); ++i) {
      _synapses[i] = synapsesProto[i];
    }
    _phase1Flag = proto.getPhase1Flag();
    _weaklyPredicting = proto.getWeaklyPredicting();
  }

  //---------------------------------------------------------------------
  void save(std::ostream &outStream) const {
    outStream << _cellIdx << " " << _segIdx << " " << _phase1Flag << " "
              << _sequenceSegment << " " << _weaklyPredicting << " "
              << _timeStamp << std::endl;
    outStream << _synapses.size() << " ";
    for (auto &elem : _synapses) {
      outStream << elem << " ";
    }
  }

  //---------------------------------------------------------------------
  void load(std::istream &inStream) {
    inStream >> _cellIdx >> _segIdx >> _phase1Flag >> _sequenceSegment >>
        _weaklyPredicting >> _timeStamp;
    UInt n, syn;
    inStream >> n;
    for (UInt i = 0; i < n; ++i) {
      inStream >> syn;
      _synapses.push_back(syn);
    }
  }

  //---------------------------------------------------------------------
  void print(std::ostream &outStream, bool longFormat = false,
             UInt nCellsPerCol = 0) const {
    if (!longFormat) {

      outStream << 'c' << _cellIdx << " s" << _segIdx
                << (_phase1Flag ? " p1 " : " p2 ")
                << (_sequenceSegment ? " ss" : " ")
                << (_weaklyPredicting ? " wp" : " sp") << " t" << _timeStamp
                << '/';

    } else {
      NTA_CHECK(nCellsPerCol > 0);
      UInt col = (UInt)(_cellIdx / nCellsPerCol);
      UInt cell = _cellIdx - col * nCellsPerCol;
      outStream << "cell: "
                << "[" << col << "," << cell << "]  ";
      outStream << " seg: " << _segIdx << (_sequenceSegment ? " seqSeg " : " ")
                << "timeStamp: " << _timeStamp << " / src cells: ";
    }

    // Print out list of source cell indices
    for (UInt i = 0; i != _synapses.size(); ++i)
      outStream << _synapses[i] << ' ';
  }
};

//--------------------------------------------------------------------------------
#ifndef SWIG
inline std::ostream &operator<<(std::ostream &outStream,
                                const SegmentUpdate &update) {
  update.print(outStream);
  return outStream;
}
#endif

// End namespace
} // namespace Cells4
} // namespace algorithms
} // namespace nupic

#endif // NTA_SEGMENTUPDATE_HPP
