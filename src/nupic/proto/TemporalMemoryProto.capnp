@0xc5bf8243b0c10764;

# TODO: Use absolute path
using import "/nupic/proto/ConnectionsProto.capnp".ConnectionsProto;
using import "/nupic/proto/ConnectionsProto.capnp".ConnectionsProto.SegmentOverlapProto;
using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 21
struct TemporalMemoryProto {

  columnDimensions @0 :List(UInt32);
  cellsPerColumn @1 :UInt32;
  activationThreshold @2 :UInt32;
  learningRadius @3 :UInt32;
  initialPermanence @4 :Float32;
  connectedPermanence @5 :Float32;
  minThreshold @6 :UInt32;
  maxNewSynapseCount @7 :UInt32;
  permanenceIncrement @8 :Float32;
  permanenceDecrement @9 :Float32;  

  connections @10 :ConnectionsProto;
  random @11 :RandomProto;

  # Lists of indices
  activeCells @12 :List(UInt32);

  # Obsolete, information contained in activeSegmentOverlaps
  predictiveCells @13 :List(UInt32);

  # Obsolete, replaced by activeSegmentOverlaps
  activeSegments @14 :List(UInt32);

  winnerCells @15 :List(UInt32);

  # Obsolete, replaced by matchingSegmentOverlaps
  matchingSegments @16 :List(UInt32);

  # Obsolete, information contained in matchingSegmentOverlaps
  matchingCells @17 :List(UInt32);

  predictedSegmentDecrement @18 :Float32;
  activeSegmentOverlaps @19 :List(SegmentOverlapProto);
  matchingSegmentOverlaps @20 :List(SegmentOverlapProto);
}
