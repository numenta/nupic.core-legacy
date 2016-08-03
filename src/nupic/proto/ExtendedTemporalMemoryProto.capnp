@0xe6601be8ebc25c9b;

# TODO: Use absolute path
using import "/nupic/proto/ConnectionsProto.capnp".ConnectionsProto;
using import "/nupic/proto/ConnectionsProto.capnp".ConnectionsProto.SegmentOverlapProto;
using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 21
struct ExtendedTemporalMemoryProto {

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
  formInternalBasalConnections @10 :Bool;

  basalConnections @11 :ConnectionsProto;
  apicalConnections @12 :ConnectionsProto;
  random @13 :RandomProto;

  # Lists of indices
  activeCells @14 :List(UInt32);
  winnerCells @15 :List(UInt32);

  predictedSegmentDecrement @16 :Float32;
  activeBasalSegmentOverlaps @17 :List(SegmentOverlapProto);
  matchingBasalSegmentOverlaps @18 :List(SegmentOverlapProto);
  activeApicalSegmentOverlaps @19 :List(SegmentOverlapProto);
  matchingApicalSegmentOverlaps @20 :List(SegmentOverlapProto);
}
