@0xc5bf8243b0c10764;

# TODO: Use absolute path
using import "/nupic/proto/ConnectionsProto.capnp".ConnectionsProto;
using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 22
struct TemporalMemoryProto {

  struct SegmentPath {
    cell @0 :UInt32;
    idxOnCell @1 :UInt32;
  }

  struct SegmentUInt32Pair {
    cell @0 :UInt32;
    idxOnCell @1 :UInt32;
    number @2 :UInt32;
  }

  struct SegmentUInt64Pair {
    cell @0 :UInt32;
    idxOnCell @1 :UInt32;
    number @2 :UInt64;
  }

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
  predictedSegmentDecrement @10 :Float32;
  maxSegmentsPerCell @11 :UInt16;
  maxSynapsesPerSegment @12 :UInt16;

  connections @13 :ConnectionsProto;
  random @14 :RandomProto;

  # Lists of indices
  activeCells @15 :List(UInt32);
  winnerCells @16 :List(UInt32);

  activeSegments @17 :List(SegmentPath);
  matchingSegments @18 :List(SegmentPath);

  numActivePotentialSynapsesForSegment @19 :List(SegmentUInt32Pair);

  iteration @20 :UInt64;
  lastUsedIterationForSegment @21 :List(SegmentUInt64Pair);
}
