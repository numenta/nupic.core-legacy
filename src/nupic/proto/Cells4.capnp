@0xd5c4908fa0384eda;

using import "/nupic/proto/Cell.capnp".CellProto;
using import "/nupic/proto/RandomProto.capnp".RandomProto;
using import "/nupic/proto/Segment.capnp".CStateProto;
using import "/nupic/proto/SegmentUpdate.capnp".SegmentUpdateProto;

# Next ID: 39
struct Cells4Proto {
  version @0 :UInt16;
  ownsMemory @1 :Bool;
  rng @2 :RandomProto;
  nColumns @3 :UInt32;
  nCellsPerCol @4 :UInt32;
  activationThreshold @5 :UInt32;
  minThreshold @6 :UInt32;
  newSynapseCount @7 :UInt32;
  nIterations @8 :UInt32;
  nLrnIterations @9 :UInt32;
  segUpdateValidDuration @10 :UInt32;
  initSegFreq @11 :Float32;
  permInitial @12 :Float32;
  permConnected @13 :Float32;
  permMax @14 :Float32;
  permDec @15 :Float32;
  permInc @16 :Float32;
  globalDecay @17 :Float32;
  doPooling @18 :Bool;
  pamLength @19 :UInt32;
  maxInfBacktrack @20 :UInt32;
  maxLrnBacktrack @21 :UInt32;
  maxSeqLength @22 :UInt32;
  learnedSeqLength @23 :UInt32;
  avgLearnedSeqLength @24 :Float32;
  maxAge @25 :UInt32;
  verbosity @26 :UInt8;
  maxSegmentsPerCell @27 :UInt32;
  maxSynapsesPerSegment @28 :UInt32;
  checkSynapseConsistency @29 :Bool;

  # Internal variables
  resetCalled @30 :Bool;

  avgInputDensity @31 :Float32;
  pamCounter @32 :UInt32;

  # The various inference and learning states
  learnActiveStateT @33 :CStateProto;
  learnActiveStateT1 @34 :CStateProto;
  learnPredictedStateT @35 :CStateProto;
  learnPredictedStateT1 @36 :CStateProto;

  cells @37 :List(CellProto);
  segmentUpdates @38 :List(SegmentUpdateProto);
}
