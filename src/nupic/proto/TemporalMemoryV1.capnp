@0xa64ad1d724c51d4c;

# TODO: Use absolute path
using import "RandomProto.capnp".RandomProto;

# Next ID: ???
struct TemporalMemoryV1Proto {
  version @0 :UInt16;
  verbosity @1 :UInt8;
  random @2 :RandomProto;

  numberOfCols @3 :UInt32;
  cellsPerColumn @4 :UInt32;
  maxSegmentsPerCell @5 :UInt32;

  activationThreshold @6 :UInt32;
  minThreshold @7 :UInt32;
  segUpdateValidDuration @8 :UInt32;
  newSynapseCount @9 :UInt32;
  maxSynapsesPerSegment @10 :UInt32;
  initSegFreq @11 :Float32;

  initialPerm @12 :Float32;
  connectedPerm @13 :Float32;
  permanenceMax @14 :Float32;
  permanenceDec @15 :Float32;
  permanenceInc @16 :Float32;
  checkSynapseConsistency @17 :Bool;

  doPooling @18 :Bool;
  globalDecay @19 :Float32;
  maxAge @20 :UInt32;
  maxInfBacktrack @21 :UInt32;
  maxLrnBacktrack @22 :UInt32;
  pamLength @23 :UInt32;
  pamCounter @24 :UInt32;
  learnedSeqLength @25 :UInt32;
  avgLearnedSeqLength @26 :Float32;
  maxSeqLength @27 :UInt32;
  avgInputDensity @28 :Float32;
  outputType @29 : Text;
  burnIn @30 : UInt32;
  collectStats @31 : Bool;
  collectSequenceStats @32 :Bool;

  iterationIdx @33 :UInt32;
  lrnIterationIdx @34 :UInt32;
  resetCalled @35 :Bool;
  nextSegIdx @36 :UInt32;

  cells @37 :List(Cell);
  segmentUpdates @38 :List(SegmentUpdate);

  ownsMemory @39 :Bool;
  lrnActiveStateT1 @40 :List(UInt8);
  lrnActiveStateT @41 :List(UInt8);
  lrnPredictedStateT1 @42 :List(UInt8);
  lrnPredictedStateT @43 :List(UInt8);
  infActiveStateT1 @44 :List(UInt8);
  infActiveStateT @45 :List(UInt8);
  infActiveStateBackup @46 :List(UInt8);
  infActiveStateCandidate @47 :List(UInt8);
  infPredictedStateT1 @48 :List(UInt8);
  infPredictedStateT @49 :List(UInt8);
  cellConfidenceT1 @50 :List(Float32);
  cellConfidenceT @51 :List(Float32);
  colConfidenceT1 @52 :List(Float32);
  colConfidenceT @53 :List(Float32);

  prevLrnPatterns @54 :List(List(UInt32));
  prevInfPatterns @55 :List(List(UInt32));

  statsNumInfersSinceReset @56 : UInt32;
  statsNumPredictions @57 : UInt32;
  statsCurPredictionScore2 @58 :Float32;
  statsPredictionScoreTotal2 @59 :Float32;
  statsCurFalseNegativeScore @60 :Float32;
  statsFalseNegativeScoreTotal @61 :Float32;
  statsCurFalsePositiveScore @62 :Float32;
  statsFalsePositiveScoreTotal @63 :Float32;
  statsPctExtraTotal @64 :Float32;
  statsPctMissingTotal @65 :Float32;
  statsCurMissing @66 : UInt32;
  statsCurExtra @67 : UInt32;
  statsTotalMissing @68 : UInt32;
  statsTotalExtra @69 : UInt32;
  statsPrevSequenceSignature @70 :List(Float32);
  statsConfHistogram @71 :List(List(Float32));

  # Next ID: ???
  struct Cell {
    segments @0 :List(Segment);
  }

  # Next ID: ???
  struct Segment {
    segIdx @0 :UInt32;
    sequenceSegment @1 :Bool;
    frequency @2 :Float32;
    numConnectedSynapses @3 :UInt32;
    totalActivations @4 :UInt32;
    positiveActivations @5 :UInt32;
    lastActiveIteration @6 :UInt32;
    lastPosDutyCycle @7 :Float32;
    lastPosDutyCycleIteration @8 :UInt32;
    synapses @9 :List(Synapse);
  }

  # Next ID: ???
  struct Synapse {
    srcCellIdx @0 :UInt64;
    permanence @1 :Float32;
  }

  # Next ID: ???
  struct SegmentUpdate {
    cellIdx @0 :UInt32;
    segIdx @1 :UInt32;
    sequenceSegment @2 :Bool;
    synapses @3 :List(UInt32);
    newSynapses @4 :List(UInt64);
    lrnIterationIdx @5 :UInt32;
    phase1 @6 :Bool;
    weaklyPredicting @7 :Bool;
  }
}
