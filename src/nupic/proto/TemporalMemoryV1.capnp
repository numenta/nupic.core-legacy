@0xa64ad1d724c51d4c;

# TODO: Use absolute path
using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 72
struct TemporalMemoryV1Proto {
  verbosity @0 :UInt8;
  random @1 :RandomProto;

  numberOfCols @2 :UInt32;
  cellsPerColumn @3 :UInt32;
  maxSegmentsPerCell @4 :Int32;

  activationThreshold @5 :UInt32;
  minThreshold @6 :UInt32;
  segUpdateValidDuration @7 :UInt32;
  newSynapseCount @8 :UInt32;
  maxSynapsesPerSegment @9 :Int32;
  initSegFreq @10 :Float32;

  initialPerm @11 :Float32;
  connectedPerm @12 :Float32;
  permanenceMax @13 :Float32;
  permanenceDec @14 :Float32;
  permanenceInc @15 :Float32;
  checkSynapseConsistency @16 :Bool;

  doPooling @17 :Bool;
  globalDecay @18 :Float32;
  maxAge @19 :UInt32;
  maxInfBacktrack @20 :UInt32;
  maxLrnBacktrack @21 :UInt32;
  pamLength @22 :UInt32;
  pamCounter @23 :UInt32;
  learnedSeqLength @24 :UInt32;
  avgLearnedSeqLength @25 :Float32;
  maxSeqLength @26 :UInt32;
  avgInputDensity @27 :Float32;
  outputType @28 : Text;
  burnIn @29 : UInt32;
  collectStats @30 : Bool;
  collectSequenceStats @31 :Bool;

  iterationIdx @32 :UInt32;
  lrnIterationIdx @33 :UInt32;
  resetCalled @34 :Bool;
  nextSegIdx @35 :UInt32;

  cells @36 :List(Cell);
  segmentUpdates @37 :List(SegmentUpdate);

  ownsMemory @38 :Bool;
  lrnActiveStateT1 @39 :List(UInt8);
  lrnActiveStateT @40 :List(UInt8);
  lrnPredictedStateT1 @41 :List(UInt8);
  lrnPredictedStateT @42 :List(UInt8);
  infActiveStateT1 @43 :List(UInt8);
  infActiveStateT @44 :List(UInt8);
  infActiveStateBackup @45 :List(UInt8);
  infActiveStateCandidate @46 :List(UInt8);
  infPredictedStateT1 @47 :List(UInt8);
  infPredictedStateT @48 :List(UInt8);
  cellConfidenceT1 @49 :List(Float32);
  cellConfidenceT @50 :List(Float32);
  colConfidenceT1 @51 :List(Float32);
  colConfidenceT @52 :List(Float32);

  prevLrnPatterns @53 :List(List(UInt32));
  prevInfPatterns @54 :List(List(UInt32));

  statsNumInfersSinceReset @55 : UInt32;
  statsNumPredictions @56 : UInt32;
  statsCurPredictionScore2 @57 :Float32;
  statsPredictionScoreTotal2 @58 :Float32;
  statsCurFalseNegativeScore @59 :Float32;
  statsFalseNegativeScoreTotal @60 :Float32;
  statsCurFalsePositiveScore @61 :Float32;
  statsFalsePositiveScoreTotal @62 :Float32;
  statsPctExtraTotal @63 :Float32;
  statsPctMissingTotal @64 :Float32;
  statsCurMissing @65 : UInt32;
  statsCurExtra @66 : UInt32;
  statsTotalMissing @67 : UInt32;
  statsTotalExtra @68 : UInt32;
  statsPrevSequenceSignature @69 :List(Float32);
  statsConfHistogram @70 :List(List(Float32));

  # Next ID: 1
  struct Cell {
    segments @0 :List(Segment);
  }

  # Next ID: 10
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

  # Next ID: 2
  struct Synapse {
    srcCellIdx @0 :UInt64;
    permanence @1 :Float32;
  }

  # Next ID: 8
  struct SegmentUpdate {
    cellIdx @0 :UInt32;
    segIdx @1 :UInt32;
    sequenceSegment @2 :Bool;
    synapses @3 :List(UInt32);
    newSynapses @4 :List(UInt32);
    lrnIterationIdx @5 :UInt32;
    phase1 @6 :Bool;
    weaklyPredicting @7 :Bool;
  }
}
