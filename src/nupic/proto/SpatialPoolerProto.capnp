@0xaf55f2dcb92e4304;

# TODO: Use absolute path
using import "/nupic/proto/SparseBinaryMatrixProto.capnp".SparseBinaryMatrixProto;
using import "/nupic/proto/SparseMatrixProto.capnp".SparseMatrixProto;
using import "/nupic/proto/RandomProto.capnp".RandomProto;

# Next ID: 37
struct SpatialPoolerProto {
  random @0 :RandomProto;
  numInputs @1 :UInt32;
  numColumns @2 :UInt32;
  columnDimensions @3 :List(UInt32);
  inputDimensions @4 :List(UInt32);
  potentialRadius @5 :UInt32;
  potentialPct @6 :Float32;
  inhibitionRadius @35 :UInt32;
  globalInhibition @7 :Bool;
  numActiveColumnsPerInhArea @8 :UInt32;
  localAreaDensity @9 :Float32;
  stimulusThreshold @10 :UInt32;
  synPermInactiveDec @11 :Float32;
  synPermActiveInc @12 :Float32;
  synPermBelowStimulusInc @13 :Float32;
  synPermConnected @14 :Float32;
  minPctOverlapDutyCycles @15 :Float32;
  minPctActiveDutyCycles @16 :Float32;
  dutyCyclePeriod @17 :UInt16;
  maxBoost @18 :Float32;
  wrapAround @36 :Bool;
  spVerbosity @19 :UInt8;

  synPermMin @20 :Float32;
  synPermMax @21 :Float32;
  synPermTrimThreshold @22 :Float32;
  updatePeriod @23 :UInt16;

  version @24 :UInt16;
  iterationNum @25 :UInt32;
  iterationLearnNum @26 :UInt32;

  # List length equals number of columns, elements are indices of input bits
  # in potential pool
  potentialPools @27 :SparseBinaryMatrixProto;

  # List length equals number of columns, elements are SparseFloat instances with
  # an input bit index and the permanence value for all non-zero permanences.
  permanences @28 :SparseMatrixProto;

  # Tie break float values for each column to break ties
  tieBreaker @29 :List(Float32);

  overlapDutyCycles @30 :List(Float32);
  activeDutyCycles @31 :List(Float32);
  minOverlapDutyCycles @32 :List(Float32);
  minActiveDutyCycles @33 :List(Float32);
  boostFactors @34 :List(Float32);
}
