@0x9baf2c52ef38ffb0;

# Next ID: 9
struct SvmParameterProto {
  kernel @0 :Int32;
  probability @1 :Bool;
  gamma @2 :Float32;
  c @3 :Float32;
  eps @4 :Float32;
  cacheSize @5 :Int32;
  shrinking @6 :Int32;
  weightLabel @7 :List(Int32);
  weight @8 :List(Float32);
}

# Next ID: 4
struct SvmProblemProto {
  recover @0 :Bool;
  nDims @1 :Int32;
  x @2 :List(List(Float32));
  y @3 :List(Float32); 
}

# Next ID: 6
struct SvmProblem01Proto {
  recover @0 :Bool;
  nDims @1 :Int32;
  threshold @2 :Float32;
  nnz @3 :List(Int32);
  x @4 :List(List(Int32));
  y @5 :List(Float32);
}

# Next ID: 9
struct SvmModelProto {
  nDims @0 :Int32;
  sv @1 :List(List(Float32));
  svCoef @2 :List(List(Float32));
  rho @3 :List(Float32);
  label @4 :List(Int32);
  nSv @5 :List(Int32);
  probA @6 :List(Float32);
  probB @7 :List(Float32);
  w @8 :List(List(Float32));
}

# Next ID: 3
struct SvmDenseProto {
  model @0 :SvmModelProto;
  param @1 :SvmParameterProto;
  problem @2 :SvmProblemProto;
}

# Next ID: 3
struct Svm01Proto {
  model @0 :SvmModelProto;
  param @1 :SvmParameterProto;
  problem @2 :SvmProblem01Proto;
}
