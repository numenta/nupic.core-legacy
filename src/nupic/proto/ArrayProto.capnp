@0xe06bf2673760a815;

# Represents an Output Array
struct ArrayProto {
  union {
    byteArray   @0 :List(UInt8);   # NTA_BasicType_Byte; picking UInt8 due to lack of char type in capnproto schema
    int16Array  @1 :List(Int16);   # NTA_BasicType_Int16
    uint16Array @2 :List(UInt16);  # NTA_BasicType_UInt16
    int32Array  @3 :List(Int32);   # NTA_BasicType_Int32
    uint32Array @4 :List(UInt32);  # NTA_BasicType_UInt32
    int64Array  @5 :List(Int64);   # NTA_BasicType_Int64
    uint64Array @6 :List(UInt64);  # NTA_BasicType_UInt64
    real32Array @7 :List(Float32); # NTA_BasicType_Real32
    real64Array @8 :List(Float64); # NTA_BasicType_Real64
  }
}
