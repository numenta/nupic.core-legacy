@0xcf9698b0f93fc8e4;

using import "/nupic/proto/Segment.capnp".SegmentProto;

struct CellProto {
  segments @0 :List(SegmentProto);
}
