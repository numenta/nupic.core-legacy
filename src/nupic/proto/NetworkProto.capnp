@0xbec20a8f487acf07;

using import "/nupic/proto/LinkProto.capnp".LinkProto;
using import "/nupic/proto/Map.capnp".Map;
using import "/nupic/proto/RegionProto.capnp".RegionProto;

struct NetworkProto {
  regions @0 :Map(Text, RegionProto);
  links @1 :List(LinkProto);
}
