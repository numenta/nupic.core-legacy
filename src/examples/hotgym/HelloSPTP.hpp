// header file for Hotgym, HelloSPTP
#ifndef NTA_EXAMPLES_HOTGYM_
#define NTA_EXAMPLES_HOTGYM_

#include <nupic/types/Types.hpp>
#include <nupic/os/Timer.hpp>

namespace examples {

using nupic::Real64;
using nupic::UInt;
using nupic::Timer;

class BenchmarkHotgym {

public:	
  Real64 run(
    UInt EPOCHS = 5000,
    bool useSPlocal=true, //can toggle which (long running) components are tested, default all
    bool useSPglobal=true,
    bool useTP=true,
    bool useTM=true,
    const UInt COLS = 2048, // number of columns in SP, TP
    const UInt DIM_INPUT = 10000,
    const UInt CELLS = 10 // cells per column in TP
  );

  //timers
  Timer tInit, tAll, tRng, tEnc, tSPloc, tSPglob, tTP, tTM,
        tAn, tAnLikelihood;

};

} //-ns
#endif //header
