#include "examples/HelloSPTP.cpp"
#include <string> // stoi

//this runs as executable
int main(int argc, char* argv[]) {
  using namespace examples;
  
  UInt EPOCHS = 5000; // number of iterations (calls to SP/TP compute() )

  if(argc == 2) {
    EPOCHS = std::stoi(argv[1]);
  }

  run(EPOCHS);
  return 0;
}
