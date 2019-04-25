#include <vector>
#include <sstream>

#include "gtest/gtest.h"
#include <nupic/algorithms/AnomalyLikelihood.hpp>

namespace testing {

using namespace nupic;
using nupic::algorithms::anomaly::AnomalyLikelihood;

TEST(AnomalyLikelihood, SelectModeLikelihood)
{
  AnomalyLikelihood a;
  int ts = 0; //timestamp
  Real likelihood;
  for(int i=0; i< 388; i++) {
    likelihood = a.anomalyProbability(0.33f,  ++ts);
    ASSERT_FLOAT_EQ(likelihood, 0.5f); //first (<=388) probationaryPeriod rounds likelihood=0.5
  }

  //real likelihood returned here
  for(int i=0; i< 10; i++) {
    likelihood = a.anomalyProbability(0.33,  ++ts);
    ASSERT_TRUE(abs(likelihood - 0.5)<0.001); //TODO port likelihood tests here
  }
};

TEST(AnomalyLikelihood, SerializationLikelihood)
{
  AnomalyLikelihood a;
  int ts = 0; //timestamp
  Real likelihood = 0;
  for(int i=0; i< 400; i++) {
    likelihood = a.anomalyProbability(0.33f,  ++ts);
  }
  EXPECT_EQ(likelihood, 0.0f);
  
  AnomalyLikelihood b;
  std::stringstream ss;
  a.saveToStream_ar(ss);
  b.loadFromStream_ar(ss);
  EXPECT_EQ(a, b);
}

}
