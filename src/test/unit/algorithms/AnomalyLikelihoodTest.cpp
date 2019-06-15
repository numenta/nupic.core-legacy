#include <vector>
#include <sstream>

#include "gtest/gtest.h"
#include <htm/algorithms/AnomalyLikelihood.hpp>

namespace testing {

using namespace htm;

TEST(DISABLED_AnomalyLikelihood, SelectModeLikelihood)
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
    likelihood = a.anomalyProbability(0.33f,  ++ts);
  }
  ASSERT_FLOAT_EQ(likelihood, 0.1f); //TODO port likelihood tests here
};

TEST(DISABLED_AnomalyLikelihood, SerializationLikelihood)
{
  AnomalyLikelihood a;
  int ts = 0; //timestamp
  Real likelihood = 0;
  for(int i=0; i< 400; i++) {
    likelihood = a.anomalyProbability(0.33f,  ++ts);
  }
  EXPECT_FLOAT_EQ(likelihood, 0.0f);
  
  AnomalyLikelihood b;
  std::stringstream ss;
  a.save(ss);
  b.load(ss);
  EXPECT_EQ(a, b);
}

}
