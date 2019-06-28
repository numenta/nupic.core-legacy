#include "gtest/gtest.h"
#include "htm/utils/SlidingWindow.hpp"

namespace testing { 
    
using htm::util::SlidingWindow;


TEST(SlidingWindow, Instance)
{
  SlidingWindow<int>  w{3, "test", 1};
  const std::vector<int> iv{1,2,3};
  const SlidingWindow<int> w2{3, std::begin(iv), std::end(iv)};

    ASSERT_EQ(w.size(), 0);
    ASSERT_EQ(w.ID, "test");
    ASSERT_EQ(w.DEBUG, 1);
    ASSERT_EQ(w2.size(), 3);
    ASSERT_TRUE(w.maxCapacity == w2.maxCapacity ); // ==3
    w.append(4);
    ASSERT_EQ(w.size(), 1);
    ASSERT_EQ(w.getData(), w.getLinearizedData());
    w.append(1);
    ASSERT_EQ(w[1], w2[0]); //==1
    w.append(2);
    ASSERT_NE(w, w2);
    int ret = -1;
    w.append(3, &ret);
    ASSERT_EQ(ret, 4);
    ASSERT_EQ(w.getLinearizedData(), w2.getLinearizedData());
    ASSERT_EQ(w, w2);
    ASSERT_NE(w.getData(), w2.getData()); // linearized data are same, but internal buffer representations are not
}
}