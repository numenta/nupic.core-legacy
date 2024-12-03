// /*
//  * Copyright 2013 Numenta Inc.
//  *
//  * Copyright may exist in Contributors' modifications
//  * and/or contributions to the work.
//  *
//  * Use of this source code is governed by the MIT
//  * license that can be found in the LICENSE file or at
//  * https://opensource.org/licenses/MIT.
//  */
//
// /** @file
//  * Declaration of class IndexUnitTest
//  */
//
// //----------------------------------------------------------------------
//
// #include <nupic/test/Tester.hpp>
// #include <nupic/math/Index.hpp>
//
// //----------------------------------------------------------------------
//
// #ifndef NTA_INDEX_UNIT_TEST_HPP
// #define NTA_INDEX_UNIT_TEST_HPP
//
// namespace nupic {
//
//   //----------------------------------------------------------------------
//   class IndexUnitTest : public Tester
//   {
//   public:
//     IndexUnitTest() {}
//     virtual ~IndexUnitTest() {}
//
//     // Run all appropriate tests
//     virtual void RunTests() override;
//
//   private:
//     typedef Index<UInt, 1> I1;
//     typedef Index<UInt, 2> I2;
//     typedef Index<UInt, 3> I3;
//     typedef Index<UInt, 4> I4;
//     typedef Index<UInt, 5> I5;
//     typedef Index<UInt, 6> I6;
//
//     //void unitTestFixedIndex();
//     //void unitTestDynamicIndex();
//
//     // Default copy ctor and assignment operator forbidden by default
//     IndexUnitTest(const IndexUnitTest&);
//     IndexUnitTest& operator=(const IndexUnitTest&);
//
//   }; // end class IndexUnitTest
//
//   //----------------------------------------------------------------------
// } // end namespace nupic
//
// #endif // NTA_INDEX_UNIT_TEST_HPP
//
//
//
