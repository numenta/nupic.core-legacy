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
//  * Declaration of class DenseTensorUnitTest
//  */
//
// //----------------------------------------------------------------------
//
// #include "../math/SparseTensorUnitTest.hpp"
//
// //----------------------------------------------------------------------
//
// #ifndef NTA_DENSE_TENSOR_UNIT_TEST_HPP
// #define NTA_DENSE_TENSOR_UNIT_TEST_HPP
//
// namespace nupic {
//
//   //----------------------------------------------------------------------
//   class DenseTensorUnitTest : public Tester
//   {
//   public:
//     DenseTensorUnitTest() {}
//     virtual ~DenseTensorUnitTest() {}
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
//     typedef DenseTensor<I5, Real> D5;
//     typedef DenseTensor<I4, Real> D4;
//     typedef DenseTensor<I3, Real> D3;
//     typedef DenseTensor<I2, Real> D2;
//     typedef DenseTensor<I1, Real> D1;
//
//     //void unitTestConstructor();
//     //void unitTestGetSet();
//     //void unitTestIsSymmetric();
//     //void unitTestPermute();
//     //void unitTestResize();
//     //void unitTestReshape();
//     //void unitTestSlice();
//     //void unitTestElementApply();
//     //void unitTestFactorApply();
//     //void unitTestAccumulate();
//     //void unitTestOuterProduct();
//     //void unitTestContract();
//     //void unitTestInnerProduct();
//
//     // Default copy ctor and assignment operator forbidden by default
//     DenseTensorUnitTest(const DenseTensorUnitTest&);
//     DenseTensorUnitTest& operator=(const DenseTensorUnitTest&);
//
//   }; // end class DenseTensorUnitTest
//
//   //----------------------------------------------------------------------
// } // end namespace nupic
//
// #endif // NTA_DENSE_TENSOR_UNIT_TEST_HPP
//
//
//
