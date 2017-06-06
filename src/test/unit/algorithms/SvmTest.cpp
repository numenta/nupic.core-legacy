/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

#include <fstream>
#include <stdio.h>
#include <vector>

#include <nupic/algorithms/Svm.hpp>

#include "gtest/gtest.h"

using namespace nupic;
using namespace nupic::algorithms::svm;

namespace {

template <typename T> void check_eq(std::vector<T> &v1, std::vector<T> &v2) {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_EQ(v1[i], v2[i]);
  }
}

template <typename T>
void check_eq(std::vector<T> &v1, std::vector<T> &v2, int ndims) {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    for (int j = 0; j < ndims; j++) {
      ASSERT_EQ(v1[i][j], v2[i][j]);
    }
  }
}

template <typename T>
void check_eq(std::vector<std::vector<T>> &v1,
              std::vector<std::vector<T>> &v2) {
  ASSERT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    check_eq(v1[i], v2[i]);
  }
}

// svm_parameter ---------------------------------------------------------------
void setup(svm_parameter &obj) {
  obj.kernel = 1;
  obj.probability = true;
  obj.gamma = 1.1;
  obj.C = 1.2;
  obj.eps = 1.3;
  obj.cache_size = 2;
  obj.shrinking = 3;
  obj.weight_label.push_back(1);
  obj.weight_label.push_back(2);
  obj.weight.push_back(1.1);
  obj.weight.push_back(1.2);
}

void check_eq(svm_parameter &obj1, svm_parameter &obj2) {
  ASSERT_EQ(obj1.kernel, obj2.kernel);
  ASSERT_EQ(obj1.probability, obj2.probability);
  ASSERT_EQ(obj1.gamma, obj2.gamma);
  ASSERT_EQ(obj1.C, obj2.C);
  ASSERT_EQ(obj1.eps, obj2.eps);
  ASSERT_EQ(obj1.cache_size, obj2.cache_size);
  check_eq(obj1.weight, obj2.weight);
  check_eq(obj2.weight_label, obj2.weight_label);
}

TEST(SvmTest, svm_parameter_testWriteRead) {
  const char *filename = "svm_parameter.bin";
  svm_parameter svm1(0, false, 0, 0, 0, 0, 0);
  svm_parameter svm2(0, false, 0, 0, 0, 0, 0);

  setup(svm1);

  std::ofstream os(filename, std::ios::binary);
  svm1.write(os);
  os.close();

  std::ifstream is(filename, std::ios::binary);
  svm2.read(is);
  is.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
}

// svm_problem -----------------------------------------------------------------
void setup(svm_problem &obj) {
  obj.recover_ = true;
  obj.n_dims_ = 2;
  obj.x_.push_back(new float[2]{2.2, 3.3});
  obj.x_.push_back(new float[2]{0.2, 13.3});
  obj.y_.push_back(4);
  obj.y_.push_back(14);
}

void check_eq(svm_problem &obj1, svm_problem &obj2) {
  ASSERT_EQ(obj1.recover_, obj2.recover_);
  ASSERT_EQ(obj1.n_dims_, obj2.n_dims_);
  check_eq(obj1.x_, obj2.x_, obj1.n_dims_);
  check_eq(obj1.y_, obj2.y_);
}

TEST(SvmTest, svm_problem_testWriteRead) {
  const char *filename = "svm_problem.bin";
  svm_problem svm1(0, false);
  svm_problem svm2(0, false);

  setup(svm1);

  std::ofstream fout(filename, std::ios::binary);
  svm1.write(fout);
  fout.close();

  std::ifstream fin(filename, std::ios::binary);
  svm2.read(fin);
  fin.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
}

// svm_problem01 ---------------------------------------------------------------
void setup(svm_problem01 &obj) {
  obj.recover_ = true;
  obj.n_dims_ = 2;
  obj.threshold_ = 2.2;
  obj.nnz_.push_back(6);
  obj.nnz_.push_back(62);
  obj.x_.push_back(new int[2]{3, 4});
  obj.x_.push_back(new int[2]{13, 41});
  obj.y_.push_back(5.5);
  obj.y_.push_back(52.5);
}

void check_eq(svm_problem01 &obj1, svm_problem01 &obj2) {
  ASSERT_EQ(obj1.recover_, obj2.recover_);
  ASSERT_EQ(obj1.n_dims_, obj2.n_dims_);
  ASSERT_EQ(obj1.threshold_, obj2.threshold_);
  check_eq(obj1.nnz_, obj2.nnz_);
  check_eq(obj1.x_, obj2.x_, obj2.n_dims_);
  check_eq(obj1.y_, obj2.y_);
}

TEST(SvmTest, svm_problem01_testWriteRead) {
  const char *filename = "svm_problem01.bin";
  svm_problem01 svm1(0, false);
  svm_problem01 svm2(0, false);

  setup(svm1);

  std::ofstream fout(filename, std::ios::binary);
  svm1.write(fout);
  fout.close();

  std::ifstream fin(filename, std::ios::binary);
  svm2.read(fin);
  fin.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
}

// svm_model -------------------------------------------------------------------
void setup(svm_model &obj) {
  obj.n_dims_ = 2;

  obj.sv.push_back(new float[2]{1.1, 1.2});
  obj.sv.push_back(new float[2]{3.1, 5.2});

  obj.sv_coef.push_back(new float[2]{5.1, 3.2});
  obj.sv_coef.push_back(new float[2]{6.1, 7.2});

  obj.w.resize(2);
  obj.w[0].push_back(33.1);
  obj.w[0].push_back(12.1);
  obj.w[1].push_back(3.1);
  obj.w[1].push_back(1.1);

  obj.rho.push_back(2.1);
  obj.rho.push_back(21.1);

  obj.label.push_back(3);
  obj.label.push_back(23);

  obj.n_sv.push_back(4);
  obj.n_sv.push_back(24);

  obj.probA.push_back(0.1);
  obj.probA.push_back(0.2);

  obj.probB.push_back(22.1);
  obj.probB.push_back(0.3);
  obj.probB.push_back(0.43);
}

void check_eq(svm_model &obj1, svm_model &obj2) {
  ASSERT_EQ(obj1.n_dims_, obj2.n_dims_);
  check_eq(obj1.sv, obj2.sv, obj2.n_dims_);
  check_eq(obj1.sv_coef, obj2.sv_coef, obj2.n_dims_);
  check_eq(obj1.rho, obj2.rho);
  check_eq(obj1.label, obj2.label);
  check_eq(obj1.n_sv, obj2.n_sv);
  check_eq(obj1.probA, obj2.probA);
  check_eq(obj1.probB, obj2.probB);
  check_eq(obj1.w, obj2.w);
}

TEST(SvmTest, svm_model_testWriteRead) {
  const char *filename = "svm_model.bin";
  svm_model svm1, svm2;

  setup(svm1);

  std::ofstream fout(filename, std::ios::binary);
  svm1.write(fout);
  fout.close();

  std::ifstream fin(filename, std::ios::binary);
  svm2.read(fin);
  fin.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;
}

// svm_dense -------------------------------------------------------------------
void check_eq(svm_dense &obj1, svm_dense &obj2) {
  check_eq(obj1.get_parameter(), obj2.get_parameter());
  check_eq(obj1.get_problem(), obj2.get_problem());
  check_eq(obj1.get_model(), obj2.get_model());
}

TEST(SvmTest, svm_dense_testWriteRead) {
  const char *filename = "svm_dense.bin";
  svm_dense svm1, svm2;

  svm1.add_sample(0.5, new float[1]{11.0});
  svm1.train(1.1, 2.2, 3.3);

  std::ofstream fout(filename, std::ios::binary);
  svm1.write(fout);
  fout.close();

  std::ifstream fin(filename, std::ios::binary);
  svm2.read(fin);
  fin.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));

  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

  svm1.add_sample(0.75, new float[1]{7.0});
  svm1.train(7.1, 7.2, 7.3);
  svm2.add_sample(0.75, new float[1]{7.0});
  svm2.train(7.1, 7.2, 7.3);
  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
}

// svm_01 ----------------------------------------------------------------------
void check_eq(svm_01 &obj1, svm_01 &obj2) {
  check_eq(obj1.get_parameter(), obj2.get_parameter());
  check_eq(obj1.get_problem(), obj2.get_problem());
  check_eq(obj1.get_model(), obj2.get_model());
}

TEST(SvmTest, svm_01_testWriteRead) {
  const char *filename = "svm_01.bin";
  svm_01 svm1, svm2;

  svm1.add_sample(0.5, new float[1]{11.0});
  svm1.train(1.1, 2.2, 3.3);

  std::ofstream fout(filename, std::ios::binary);
  svm1.write(fout);
  fout.close();

  std::ifstream fin(filename, std::ios::binary);
  svm2.read(fin);
  fin.close();

  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));

  int ret = ::remove(filename);
  ASSERT_TRUE(ret == 0) << "Failed to delete " << filename;

  svm1.add_sample(0.75, new float[1]{7.0});
  svm1.train(7.1, 7.2, 7.3);
  svm2.add_sample(0.75, new float[1]{7.0});
  svm2.train(7.1, 7.2, 7.3);
  ASSERT_NO_FATAL_FAILURE(check_eq(svm1, svm2));
}
} // end anonymous namespace
