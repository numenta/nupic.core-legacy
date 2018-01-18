/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

/*
Copyright (c) 2000-2007 Chih-Chung Chang and Chih-Jen Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <sstream>

#include <nupic/algorithms/Svm.hpp>

namespace nupic {
namespace algorithms {
namespace svm {

using namespace std;

//------------------------------------------------------------------------------
void svm_parameter::print() const {
  std::cout << "kernel = " << kernel << std::endl
            << "probability = " << probability << std::endl
            << "gamma = " << gamma << std::endl
            << "C = " << C << std::endl
            << "eps = " << eps << std::endl
            << "cache_size = " << cache_size << std::endl
            << "shrinking = " << shrinking << std::endl;
}

//------------------------------------------------------------------------------
int svm_parameter::persistent_size() const {
  stringstream b;
  b << kernel << ' ' << probability << ' ' << gamma << ' ' << C << ' ' << eps
    << ' ' << cache_size << ' ' << shrinking << ' ' << weight_label << ' '
    << weight << ' ';

  return b.str().size();
}

//------------------------------------------------------------------------------
void svm_parameter::save(std::ostream &outStream) const {
  outStream << kernel << ' ' << probability << ' ' << gamma << ' ' << C << ' '
            << eps << ' ' << cache_size << ' ' << shrinking << ' '
            << weight_label << ' ' << weight << ' ';
}

//------------------------------------------------------------------------------
void svm_parameter::load(std::istream &inStream) {
  inStream >> kernel >> probability >> gamma >> C >> eps >> cache_size >>
      shrinking >> weight_label >> weight;
}

//------------------------------------------------------------------------------
void svm_parameter::read(SvmParameterProto::Reader &proto) {
  kernel = proto.getKernel();
  probability = proto.getProbability();
  gamma = proto.getGamma();
  C = proto.getC();
  eps = proto.getEps();
  cache_size = proto.getCacheSize();
  shrinking = proto.getShrinking();

  auto weightList = proto.getWeight();
  size_t size = weightList.size();
  weight.resize(size);
  for (size_t i = 0; i < size; i++) {
    weight[i] = weightList[i];
  }

  auto labelList = proto.getWeightLabel();
  size = labelList.size();
  weight_label.resize(size);
  for (size_t i = 0; i < size; i++) {
    weight_label[i] = labelList[i];
  }
}

//------------------------------------------------------------------------------
void svm_parameter::write(SvmParameterProto::Builder &proto) const {
  proto.setKernel(kernel);
  proto.setProbability(probability);
  proto.setGamma(gamma);
  proto.setC(C);
  proto.setEps(eps);
  proto.setCacheSize(cache_size);
  proto.setShrinking(shrinking);

  size_t size = weight.size();
  auto weightList = proto.initWeight(size);
  for (size_t i = 0; i < size; i++) {
    weightList.set(i, weight[i]);
  }

  size = weight_label.size();
  auto labelList = proto.initWeightLabel(size);
  for (size_t i = 0; i < size; i++) {
    labelList.set(i, weight_label[i]);
  }
}

//------------------------------------------------------------------------------
int svm_problem::persistent_size() const {
  stringstream b;

  b << size() << " " << n_dims() << " ";

  return b.str().size() + y_.size() * sizeof(label_type) +
         size() * n_dims() * sizeof(feature_type) + 1;
}

//------------------------------------------------------------------------------
void svm_problem::save(std::ostream &outStream) const {
  outStream << size() << " " << n_dims() << " ";

  nupic::binary_save(outStream, y_);

  for (int i = 0; i < size(); ++i)
    nupic::binary_save(outStream, x_[i], x_[i] + n_dims());
  outStream << " ";
}

//------------------------------------------------------------------------------
void svm_problem::load(std::istream &inStream) {
  int s = 0;
  inStream >> s >> n_dims_;

  if (recover_)
    for (size_t i = 0; i != x_.size(); ++i)
      delete[] x_[i];

  y_.resize(s, 0);
  x_.resize(s, nullptr);

  inStream.ignore(1);
  nupic::binary_load(inStream, y_);

  for (int i = 0; i < size(); ++i) {
#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
    x_[i] = (float *)_aligned_malloc(4 * n_dims(), 16);
#else
    x_[i] = new feature_type[n_dims()];
#endif

    std::fill(x_[i], x_[i] + n_dims(), (float)0);
    nupic::binary_load(inStream, x_[i], x_[i] + n_dims());
  }
}

//------------------------------------------------------------------------------
void svm_problem::read(SvmProblemProto::Reader &proto) {
  recover_ = proto.getRecover();
  n_dims_ = proto.getNDims();

  auto yList = proto.getY();
  size_t size = yList.size();
  y_.resize(size);
  for (size_t i = 0; i < size; i++) {
    y_[i] = yList[i];
  }

  for (auto &elem : x_)
    delete[] elem;

  x_.clear();
  for (auto list : proto.getX()) {
    size_t size = list.size();
    float *values = new float[size];
    for (size_t i = 0; i < size; i++) {
      values[i] = list[i];
    }
    x_.push_back(values);
  }
}

//------------------------------------------------------------------------------
void svm_problem::write(SvmProblemProto::Builder &proto) const {
  proto.setRecover(recover_);
  proto.setNDims(n_dims_);

  size_t size = y_.size();
  auto yList = proto.initY(size);
  for (size_t i = 0; i < size; i++) {
    yList.set(i, y_[i]);
  }

  size = x_.size();
  auto xList = proto.initX(size);
  for (size_t i = 0; i < size; i++) {
    auto dims = xList.init(i, n_dims_);
    for (int j = 0; j < n_dims_; j++) {
      dims.set(j, x_[i][j]);
    }
  }
}

//------------------------------------------------------------------------------
int svm_problem01::persistent_size() const {
  stringstream b;
  b << size() << " " << n_dims() << " " << threshold_ << " ";
  int n = b.str().size();

  n += y_.size() * sizeof(float);
  n += nnz_.size() * sizeof(int);

  for (int i = 0; i != size(); ++i)
    n += nnz_[i] * sizeof(feature_type);

  return n + 1;
}

//------------------------------------------------------------------------------
void svm_problem01::save(std::ostream &outStream) const {
  outStream << size() << " " << n_dims() << " " << threshold_ << " ";

  nupic::binary_save(outStream, y_);
  nupic::binary_save(outStream, nnz_);

  for (int i = 0; i < size(); ++i)
    nupic::binary_save(outStream, x_[i], x_[i] + nnz_[i]);
  outStream << " ";
}

//------------------------------------------------------------------------------
void svm_problem01::load(std::istream &inStream) {
  int s = 0;
  inStream >> s >> n_dims_ >> threshold_;

  if (recover_)
    for (auto &elem : x_)
      delete[] elem;

  y_.resize(s, 0);
  nnz_.resize(s, 0);
  x_.resize(s, nullptr);

  inStream.ignore(1);
  nupic::binary_load(inStream, y_);
  nupic::binary_load(inStream, nnz_);

  for (int i = 0; i < s; ++i) {
    x_[i] = new feature_type[nnz_[i]];
    nupic::binary_load(inStream, x_[i], x_[i] + nnz_[i]);
  }
}

//------------------------------------------------------------------------------
void svm_problem01::read(SvmProblem01Proto::Reader &proto) {
  recover_ = proto.getRecover();
  n_dims_ = proto.getNDims();
  threshold_ = proto.getThreshold();

  auto yList = proto.getY();
  size_t size = yList.size();
  y_.resize(size);
  for (size_t i = 0; i < size; i++) {
    y_[i] = yList[i];
  }

  auto nnzList = proto.getNnz();
  nnz_.resize(nnzList.size());
  size = nnzList.size();
  nnz_.resize(size);
  for (size_t i = 0; i < size; i++) {
    nnz_[i] = nnzList[i];
  }

  for (auto &elem : x_)
    delete[] elem;

  x_.clear();
  for (auto list : proto.getX()) {
    size_t size = list.size();
    int *values = new int[size];
    for (size_t i = 0; i < size; i++) {
      values[i] = list[i];
    }
    x_.push_back(values);
  }
}

//------------------------------------------------------------------------------
void svm_problem01::write(SvmProblem01Proto::Builder &proto) const {
  proto.setRecover(recover_);
  proto.setNDims(n_dims_);
  proto.setThreshold(threshold_);

  size_t size = y_.size();
  auto yList = proto.initY(size);
  for (size_t i = 0; i < size; i++) {
    yList.set(i, y_[i]);
  }

  size = nnz_.size();
  auto nnzList = proto.initNnz(size);
  for (size_t i = 0; i < size; i++) {
    nnzList.set(i, nnz_[i]);
  }

  size = x_.size();
  auto xList = proto.initX(size);
  for (size_t i = 0; i < size; i++) {
    auto dims = xList.init(i, n_dims_);
    for (int j = 0; j < n_dims_; j++) {
      dims.set(j, x_[i][j]);
    }
  }
}

//------------------------------------------------------------------------------
svm_model::~svm_model() {
  // in all cases, ownership of the mem for the sv is with svm_model

  if (sv_mem == nullptr) {
    for (size_t i = 0; i != sv.size(); ++i)

#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
      _aligned_free(sv[i]);
#else
      delete[] sv[i];
#endif

  } else {
#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
    _aligned_free(sv_mem);
#else
    delete[] sv_mem;
#endif

    sv_mem = nullptr;
    sv.clear();
  }

  for (size_t i = 0; i != sv_coef.size(); ++i)
    delete[] sv_coef[i];
}

//------------------------------------------------------------------------------
void svm_model::print() const {
  std::cout << "n classes = " << n_class() << " n sv = " << size()
            << " n dims = " << n_dims() << std::endl;

  std::cout << "Support vectors: " << std::endl;
  for (size_t i = 0; i != sv.size(); ++i) {
    for (int j = 0; j != n_dims(); ++j)
      std::cout << sv[i][j] << " ";
    std::cout << std::endl;
  }

  std::cout << "Support vector coefficients: " << std::endl;
  for (size_t i = 0; i != sv_coef.size(); ++i) {
    for (int j = 0; j != size(); ++j)
      std::cout << sv_coef[i][j] << " ";
    std::cout << std::endl;
  }

  std::cout << "Rho: " << std::endl;
  for (size_t i = 0; i != rho.size(); ++i)
    std::cout << rho[i] << " ";
  std::cout << std::endl;

  if (!probA.empty()) {
    std::cout << "Probabilities A: " << std::endl;
    for (size_t i = 0; i != probA.size(); ++i)
      std::cout << probA[i] << " ";
    std::cout << std::endl;

    std::cout << "Probabilities B: " << std::endl;
    for (size_t i = 0; i != probB.size(); ++i)
      std::cout << probB[i] << " ";
    std::cout << std::endl;
  }
}

//------------------------------------------------------------------------------
int svm_model::persistent_size() const {
  stringstream b;
  b << n_class() << " " << size() << " " << n_dims() << " ";

  int n = b.str().size();

  n += sv.size() * n_dims() * sizeof(float) + 1;

  {
    stringstream b2;
    for (auto &elem : sv_coef) {
      for (int j = 0; j < size(); ++j)
        b2 << elem[j] << " ";
    }
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << rho << " ";
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << label << " ";
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << n_sv << " ";
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << probA << " ";
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << probB << " ";
    n += b2.str().size();
  }

  {
    stringstream b2;
    b2 << w << " ";
    n += b2.str().size();
  }

  return n;
}

//------------------------------------------------------------------------------
void svm_model::save(std::ostream &outStream) const {
  outStream << n_class() << " " << size() << " " << n_dims() << " ";

  for (auto &elem : sv)
    nupic::binary_save(outStream, elem, elem + n_dims());
  outStream << " ";

  for (auto &elem : sv_coef)
    for (int j = 0; j < size(); ++j)
      outStream << elem[j] << " ";

  outStream << rho << ' ' << label << ' ' << n_sv << ' ' << probA << ' '
            << probB << ' ' << w << ' ';
}

//------------------------------------------------------------------------------
void svm_model::load(std::istream &inStream) {
  int n_class = 0, l = 0;
  inStream >> n_class >> l >> n_dims_;

  if (sv_mem == nullptr) {
    for (auto &elem : sv)
      delete[] elem;

  } else {
    delete[] sv_mem;
    sv_mem = nullptr;
  }

#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
  sv_mem = (float *)_aligned_malloc(4 * l * n_dims(), 16);
#else
  sv_mem = new float[l * n_dims()];
#endif

  std::fill(sv_mem, sv_mem + l * n_dims(), (float)0);

  sv.resize(l, nullptr);
  inStream.ignore(1);
  for (int i = 0; i < l; ++i) {
    sv[i] = sv_mem + i * n_dims();
    nupic::binary_load(inStream, sv[i], sv[i] + n_dims());
  }

  for (auto &elem : sv_coef)
    delete[] elem;

  sv_coef.resize(n_class - 1, nullptr);
  for (int i = 0; i < n_class - 1; ++i) {
    sv_coef[i] = new float[l];
    for (int j = 0; j < l; ++j)
      inStream >> sv_coef[i][j];
  }

  inStream >> rho >> label >> n_sv >> probA >> probB >> w;
}

//------------------------------------------------------------------------------
void svm_model::read(SvmModelProto::Reader &proto) {
  n_dims_ = proto.getNDims();

  if (sv_mem == nullptr) {
    for (auto &elem : sv)
      delete[] elem;
  } else {
    delete[] sv_mem;
    sv_mem = nullptr;
  }
  sv.clear();

  for (auto list : proto.getSv()) {
    size_t size = list.size();
    float *values = new float[size];
    for (size_t i = 0; i < size; i++) {
      values[i] = list[i];
    }
    sv.push_back(values);
  }

  for (auto &elem : sv_coef)
    delete[] elem;
  sv_coef.clear();

  for (auto list : proto.getSvCoef()) {
    size_t size = list.size();
    float *values = new float[size];
    for (size_t i = 0; i < size; i++) {
      values[i] = list[i];
    }
    sv_coef.push_back(values);
  }

  auto wList = proto.getW();
  size_t size = wList.size();
  w.resize(size);
  for (size_t i = 0; i < size; i++) {
    auto values = wList[i];
    size_t len = values.size();
    w[i].resize(len);
    for (size_t j = 0; j < len; j++) {
      w[i][j] = values[j];
    }
  }

  auto rhoList = proto.getRho();
  size = rhoList.size();
  rho.resize(size);
  for (size_t i = 0; i < size; i++) {
    rho[i] = rhoList[i];
  }

  auto probAList = proto.getProbA();
  size = probAList.size();
  probA.resize(size);
  for (size_t i = 0; i < size; i++) {
    probA[i] = probAList[i];
  }

  auto probBList = proto.getProbB();
  size = probBList.size();
  probB.resize(size);
  for (size_t i = 0; i < size; i++) {
    probB[i] = probBList[i];
  }

  auto labelList = proto.getLabel();
  size = labelList.size();
  label.resize(size);
  for (size_t i = 0; i < size; i++) {
    label[i] = labelList[i];
  }

  auto nsvList = proto.getNSv();
  size = nsvList.size();
  n_sv.resize(size);
  for (size_t i = 0; i < size; i++) {
    n_sv[i] = nsvList[i];
  }
}

//------------------------------------------------------------------------------
void svm_model::write(SvmModelProto::Builder &proto) const {

  proto.setNDims(n_dims_);

  size_t size = sv.size();
  auto svList = proto.initSv(size);
  for (size_t i = 0; i < size; i++) {
    auto dims = svList.init(i, n_dims_);
    for (int j = 0; j < n_dims_; j++) {
      dims.set(j, sv[i][j]);
    }
  }

  size = sv_coef.size();
  auto svCoefList = proto.initSvCoef(size);
  for (size_t i = 0; i < size; i++) {
    auto dims = svCoefList.init(i, n_dims_);
    for (int j = 0; j < n_dims_; j++) {
      dims.set(j, sv_coef[i][j]);
    }
  }

  size = rho.size();
  auto rhoList = proto.initRho(size);
  for (size_t i = 0; i < size; i++) {
    rhoList.set(i, rho[i]);
  }

  size = label.size();
  auto labelList = proto.initLabel(size);
  for (size_t i = 0; i < size; i++) {
    labelList.set(i, label[i]);
  }

  size = n_sv.size();
  auto nsvList = proto.initNSv(size);
  for (size_t i = 0; i < size; i++) {
    nsvList.set(i, n_sv[i]);
  }

  size = probA.size();
  auto probAList = proto.initProbA(size);
  for (size_t i = 0; i < size; i++) {
    probAList.set(i, probA[i]);
  }

  size = probB.size();
  auto probBList = proto.initProbB(size);
  for (size_t i = 0; i < size; i++) {
    probBList.set(i, probB[i]);
  }

  size = w.size();
  auto wList = proto.initW(size);
  for (size_t i = 0; i < size; i++) {
    size_t len = w[i].size();
    auto dims = wList.init(i, len);
    for (size_t j = 0; j < len; j++) {
      dims.set(j, w[i][j]);
    }
  }
}

//------------------------------------------------------------------------------
void svm_dense::write(SvmDenseProto::Builder &proto) const {
  auto paramProto = proto.getParam();
  svm_.param_.write(paramProto);

  if (svm_.model_) {
    auto modelProto = proto.getModel();
    svm_.model_->write(modelProto);
  }
  if (svm_.problem_) {
    auto problemProto = proto.getProblem();
    svm_.problem_->write(problemProto);
  }
}

//------------------------------------------------------------------------------
void svm_dense::read(SvmDenseProto::Reader &proto) {
  auto paramProto = proto.getParam();
  svm_.param_.read(paramProto);

  if (svm_.model_) {
    delete svm_.model_;
    svm_.model_ = nullptr;
  }
  if (proto.hasModel()) {
    auto modelProto = proto.getModel();
    svm_.model_ = new svm_model;
    svm_.model_->read(modelProto);
  }
  if (svm_.problem_) {
    delete svm_.problem_;
    svm_.problem_ = nullptr;
  }
  if (proto.hasProblem()) {
    auto problemProto = proto.getProblem();
    svm_.problem_ = new svm_problem(1, false);
    svm_.problem_->read(problemProto);
  }
}

//------------------------------------------------------------------------------
void svm_01::write(Svm01Proto::Builder &proto) const {
  auto paramProto = proto.getParam();
  svm_.param_.write(paramProto);

  if (svm_.model_) {
    auto modelProto = proto.getModel();
    svm_.model_->write(modelProto);
  }
  if (svm_.problem_) {
    auto problemProto = proto.getProblem();
    svm_.problem_->write(problemProto);
  }
}

//------------------------------------------------------------------------------
void svm_01::read(Svm01Proto::Reader &proto) {
  auto paramProto = proto.getParam();
  svm_.param_.read(paramProto);

  if (svm_.model_) {
    delete svm_.model_;
    svm_.model_ = nullptr;
  }
  if (proto.hasModel()) {
    auto modelProto = proto.getModel();
    svm_.model_ = new svm_model;
    svm_.model_->read(modelProto);
  }
  if (svm_.problem_) {
    delete svm_.problem_;
    svm_.problem_ = nullptr;
  }
  if (proto.hasProblem()) {
    auto problemProto = proto.getProblem();
    svm_.problem_ = new svm_problem01(1, false);
    svm_.problem_->read(problemProto);
  }
}
} // namespace svm
} // namespace algorithms
} // namespace nupic
