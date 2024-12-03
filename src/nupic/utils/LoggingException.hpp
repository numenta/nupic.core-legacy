/*
 * Copyright 2013 Numenta Inc.
 *
 * Copyright may exist in Contributors' modifications
 * and/or contributions to the work.
 *
 * Use of this source code is governed by the MIT
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */

/** @file */

//----------------------------------------------------------------------

#ifndef NTA_LOGGING_EXCEPTION_HPP
#define NTA_LOGGING_EXCEPTION_HPP

#include <nupic/types/Exception.hpp>
#include <sstream>
#include <vector>

namespace nupic {
class LoggingException : public Exception {
public:
  LoggingException(const std::string &filename, UInt32 lineno)
      : Exception(filename, lineno, std::string()), ss_(std::string()),
        lmessageValid_(false), alreadyLogged_(false) {}

  virtual ~LoggingException() throw();

  const char *getMessage() const override {
    // Make sure we use a persistent string. Otherwise the pointer may
    // become invalid.
    // If the underlying stringstream object hasn't changed, don't regenerate
    // lmessage_. This is important because if we catch this exception a second
    // call to exception.what() will trash the buffer returned by a first call
    // to exception.what()
    if (!lmessageValid_) {
      lmessage_ = ss_.str();
      lmessageValid_ = true;
    }
    return lmessage_.c_str();
  }

  // for Index.hpp: // because stringstream cant take << vector
  LoggingException &
  operator<<(std::vector<nupic::UInt32, std::allocator<nupic::UInt32>> v) {
    lmessageValid_ = false;
    ss_ << "[";
    for (auto &elem : v)
      ss_ << elem << " ";
    ss_ << "]";
    return *this;
  }

  template <typename T> LoggingException &operator<<(const T &obj) {
    // underlying stringstream changes, so let getMessage() know
    // to regenerate lmessage_
    lmessageValid_ = false;
    ss_ << obj;
    return *this;
  }

  LoggingException(const LoggingException &l)
      : Exception(l), ss_(l.ss_.str()), lmessage_(""), lmessageValid_(false),
        alreadyLogged_(true) // copied exception does not log

  {
    // make sure message string is up to date for debuggers.
    getMessage();
  }

private:
  std::stringstream ss_;
  mutable std::string lmessage_; // mutable because getMesssage() modifies it
  mutable bool lmessageValid_;
  bool alreadyLogged_;
}; // class LoggingException

} // namespace nupic

#endif // NTA_LOGGING_EXCEPTION_HPP
