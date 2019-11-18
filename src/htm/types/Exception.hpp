/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2013, Numenta, Inc.
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
 * --------------------------------------------------------------------- */

/** @file */

//----------------------------------------------------------------------

#ifndef NTA_EXCEPTION_HPP
#define NTA_EXCEPTION_HPP

#include <htm/types/Types.hpp>
#include <htm/os/Path.hpp>

#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

//----------------------------------------------------------------------

namespace htm {
/**
 * @b Responsibility
 *  The Exception class is the standard Numenta exception class.
 *  It is responsible for storing rich error information:
 *  the filename and line number where the exceptional situation
 *  occured and a message that describes the exception.
 *
 * @b Rationale:
 *  It is important to store the location (filename, line number) where
 *  an exception originated, but no standard excepton class does it.
 *  The stack trace is even better and brings C++ programming to the
 *  ease of use of languages like Java and C#.
 *
 * @b Usage:
 *  This class may be used directly by instatiating an instance
 *  and throwing it, but usually you will use the NTA_THROW macro
 *  that makes it much simpler by automatically retreiving the __FILE__
 *  and __LINE__ for you.
 *
 */
class Exception : public std::runtime_error { 
public:
  /**
   * Constructor
   *
   * Take the filename, line number and message
   * and store it in private members
   *
   * @param filename [const std::string &] the name of the source file
   *        where the exception originated.
   * @param lineno [UInt32] the line number in the source file where
   *        the exception originated.
   *
   * @param message [const std::string &] the description of exception
   */
  Exception(std::string filename="",  UInt32 lineno=0,  std::string message = "", std::string stacktrace = "")
      : std::runtime_error(""), filename_(std::move(filename)), lineno_(lineno),
        message_(std::move(message)), stackTrace_(std::move(stacktrace))  {}


  Exception(const Exception &copy) : std::runtime_error("") {
    filename_ = copy.filename_;
    lineno_ = copy.lineno_;
    message_ = copy.getMessage();
    stackTrace_ = copy.stackTrace_;
  }
    

  /**
   * Destructor
   *
   * Doesn't do anything, but must be present
   * because the base class std::runtime_error
   * defines a pure virtual destructor and the
   * default destructor provided by the compiler
   * doesn't have a empty exception specification
   *
   */
  virtual ~Exception() throw() {}

  /**
   * Get the exception message via what()
   *
   * Overload the what() method of std::runtime_error
   * and returns the exception description message.
   * The emptry exception specification is required because
   * it is part of the signature of std::runtime_error::what().
   *
   * @retval [const Byte *] the exception message
   */
  virtual const char *what() const noexcept override {
    try {
      what_ = "Exception: " + Path::getBasename(filename_) + "(" + std::to_string(lineno_) 
                      + ") message: " + std::string(getMessage());
    } catch (...) {
    }
    return what_.c_str();
  }

  /**
   * Get the source filename
   *
   * Returns the full path to the source file, from which
   * the exception was thrown.
   *
   * @retval [const Byte *] the source filename
   */
  const char *getFilename() const noexcept { return filename_.c_str(); }

  /**
   * Get the line number in the source file
   *
   * Returns the (0-based) line number in the source file,
   * from which the exception was thrown.
   *
   * @retval [UInt32] the line number in the source file
   */
  UInt32 getLineNumber() const noexcept { return lineno_; }

  /**
   * Get the error message
   *
   * @retval [const char *] the error message
   */
  virtual const char *getMessage() const noexcept { 
	  message_ += ss_.str();
	  ss_.clear();
	  return message_.c_str(); }

  /**
   * Get the stack trace
   *
   * Returns the stack trace from the point the exception
   * was thrown.
   *
   * @retval [const Byte *] the stack trace
   */
  virtual const char *getStackTrace() const noexcept { return stackTrace_.c_str(); }


  template <typename T> 
  Exception &operator<<(const T &obj) {
    ss_ << obj;
    return *this;
  }

protected:
  std::string filename_;
  UInt32 lineno_;
  mutable std::string message_; //mutable bcs modified in getMessage which is used in what() but that needs be const
  std::string stackTrace_;
  mutable std::string what_;

private:
  mutable std::stringstream ss_;

}; // end class Exception
} // end namespace htm

#endif // NTA_EXCEPTION_HPP
