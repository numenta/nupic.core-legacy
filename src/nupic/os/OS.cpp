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

/** @file
 * Generic OS Implementations for the OS class
 */

#include <apr-1/apr_errno.h>
#include <apr-1/apr_network_io.h>
#include <apr-1/apr_time.h>
#include <nupic/os/Directory.hpp>
#include <nupic/os/Env.hpp>
#include <nupic/os/OS.hpp>
#include <nupic/os/Path.hpp>
#include <nupic/utils/Log.hpp>

#if defined(NTA_OS_DARWIN)
extern "C" {
#include <mach/mach_init.h>
#include <mach/task.h>
}
#elif defined(NTA_OS_WINDOWS)
// We only run on XP/2003 and above
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#include <psapi.h>
#endif

using namespace nupic;

void OS::getProcessMemoryUsage(size_t &realMem, size_t &virtualMem) {
#if defined(NTA_OS_DARWIN)
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

  if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO,
                                (task_info_t)&t_info, &t_info_count)) {
    NTA_THROW << "getProcessMemoryUsage -- unable to get memory usage";
  }
  realMem = t_info.resident_size;
  virtualMem = t_info.virtual_size;
#elif defined(NTA_OS_WINDOWS)
  HANDLE hProcess = ::GetCurrentProcess();
  BOOL rc;
  SYSTEM_INFO si;

  ::GetSystemInfo(&si);

  PSAPI_WORKING_SET_INFORMATION *pWSI = NULL;
  unsigned int pageCount = 2500;
  unsigned int size;

  unsigned int retries;
  for (retries = 0; retries < 20; retries++) {
    size = sizeof(PSAPI_WORKING_SET_INFORMATION) +
           pageCount * sizeof(PSAPI_WORKING_SET_BLOCK);

    pWSI = (PSAPI_WORKING_SET_INFORMATION *)realloc((void *)pWSI, size);

    if (::QueryWorkingSet(hProcess, pWSI, size)) {
      break;
    }

    if (::GetLastError() != ERROR_BAD_LENGTH) {
      free((void *)pWSI);
      ::CloseHandle(hProcess);
      NTA_THROW << "getProcessMemoryUsage -- unable to get memory usage";
    }

    pageCount = (pWSI->NumberOfEntries + 1);
    pageCount += pageCount >> 2;
  }

  if (retries >= 20) {
    free((void *)pWSI);
    ::CloseHandle(hProcess);
    NTA_THROW << "getProcessMemoryUsage -- unable to get memory usage";
  }

  unsigned int actualPages;

  pWSI->NumberOfEntries > pageCount ? (actualPages = pageCount)
                                    : (actualPages = pWSI->NumberOfEntries);

  unsigned int privateWorkingSet = 0;

  for (unsigned int i = 0; i < actualPages; i++) {
    if (!pWSI->WorkingSetInfo[i].Shared) {
      privateWorkingSet += si.dwPageSize;
    }
  }

  // subtract off memory allocated for our pWSI
  privateWorkingSet -= ((size / si.dwPageSize) + 1) * si.dwPageSize;

  free((void *)pWSI);

  PROCESS_MEMORY_COUNTERS_EX pmcEx;

  pmcEx.cb = sizeof(PROCESS_MEMORY_COUNTERS_EX);

  rc = ::GetProcessMemoryInfo(
      hProcess, reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmcEx),
      sizeof(PROCESS_MEMORY_COUNTERS_EX));

  if (!rc) {
    NTA_THROW << "getProcessMemoryUsage -- unable to get memory usage";
  }

  // Private usage corresponds to the total amount of private virtual memory
  virtualMem = pmcEx.PrivateUsage;

  // The private working set corresponds to the unshared virtual memory in
  // the processes' working set
  realMem = privateWorkingSet;

  ::CloseHandle(hProcess);
#else
  // TODO -- ADD!
  realMem = 100;
  virtualMem = 100;
#endif
}

std::string OS::executeCommand(std::string command) {
#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
  FILE *pipe = _popen(&command[0], "r");
#else
  FILE *pipe = popen(&command[0], "r");
#endif
  if (!pipe) {
    return "ERROR";
  }
  char buffer[128];
  std::string result = "";
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != nullptr) {
      result += buffer;
    }
  }
#if defined(NTA_OS_WINDOWS) && defined(NTA_COMPILER_MSVC)
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  return result;
}
