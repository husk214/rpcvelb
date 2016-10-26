#ifndef CORE_H_
#define CORE_H_

// include list
#include <string>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>

namespace sdm {

#ifndef M_PI
#define M_PI 3.1415926535897932846
#endif

// ----- error -----
#define SDMError(MSG) doSDMError(__FILE__, __FUNCTION__, __LINE__, MSG)
void doSDMError(const char* fileName, const char* funcName, int lineNumber, const char* msg);

// ----- warining -----
#define SDMWarning(MSG) doSDMWarning(__FILE__, __FUNCTION__, __LINE__, MSG)
void doSDMWarning(const char* filename, const char* funcname, int lineNO, const char* msg);

// ----- assert -----
#ifdef DBG
#define SDMAssert(COND) doSDMAssert((COND), __FILE__, __FUNCTION__, __LINE__)
void doSDMAssert(bool cond, const char* fileName, const char* funcName, int lineNumber);
#else
#define SDMAssert(COND)
#endif

// ----- break points for debuging  -----
#ifdef DBG
#define breakPoint(COND) SDMBreakPoint()
void SDMBreakPoint();
#else
#define breakPoint()
#endif

template<class T>
std::string toString(const T& x);

bool toBool(const std::string& s);
int toInt(const std::string& s);
double toDouble(const std::string& s);

// check if the file exists
bool fileExists(const std::string& filename);

// a useful class to set options from commandline
class CommandOptions
{
private:
  int command_size;
  char** command_lines;
public:
  CommandOptions(int the_argc, char** the_argv);
  std::string get(const std::string& the_option, const std::string& the_default);
};

// IMPLEMENTATION of template function
template<class T>
std::string toString(const T& x)
{
  std::ostringstream out;
  out << x;
  return out.str();
}


} //namespace sdm

#endif
