#include "Core.h"

namespace sdm {

void doSDMError(const char* fileName, const char* funcName, int lineNumber, const char* msg)
{
  std::cerr << "ERROR: \"" << msg << "\" at file: " << fileName << ", func: " << funcName << ", line: " << lineNumber << ", exit now." << std::endl;
  breakPoint();
  exit(1);
}

void doSDMWarning(const char* fileName, const char* funcName, int lineNumber, const char* msg)
{
  std::cerr << "WARNING: \"" << msg << "\" at file: " << fileName << ", func: " << funcName << ", line: " << lineNumber << std::endl;
  breakPoint();
}

#ifdef DBG
void doSDMAssert(bool cond, const char* fileName, const char* funcName, int lineNumber)
{
  if (!cond) {
    std::cerr << "ASSERTION FAILED at file: " << fileName << ", func: " << funcName << ", line: " << lineNumber << ", exit now." << std::endl;
    breakPoint();
    exit(1);
  }
}
#endif

#ifdef DBG
void SDMBreakPoint() {}
#endif

// ----- convert from {bool, int, double} to string, and vice-versa
bool toBool(const std::string& s)
{
  if (s == "true")
    return true;
  else if (s == "false")
      return false;
  int ss = toInt(s);
  if (ss == 0)
    return false;
  return true;
}

int toInt(const std::string& s)
{
  const char* ss = s.c_str();
  char* errorcheck;
  int ans = strtol(ss, &errorcheck, 10);
  if (errorcheck == ss)
    SDMError("cannot convert the string to int");
  return ans;
}

double toDouble(const std::string& s)
{
  const char* ss = s.c_str();
  char* errorcheck;
  double ans = strtod(ss, &errorcheck);
  if (errorcheck == ss)
    SDMError("cannot convert the string to double");
  return ans;
}

// check the existence of file
bool fileExists(const std::string& filename)
{
  FILE* fp = fopen(filename.c_str(),"r");
  if (fp) {
    fclose(fp);
    return true;
  }
  return false;
}

// ----- commandOption class -----
CommandOptions::CommandOptions(int the_argc, char** the_argv)
  : command_size(the_argc), command_lines(the_argv)
{}

std::string CommandOptions::get(const std::string& the_option, const std::string& the_default)
{
  for (int i=0; i<command_size; i++)
    if (toString(command_lines[i]) == the_option && i+1 < command_size)
      return toString(command_lines[i+1]);
  return the_default;
}

} //namespace sdm

