#include <stdio.h>
#include "tensorflow/lite/micro/debug_log.h"

extern "C" void DebugLog(const char* s)
{
  printf("%s", s);
}
