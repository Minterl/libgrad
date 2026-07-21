#ifndef LG_DEBUG_H_
#define LG_DEBUG_H_

#include <stdbool.h>

#ifdef LG_DEBUG
#   if defined(__has_builtin) && __has_builtin(__builtin_unreachable)
#       define LG__BUILTIN_UNREACHABLE __builtin_unreachable()
#   else
#       define LG__BUILTIN_UNREACHABLE
#   endif // __has_builtin(__builtin_unreachable)
#   define LG__Dbgf(fmt, ...) LG___Dbgf(__FILE__, __LINE__, fmt, __VA_ARGS__)
#   define LG__Assert(cond) LG___Assert(__FILE__, __LINE__, (cond), #cond)
#   define LG__Unreachable(...) do { LG__Assert(false); LG__BUILTIN_UNREACHABLE; } while (0)
#else
#   define LG__Dbgf(fmt, ...)
#   define LG__Assert(cond) ((void)(cond))
#   define LG__Unreachable(...)
#endif // LG_DEBUG

void LG___Dbgf(const char *file, int line, const char* fmt, ...);
void LG___Assert(const char *file, int line, bool cond, const char *cond_str);

#endif // LG_DEBUG_H_
