#ifndef LG_DEBUG_H_
#define LG_DEBUG_H_
#endif // LG_DEBUG_H_

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
#   define LG__DBGF(fmt, ...)
#   define LG__Assert(cond) ((void)(cond))
#   define LG__Unreachable(...)
#endif // LG_DEBUG

// Use a guard here b/c this block expects libc,
// which may not be available.
#ifdef LG_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>

static inline void LG___Dbgf(const char *file, int line, const char* fmt, ...) {
    fprintf(stderr, "\033[32m[DEBUG]\033[0m (%s:%d) -- ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

static inline void LG___Assert(const char *file, int line, bool cond, const char *cond_str) {
    if (!cond) {
        fprintf(stderr, "\x1b[31m[ASSERTION FAILED]\x1b[0m (%s) at %s:%d\n", cond_str, file, line);
        abort();
    }
}

#endif // LG_DEBUG
