#ifndef LG_INTERNAL_DEBUG_H_
#define LG_INTERNAL_DEBUG_H_
#endif // LG_INTERNAL_DEBUG_H_

#ifdef LG_DEBUG
#define lg_dbgf(fmt, ...) __lg_dbgf(__FILE__, __LINE__, fmt, __VA_ARGS__)
#else
#define lg_dbgf(fmt, ...) /* nothing */
#endif // LG_DEBUG

void __lg_dbgf(const char *file, int line, const char* fmt, ...);

#if defined(LG_INTERNAL_DEBUG_IMPLEMENTATION) && defined(LG_DEBUG)
#undef LG_INTERNAL_DEBUG_IMPLEMENTATION

#include <stdio.h>
#include <stdarg.h>

void __lg_dbgf(const char *file, int line, const char* fmt, ...) {
    fprintf(stderr, "\033[32m[DEBUG]\033[0m (%s:%d) -- ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#endif // LG_INTERNAL_DEBUG_IMPLEMENTATION
