#include <libgrad/internal/debug.h>

// Use a guard here b/c this block expects libc,
// which may not be available.
#ifdef LG_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void LG___Dbgf(const char *file, int line, const char* fmt, ...) {
    fprintf(stderr, "\033[32m[DEBUG]\033[0m (%s:%d) -- ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void LG___Assert(const char *file, int line, bool cond, const char *cond_str) {
    if (!cond) {
        fprintf(stderr, "\x1b[31m[ASSERTION FAILED]\x1b[0m (%s) at %s:%d\n", cond_str, file, line);
        abort();
    }
}

#endif // LG_DEBUG
