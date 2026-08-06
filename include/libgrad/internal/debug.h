#ifndef LG_DEBUG_H_
#define LG_DEBUG_H_

#include <stdbool.h>

#ifdef LG_DEBUG
#   if defined(__has_builtin) && __has_builtin(__builtin_unreachable)
#       define lg_unreachable_ __builtin_unreachable()
#   else
#       define lg_unreachable_
#   endif // __has_builtin(__builtin_unreachable)
#   define lg_dbgf(fmt, ...) lg_dbgf_(__FILE__, __LINE__, fmt, __VA_ARGS__)
#   define lg_assert(cond) lg_assert_(__FILE__, __LINE__, (cond), #cond)
#   define lg_unreachable(...) do { lg_assert(false); lg_unreachable_; } while (0)
#else
#   define lg_dbgf(fmt, ...)
#   define lg_assert(cond) ((void)(cond))
#   define lg_unreachable(...)
#endif // LG_DEBUG

void lg_dbgf_(const char *file, int line, const char* fmt, ...);
void lg_assert_(const char *file, int line, bool cond, const char *cond_str);

#endif // LG_DEBUG_H_
