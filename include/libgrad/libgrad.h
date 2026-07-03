#ifndef LIBGRAD_H_
#define LIBGRAD_H_

#ifdef __cplusplus
extern "C" {
#endif //  __cplusplus

#ifdef LIBGRAD_IMPLEMENTATION
#   define LG_CORE_IMPLEMENTATION
#   define LG_ALLOC_IMPLEMENTATION
#   define LG_DEBUG_IMPLEMENTATION
#endif // LIBGRAD_IMPLEMENTATION

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

#ifdef __cplusplus
}
#endif //  __cplusplus
       
#endif //  LIBGRAD_H_
