#ifndef LIBGRAD_H_
#define LIBGRAD_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

#ifdef LIBGRAD_IMPLEMENTATION
#undef LIBGRAD_IMPLEMENTATION
#   include <libgrad/internal/core.c>
#   include <libgrad/internal/vm.c>
#   include <libgrad/internal/alloc.c>
#endif // LIBGRAD_IMPLEMENTATION

#ifdef __cplusplus
}
#endif // __cplusplus
       
#endif // LIBGRAD_H_
