#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/internal/core.h>

#define lg_align_up(x, align) (((x) + (align) - 1) & ~((align) - 1))

#if defined(__has_builtin) && __has_builtin(__builtin_memcpy)
#   define lg_memcpy(dest, src, size) __builtin_memcpy(dest, src, size)
#else
#   define lg_memcpy(dest, src, size) do { \
         for(size_t LG__MACRO_ITER__ = 0; LG__MACRO_ITER__ < (size); LG__MACRO_ITER__++) { \
             ((uint8_t*)(dest))[LG__MACRO_ITER__] = ((uint8_t*)(src))[LG__MACRO_ITER__] ; \
         } \
     } while(0) 
#endif // defined(__has_builtin) && __has_builtin(__builtin_memcpy)

#if defined(__has_builtin) && __has_builtin(__builtin_memset)
#   define lg_memzero(ptr, size) __builtin_memset(ptr, 0, size) 
#else
#   define lg_memzero(ptr, size) do { \
         for(size_t LG__MACRO_ITER__ = 0; LG__MACRO_ITER__ < (size); LG__MACRO_ITER__++) { \
             ((uint8_t*)(ptr))[LG__MACRO_ITER__] = 0; \
         } \
     } while(0) 
#endif // defined(__has_builtin) && __has_builtin(__builtin_memset)

#define lg_allocator_supports_scratch(alloc) (((alloc)->scratch_acquire != NULL) && ((alloc)->scratch_release != NULL))

/// The only method that must be defined is `alloc`, the others may
/// legally be NULL. Such is the case when using an arena-style allocator,
/// where free is a no-op and the user deallocates memory outside of this interface.
///
/// Don't try to get clever and swap this out under the library's feet between calls.
/// That will almost certainly end poorly.
typedef struct
LG_Allocator {
    /// Context passed to each allocator method.
    void *ctx;

    /// Allocate `size_bytes` bytes.
    /// Callers will assume that this pointer is aligned.
    void* (*alloc)(void *ctx, size_t size_bytes);
    /// Free the memory at `ptr`.
    /// TODO: find the direct calls to this and replace them with a macro or something.
    void  (*lg_nullable free)(void *ctx, void *ptr);

    /// If `AcquireScratch` and `ReleaseScratch` are both non-null, then the allocator will relinquish control
    /// over scratch memory to whatever reclamation mechanism the caller chooses.
    /// Most commonly, this is saving a highwater mark in an arena-style allocator.
    ///
    /// If either of these are null, then the allocator will use its own internal reclamation system (which does
    /// not reqiure any special attention from the caller).
    void* (*lg_nullable scratch_acquire)(void *ctx);
    void  (*lg_nullable scratch_release)(void *ctx, void *waypoint);
} LG_Allocator;

typedef struct
LG_ScratchWaypoint {
    struct LG_ScratchWaypoint *next;
} LG_ScratchWaypoint;

_Static_assert(sizeof(void*) == sizeof(LG_ScratchWaypoint), "");

uint8_t*
lg_alloc_zero(LG_Allocator *alloc, size_t size_bytes);

void 
lg_free(LG_Allocator *alloc, void *ptr);

LG_ScratchWaypoint*
lg_scratch_acquire(LG_Allocator *alloc);

uint8_t*
lg_scratch_alloc(LG_Allocator *alloc, LG_ScratchWaypoint **waypoint, size_t size_bytes);

void 
lg_scratch_release(LG_Allocator *alloc, LG_ScratchWaypoint **waypoint);

/// Allocates `n` blocks of size `sizes[i]` and puts the resulting pointer
/// in `out_ptrs[i]`.
///
/// These blocks are guaranteed to be contiguous in memory and aligned to `align`.
///
/// `out_ptrs[0]` is the pointer the allocated region itself i.e the pointers
/// are allocated in the order of `out_ptrs`.
LG_StatusKind 
lg_alloc_contiguous_blocks(
    LG_Allocator *alloc,
    uint8_t **out_ptrs,
    size_t *lg_nullable out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
);

#endif // LG_ALLOC_H_
