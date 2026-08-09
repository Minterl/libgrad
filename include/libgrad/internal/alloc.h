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

typedef uint32_t 
LG_AllocatorFlags;
enum {
    LG_AllocatorFlag_NoRecycle = UINT32_C(0x1),
    LG_AllocatorFlag_AssumeZeroed = UINT32_C(0x1 << 1),
};

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

    size_t default_slab_size_bytes; 

    LG_AllocatorFlags flags;
} LG_Allocator;

typedef struct
LG_Slab {
    struct LG_Slab       *prev;
    struct LG_Slab       *next;
    size_t                cap;
    uint8_t _Alignas(16)  buf[] lg_check_bounds(cap);
} LG_Slab;

typedef struct 
LG_Arena {
    LG_Allocator *host;

    size_t current_offset;
    struct LG_Slab *current_slab;

    struct LG_Slab *recycled_slabs_head;
} LG_Arena;

typedef struct
LG_Scope {
    LG_Slab *slab;
    size_t offset;
} LG_Scope;

uint8_t*
lg_alloc_zero(LG_Allocator *alloc, size_t size_bytes);

void 
lg_free(LG_Allocator *alloc, void *ptr);

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

void
lg_arena_init(LG_Arena *arena, LG_Allocator *host);
uint8_t*
lg_arena_alloc(LG_Arena *arena, size_t unaligned_size_bytes);
LG_Scope
lg_push_scope(LG_Arena *arena);
void
lg_pop_scope(LG_Arena *arena, LG_Scope scope);
void
lg_arena_free_recycled(LG_Arena *arena);
void
lg_arena_recycle_all(LG_Arena *arena);
void
lg_arena_free_all(LG_Arena *arena);

#endif // LG_ALLOC_H_
