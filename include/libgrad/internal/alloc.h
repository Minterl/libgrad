#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/internal/core.h>

#define LG__ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

#if defined(__has_builtin) && __has_builtin(__builtin_memcpy)
#   define LG__MEMCPY(dest, src, size) __builtin_memcpy(dest, src, size)
#else
#   define LG__MEMCPY(dest, src, size) do { \
         for(size_t LG__MACRO_ITER__ = 0; LG__MACRO_ITER__ < (size); LG__MACRO_ITER__++) { \
             ((uint8_t*)(dest))[LG__MACRO_ITER__] = ((uint8_t*)(src))[LG__MACRO_ITER__] ; \
         } \
     } while(0) 
#endif // defined(__has_builtin) && __has_builtin(__builtin_memcpy)

#if defined(__has_builtin) && __has_builtin(__builtin_memset)
#   define LG__ZERO(ptr, size) __builtin_memset(ptr, 0, size) 
#else
#   define LG__ZERO(ptr, size) do { \
         for(size_t LG__MACRO_ITER__ = 0; LG__MACRO_ITER__ < (size); LG__MACRO_ITER__++) { \
             ((uint8_t*)(ptr))[LG__MACRO_ITER__] = 0; \
         } \
     } while(0) 
#endif // defined(__has_builtin) && __has_builtin(__builtin_memset)

#define LG_ALLOCATOR_SUPPORTS_SCRATCH(alloc) (((alloc)->AcquireScratch != NULL) && ((alloc)->ReleaseScratch != NULL))

/// Helper interface for allocating tensors
///
/// Many users will choose to completely omit use of this utility,
/// but it is very useful for allocating tensors quickly.
///
/// The only method that must be defined is `alloc`, the others may
/// legally be NULL. Such is the case when using an arena-style allocator,
/// where free is a no-op and the user deallocates memory outside of this interface.
///
/// Don't try to get clever and swap this out under the library's feet between calls.
/// That will almost certainly end poorly.
struct lg_allocator {
    /// Context passed to each allocator method.
    void *ctx;

    /// Allocate `size_bytes` bytes.
    /// Callers will assume that this pointer is aligned.
    void* (*Alloc)(void *ctx, size_t size_bytes);
    /// Free the memory at `ptr`.
    /// TODO: find the direct calls to this and replace them with a macro or something.
    void  (*LG_NULLABLE Free)(void *ctx, void *ptr);

    /// If `AcquireScratch` and `ReleaseScratch` are both non-null, then the allocator will relinquish control
    /// over scratch memory to whatever reclamation mechanism the caller chooses.
    /// Most commonly, this is saving a highwater mark in an arena-style allocator.
    ///
    /// If either of these are null, then the allocator will use its own internal reclamation system (which does
    /// not reqiure any special attention from the caller).
    void* (*LG_NULLABLE AcquireScratch)(void *ctx);
    void  (*LG_NULLABLE ReleaseScratch)(void *ctx, void *waypoint);
};

struct lg_scratch_node {
    struct lg_scratch_node *next;
};
_Static_assert(sizeof(void*) == sizeof(struct lg_scratch_node), "");

uint8_t *LG__AllocZero(struct lg_allocator *alloc, size_t size_bytes);
void LG__Free(struct lg_allocator *alloc, void *ptr);

struct lg_scratch_node *LG__AcquireScratch(struct lg_allocator *alloc);
uint8_t *LG__AllocScratch(struct lg_allocator *alloc, struct lg_scratch_node **waypoint, size_t size_bytes);
void LG__ReleaseScratch(struct lg_allocator *alloc, struct lg_scratch_node **waypoint);

/// Allocates `n` blocks of size `sizes[i]` and puts the resulting pointer
/// in `out_ptrs[i]`.
///
/// These blocks are guaranteed to be contiguous in memory and aligned to `align`.
///
/// `out_ptrs[0]` is the pointer the allocated region itself i.e the pointers
/// are allocated in the order of `out_ptrs`.
enum lg_status LG__AllocContiguousBlocks(
    struct lg_allocator *alloc,
    uint8_t **out_ptrs,
    size_t *LG_NULLABLE out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
);

#endif // LG_ALLOC_H_
