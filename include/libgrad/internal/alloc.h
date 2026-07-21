#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/internal/core.h>

#define LG__ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

/// Helper interface for allocating tensors
///
/// Many users will choose to completely omit use of this utility,
/// but it is very useful for allocating tensors quickly.
///
/// The only method that must be defined is `alloc`, the others may
/// legally be NULL. Such is the case when using an arena-style allocator,
/// where free is a no-op and the user deallocates memory outside of this interface.
///
/// If you are not sure as to how to use this interface, then either
///
/// a) don't (just write it yourself), or
///
/// b) read the source. It's more boilerplate than anything else.
struct lg_allocator {
    /// Context passed to each allocator method.
    void *ctx;
    /// Allocate `size_bytes` bytes.
    /// Callers will assume that this pointer is aligned.
    void* (*Alloc)(void *ctx, size_t size_bytes);
    /// Free the memory at `ptr`.
    void (*Free)(void *ctx, void *ptr);
};

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
