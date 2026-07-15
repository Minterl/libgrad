#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

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
typedef struct lg_allocator {
    /// Context passed to each allocator method.
    void *ctx;
    /// Allocate `size_bytes` bytes.
    /// Callers will assume that this pointer is aligned.
    void* (*alloc)(void *ctx, lg_size size_bytes);
    /// Free the memory at `ptr`.
    void (*free)(void *ctx, void *ptr);
} lg_allocator;

/// Allocate necessary memory for `tensor`, and mutate the `data` pointer.
/// 
/// If a tensor is zero-sized, returns early without mutating `tensor`.
lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor);

/// Frees the buffers backing `tensor`.
lg_status lg_free_tensor(lg_allocator *allocator, lg_tensor *tensor);

/// Allocate the memory necessary for an expr of capacity `cap`,
/// and assign offsets into the buffer for each item in the SoA.
lg_status lg_alloc_expr(
    lg_allocator *allocator,
    uint8_t **out_ptr,
    lg_size *out_bytes_allocated,
    lg_expr *expr,
    lg_size cap
);

/// Frees the memory required for an expr.
void lg_free_expr(lg_allocator *allocator, lg_expr *expr);

/// Allocates the minimum amount of memory necessary to execute an expression
/// and populates the `data` pointers for any tensor where they're NULL.
///
/// Allocates the expression backing buffer monolithically using `perm` and returns 
/// and allocates O(N) scratch memory using `scratch`.
///
/// `out_ptr` and `out_bytes_allocated` are nullable.
lg_status lg_alloc_expr_data(
    lg_allocator *perm,
    lg_allocator *scratch,
    lg_scalar **out_ptr,
    lg_size *out_bytes_allocated,
    lg_expr *expr
);

#endif // LG_ALLOC_H_
