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
struct lg_allocator {
    /// Context passed to each allocator method.
    void *ctx;
    /// Allocate `size_bytes` bytes.
    /// Callers will assume that this pointer is aligned.
    void* (*Alloc)(void *ctx, size_t size_bytes);
    /// Free the memory at `ptr`.
    void (*Free)(void *ctx, void *ptr);
};

/// Allocate the memory necessary for an expr of capacity `cap`,
/// and assign offsets into the buffer for each item in the SoA.
enum lg_status LG_AllocExpr(
    struct lg_allocator *allocator,
    uint8_t **out_ptr,
    size_t *out_bytes_allocated,
    struct lg_ir_expr *expr,
    size_t cap
);

/// Frees the memory required for an expr.
void LG_FreeExpr(struct lg_allocator *allocator, struct lg_ir_expr *expr);

/// Allocates the minimum amount of memory necessary to execute an expression
/// and populates the `data` pointers for any tensor where they're NULL.
///
/// Allocates the expression backing buffer monolithically using `perm` and returns 
/// and allocates O(N) scratch memory using `scratch`.
///
/// `out_ptr` and `out_bytes_allocated` are nullable.
enum lg_status LG_AllocExprData(
    struct lg_allocator *perm,
    struct lg_allocator *scratch,
    lg_scalar **out_ptr,
    size_t *out_bytes_allocated,
    struct lg_ir_expr *expr
);

#endif // LG_ALLOC_H_
