#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/internal/core.h>

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
    /// In contiguous allocations, buffers are allocated to align with the 
    /// return value of this function. This is primarily useful for SIMD 
    /// operations that require strict alignment guarantees.
    ///
    /// Most implementations should just return a static value.
    ///
    /// Returning zero from this method triggers undefined behavior.
    lg_size (*align_hint)(void *ctx);
} lg_allocator;

/// Allocate necessary memory for `tensor`, and mutate the `data` pointer.
/// 
/// If a tensor is zero-sized, returns early without mutating `tensor`.
lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor);

/// Frees the buffers backing `tensor`.
lg_status lg_free_tensor(lg_allocator *allocator, lg_tensor *tensor);

/// Allocates the minimum amount of memory necessary to execute an expression
/// and populates the `data` pointers for any tensor where it is not null.
///
/// Allocates the expression backing buffer monolithically using `perm` and returns
/// the pointer using `out_data`, and allocates O(N) scratch memory using `scratch`.
lg_status lg_alloc_expr(
    lg_allocator *perm,
    lg_allocator *scratch,
    lg_scalar **out_data,
    lg_expr *expr
);

#endif // LG_ALLOC_H_

#ifdef LG_ALLOC_IMPLEMENTATION
#undef LG_ALLOC_IMPLEMENTATION

lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor) {
    lg_size one_size = lg_desc_size_bytes(tensor->desc);
    if (one_size == 0) {
        return LG_STATUS_OK;
    }

    lg_size total_size = one_size;

    uint8_t *ptr = allocator->alloc(allocator->ctx, total_size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    tensor->data = (lg_scalar*)ptr;

    return LG_STATUS_OK;
}

lg_status lg_free_tensor(lg_allocator *allocator, lg_tensor *tensor) {
#ifdef LG_SAFE
    if (tensor->data == NULL) {
        return LG_STATUS_NULL_POINTER;
    }
#endif // LG_SAFE
    allocator->free(allocator->ctx, tensor->data);
    tensor->data = NULL;
    return LG_STATUS_OK;
}

lg_status lg_alloc_expr(
    lg_allocator *perm,
    lg_allocator *scratch,
    lg_scalar **data,
    lg_expr *expr
) {
    // under the conditions of a cyclomatic complexity
    // of 1 and infinite registers, lsra reaches the global
    // optimum
    // TODO: add alignment
    
    lg_status status = LG_STATUS_OK;

    uint8_t *scratch_buf = scratch->alloc(scratch->ctx, expr->len * 4 * sizeof(lg_size));
    if (scratch_buf == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    lg_size *sizes = (lg_size*)scratch_buf;
    lg_size *dead_after = sizes + expr->len;
    lg_size *total_freed_after_time = dead_after + expr->len;
    lg_size *offsets = total_freed_after_time + expr->len;

    for (lg_size i = 0; i < expr->len; i++) {
        sizes[i] = 0;
        dead_after[i] = 0;
        total_freed_after_time[i] = 0;
        offsets[i] = 0;
    }

    for (lg_size i = 0; i < expr->len; i++) {
        if (expr->x0[i].data == NULL) {
            dead_after[expr->x0[i].born_at] = i;
        }
        if (expr->x1[i].data == NULL) {
            dead_after[expr->x1[i].born_at] = i;
        }
        sizes[expr->y[i].born_at] = lg_desc_size_bytes(expr->y[i].desc);
    }
    for (lg_size i = 0; i < expr->len; i++) {
        // TODO: maybe we need a pin flag or something?
        total_freed_after_time[dead_after[i]] += sizes[i];
    }

    lg_size current_offset = 0;
    lg_size max_offset = 0;
    for (lg_size time = 0; time < expr->len; time++) {
        offsets[time] = current_offset;
        current_offset += sizes[expr->y[time].born_at];
        if (current_offset > max_offset) {
            max_offset = current_offset;
        }
        current_offset -= total_freed_after_time[time];
    }

    uint8_t *_data = perm->alloc(perm->ctx, max_offset);
    if (_data == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out;
    }
    if (data != NULL) {
        *data = (lg_scalar*)_data;
    }

    for (lg_size i = 0; i < expr->len; i++) {
        expr->y[i].data = (lg_scalar*)(_data + offsets[expr->y[i].born_at]);
        if (expr->x0[i].data == NULL) {
            expr->x0[i].data = (lg_scalar*)(_data + offsets[expr->x0[i].born_at]);
        }
        if (expr->x1[i].data == NULL) {
            expr->x1[i].data = (lg_scalar*)(_data + offsets[expr->x1[i].born_at]);
        }
    }

out:
    scratch->free(scratch->ctx, scratch_buf);
    return status;
}

#endif // LG_ALLOC_IMPLEMENTATION
