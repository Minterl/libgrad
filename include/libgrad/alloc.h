#ifndef LG_ALLOC_H_
#define LG_ALLOC_H_

#include <libgrad/core.h>

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

/// Allocate necessary memory for `tensor`, and mutate the `data`
/// and `grad` pointers, the latter iff `with_grad` == true.
///
/// Uses a single allocation for `data` and `grad` buffers.
///
/// If allocator.align_hint != NULL, then the gradient buffer will be aligned
/// with the result.
/// 
/// If a tensor is zero-sized, returns early without mutating `tensor`.
lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor, lg_bool with_grad);

/// Frees the buffers backing `tensor`.
///
/// Assumes `tensor` was allocated using `lg_alloc_tensor`, which uses a single allocation
/// for both the `data` and `grad` buffers.
lg_status lg_free_tensor(lg_allocator *allocator, lg_tensor *tensor);

lg_status lg_alloc_tensor_many(lg_allocator *allocator, const lg_tensor *tensors, lg_size len, lg_bool with_grad);
lg_status lg_free_tensor_many(lg_allocator *allocator, const lg_tensor *tensors, lg_size len, lg_bool with_grad);

#endif // LG_ALLOC_H_

#ifdef LG_ALLOC_IMPLEMENTATION
#undef LG_ALLOC_IMPLEMENTATION

#include <libgrad/internal/debug.h>

lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor, lg_bool with_grad) {
    lg_size one_size = lg_tensor_size_bytes(*tensor);
    if (one_size == 0) {
        return LG_STATUS_OK;
    }

    lg_size total_size = one_size;
    lg_size padding = 0;
    if (with_grad) {
        lg_size alignment = allocator->align_hint ? allocator->align_hint(allocator->ctx) : 1;
        padding = ((one_size + alignment - 1) & ~(alignment - 1)) - one_size;
        total_size += padding + one_size;
    }

    lg_byte *ptr = allocator->alloc(allocator->ctx, total_size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    if (with_grad) {
        lg_byte *grad_offset = ptr + one_size + padding;
        tensor->data = (lg_dtype*)ptr;
        tensor->grad = (lg_dtype*)grad_offset;
    } else {
        tensor->data = (lg_dtype*)ptr;
        tensor->grad = NULL;
    }

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
    tensor->grad = NULL;
    return LG_STATUS_OK;
}

#endif // LG_ALLOC_IMPLEMENTATION
