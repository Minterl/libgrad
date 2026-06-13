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

/// Returns a tensor with dim `dim`, with strides as if it was stored in row-major
/// order. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `row_align` if `row_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
static inline lg_tensor lg_tensor_init_default(lg_size dim[LG_MAX_RANK], lg_size row_align);

/// Compute the size in bytes of a tensor's data buffer.
static inline lg_size lg_tensor_size_bytes(const lg_tensor tensor);

#endif // LG_ALLOC_H_

#ifdef LG_ALLOC_IMPLEMENTATION
#undef LG_ALLOC_IMPLEMENTATION

#include <libgrad/internal/debug.h>

static inline lg_tensor lg_tensor_init_default(lg_size dim[LG_MAX_RANK], lg_size row_align) {
    lg_tensor ten = {
        .dim = {*dim},
    };

    for (ten.rank = 0; dim[ten.rank] > 0; ten.rank++) {
         ten.dim[ten.rank] = dim[ten.rank];
    }

    lg_size last_stride = 1;
    for (lg_size i = 1; i <= ten.rank; i++) {
        lg_size axis = ten.rank - i;
        ten.strides[axis] = last_stride;
        last_stride *= dim[ten.rank - i];
        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (row_align > 1 && i == 1) {
            last_stride = (last_stride + row_align - 1) & ~(row_align - 1);
        }
    }

    return ten;
}

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

static inline lg_size lg_tensor_size_bytes(const lg_tensor tensor) {
    if (tensor.rank == 0) {
        return 0;
    }

    lg_size max_offset = 0;
    for (lg_size i = 0; i < tensor.rank; i++) {
        if (tensor.dim[i] > 0) {
            max_offset += (tensor.dim[i] - 1) * tensor.strides[i];
        }
    }

    return (max_offset + 1) * sizeof(lg_dtype);
}

#endif // LG_ALLOC_IMPLEMENTATION
