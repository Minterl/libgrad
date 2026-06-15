#ifndef LG_CORE_IMPLEMENTATION
#define LG_CORE_IMPLEMENTATION
#endif // LG_CORE_IMPLEMENTATION
#ifndef LG_ALLOC_IMPLEMENTATION
#define LG_ALLOC_IMPLEMENTATION
#endif // LG_ALLOC_IMPLEMENTATION
#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#ifndef LG_INTERNAL_DEBUG_IMPLEMENTATION
#define LG_INTERNAL_DEBUG_IMPLEMENTATION
#endif // LG_INTERNAL_DEBUG_IMPLEMENTATION
 
#include <stdbool.h>
#include <libgrad/core.h>
#include <libgrad/alloc.h>
#include "testing.h"

// Using unusual memory alignment of 5 for testing purposes
#define ALLOC_ADDR (void*)(5 * sizeof(lg_dtype))
#define ALLOC_ALIGN 5 * sizeof(lg_dtype)

typedef struct mock_allocator_context {
    lg_size bytes_allocated;
} mock_allocator_context;

void* mock_alloc(void *_ctx, lg_size size_bytes) {
    mock_allocator_context *ctx = _ctx;
    ctx->bytes_allocated += size_bytes;
    return ALLOC_ADDR;
}

inline static lg_size mock_align_hint() {
    return ALLOC_ALIGN;
}

test_status test_tensor_init() {
    // --- w/o padding ---
    {
        lg_size expected_strides[] = {8, 4, 1};
        lg_tensor ten = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){3, 2, 4}, 1);
        test_assert(ten.rank == 3, "got tensor rank %lu", ten.rank);
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }

    // --- w/ padding ---
    {
        lg_size expected_strides[4] = {224, 32, 8, 1};
        lg_tensor ten = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){2, 7, 4, 3}, 8);
        test_assert(ten.rank == 4, "got tensor rank %lu", ten.rank);
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }

    // --- isotropic w/o padding ---
    {
        lg_size expected_dims[4] = {4, 4, 4, 4};
        lg_size expected_strides[4] = {64, 16, 4, 1};

        lg_tensor ten = {0};
        lg_status status = lg_tensor_rmaj_isotropic(&ten, 4, 4, 1);

        test_assert(status == LG_STATUS_OK, "failed to initialize isotropic tensor");
        test_assert(ten.rank == 4, "got tensor rank %lu", ten.rank);
        test_assert(lg_tensor_is_isotropic(ten), "tensor was not isotropic");
        test_assert_array_eq(expected_dims, ten.dim, 3, "%lu");
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }

    return TEST_STATUS_OK;
}

test_status test_tensor_size() {
    lg_tensor _36 = {
        .rank = 2, 
        .dim = {6, 6},
        .strides = {6, 1},
        .data = NULL,
        .grad = NULL,
    };
    lg_size _36_size = lg_tensor_size_bytes(_36);
    test_assert(_36_size == 36 * sizeof(lg_dtype), "tensor size was %lu", _36_size);

    lg_tensor also_36 = {
        .rank = 3,
        .dim = {6, 2, 3},
        .strides = {6, 3, 1},
        .data = NULL,
        .grad = NULL,
    };
    lg_size also_36_size = lg_tensor_size_bytes(also_36);
    test_assert(also_36_size == 36 * sizeof(lg_dtype), "tensor size was %lu", also_36_size);

    lg_tensor padded = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){3, 3, 3}, 4);
    lg_size calculated_bytes = lg_tensor_size_bytes(padded);
    // Strides should be (12, 4, 1), meaning the maximum offset at (2, 2, 2) is
    // (12 + 4 + 1) * 2, so the max size is one more than that.
    lg_size expected_bytes = ((12 + 4 + 1) * 2 + 1) * sizeof(lg_dtype);
    test_assert(
        calculated_bytes == expected_bytes,
        "tensor size calculated to be %lu bytes, wanted %lu",
        calculated_bytes,
        expected_bytes
    );

    lg_tensor zero_ten = {0};
    calculated_bytes = lg_tensor_size_bytes(zero_ten);
    test_assert(calculated_bytes == 0, "tensor size calculated to be %lu bytes", calculated_bytes);

    lg_tensor scalar = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){1}, 0);
    calculated_bytes = lg_tensor_size_bytes(scalar);
    test_assert(calculated_bytes == sizeof(lg_dtype), "tensor size calculated to be %lu bytes", calculated_bytes);

    return TEST_STATUS_OK;
}

test_status test_alloc_tensor() {
    mock_allocator_context ctx = {0};
    lg_allocator allocator = {
        .ctx = (void*)&ctx,
        .alloc = mock_alloc,
        .free = NULL,
        .align_hint = mock_align_hint,
    }; 
    lg_tensor ten = {
        .rank = 2, 
        .dim = {6, 6},
        .strides = {6, 1},
        .data = NULL,
        .grad = NULL,
    };

    // If we're aligning to a 5 * sizeof(lg_dtype)-byte boundary and starting at 5, then
    // the data allocation should end at 41 * sizeof(lg_dtype),
    // and the gradient allocation should start at the nearest multiple of 5,
    // which is 40 * sizeof(lg_dtype) if lg_dtype is float, yielding 4 * sizeof(lg_dtype) bytes of padding.
    lg_size expected_one_bytes = 36 * sizeof(lg_dtype);
    lg_size expected_total_bytes = expected_one_bytes * 2 + 4 * sizeof(lg_dtype);
    lg_dtype *expected_grad_addr = (lg_dtype*)ALLOC_ADDR + 40;

    lg_size calculated_one_bytes = lg_tensor_size_bytes(ten);
    test_assert(calculated_one_bytes == expected_one_bytes, "tensor size calculated to be %lu bytes", calculated_one_bytes);
    test_assert(lg_alloc_tensor(&allocator, &ten, 1) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(ten.data == (lg_dtype*)ALLOC_ADDR, "allocated data at address %lu, expected %lu", ten.data, (lg_dtype*)ALLOC_ADDR);
    test_assert(ten.grad == expected_grad_addr, "allocated grad at address %lu, expected %lu", ten.grad, expected_grad_addr);
    test_assert(ctx.bytes_allocated == expected_total_bytes, "allocated %lu bytes, wanted %lu bytes" , ctx.bytes_allocated, expected_total_bytes);

    return TEST_STATUS_OK;
}

test_status test_tensor_aligned_views_not_compatible() {
    // 4 != 5, so these should clash and not be compatible
    lg_tensor a = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){4, 4}, 1);
    lg_tensor b = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){6, 5, 4}, 1);

    lg_status status = lg_tensor_optimize_views((lg_tensor*[]){&a, &b}, 2);
    test_assert(status == LG_STATUS_SHAPE_MISMATCH, "failed to detect shape mismatch");

    return TEST_STATUS_OK;
}

test_status test_tensor_aligned_views() {
    // This is a 6x4x4 tensor.
    // In memory, this looks like this:
    // { 
    //     (0, 0, 0), (0, 0, 1) ...
    //     (0, 1, 0), (0, 1, 1) ...
    //     (1, 0, 0), (1, 0, 1) ...
    //     (m-1, n-1, k-1)
    // }
    lg_tensor a = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){6, 4, 4}, 1);
    // This is a mat44.
    // In memory, with no alignment, this should be a contiguous
    // row-major 2d array (these are (x, y) pairs, not matrix coords):
    // { (0, 0), (0, 1) ... (1, 0), (1, 1) ... (m-1, n-1) }
    lg_tensor b = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){4, 4}, 1);

    test_assert(b.strides[0] == 4, "got first stride of %lu", a.strides[0]);
    test_assert(b.strides[1] == 1, "got second stride of %lu", a.strides[1]);

    // Coalesced dimensions
    lg_size expected_strides[] = {1};

    test_assert(lg_tensor_optimize_views((lg_tensor*[]){&a, &b}, 2) == LG_STATUS_OK, "failed to align views");
    test_assert_array_eq(expected_strides, a.strides, 1, "%lu");
    test_assert_array_eq(expected_strides, b.strides, 1, "%lu");
    
    return TEST_STATUS_OK;
}

int main(void) {
    test_run(tensor_init);
    test_run(tensor_size);
    test_run(alloc_tensor);
    test_run(tensor_aligned_views_not_compatible);
    test_run(tensor_aligned_views);
    return 0;
}
