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
    // --- no padding ---
    {
        lg_size expected_strides[] = {8, 4, 1};
        lg_tensor ten = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){3, 2, 4}, 1);
        test_assert(ten.rank == 3, "got tensor rank %d", ten.rank);
        for (int i = 0; i < 3; i++) {
            test_assert(
                expected_strides[i] == ten.strides[i],
                "wanted stride of %d, got %d, i = %d",
                expected_strides[i],
                ten.strides[i],
                i
            );
        }
    }

    // --- w/ padding ---
    {
        lg_size expected_strides[4] = {224, 32, 8, 1};
        lg_tensor ten = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){2, 7, 4, 3}, 8);
        test_assert(ten.rank == 4, "got tensor rank %d", ten.rank);
        for (int i = 0; i < 4; i++) {
            test_assert(
                expected_strides[i] == ten.strides[i],
                "wanted stride of %d, got %d, i = %d",
                expected_strides[i],
                ten.strides[i],
                i
            );
        }
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
    test_assert(_36_size == 36 * sizeof(lg_dtype), "tensor size was %d", _36_size);

    lg_tensor also_36 = {
        .rank = 3,
        .dim = {6, 2, 3},
        .strides = {6, 3, 1},
        .data = NULL,
        .grad = NULL,
    };
    lg_size also_36_size = lg_tensor_size_bytes(also_36);
    test_assert(also_36_size == 36 * sizeof(lg_dtype), "tensor size was %d", also_36_size);

    lg_tensor padded = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){3, 3, 3}, 4);
    lg_size calculated_bytes = lg_tensor_size_bytes(padded);
    // Strides should be (12, 4, 1), meaning the maximum offset at (2, 2, 2) is
    // (12 + 4 + 1) * 2, so the max size is one more than that.
    lg_size expected_bytes = ((12 + 4 + 1) * 2 + 1) * sizeof(lg_dtype);
    test_assert(
        calculated_bytes == expected_bytes,
        "tensor size calculated to be %d bytes, wanted %d",
        calculated_bytes,
        expected_bytes
    );

    lg_tensor zero_ten = {0};
    calculated_bytes = lg_tensor_size_bytes(zero_ten);
    test_assert(calculated_bytes == 0, "tensor size calculated to be %d bytes", calculated_bytes);

    lg_tensor scalar = lg_tensor_rmaj((lg_size[LG_MAX_RANK]){1}, 0);
    calculated_bytes = lg_tensor_size_bytes(scalar);
    test_assert(calculated_bytes == sizeof(lg_dtype), "tensor size calculated to be %d bytes", calculated_bytes);

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
    test_assert(calculated_one_bytes == expected_one_bytes, "tensor size calculated to be %d bytes", calculated_one_bytes);
    test_assert(lg_alloc_tensor(&allocator, &ten, 1) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(ten.data == (lg_dtype*)ALLOC_ADDR, "allocated data at address %d, expected %d", ten.data, (lg_dtype*)ALLOC_ADDR);
    test_assert(ten.grad == expected_grad_addr, "allocated grad at address %d, expected %d", ten.grad, expected_grad_addr);
    test_assert(ctx.bytes_allocated == expected_total_bytes, "allocated %d bytes, wanted %d bytes" , ctx.bytes_allocated, expected_total_bytes);

    return TEST_STATUS_OK;
}

int main(void) {
    test_run(tensor_init);
    test_run(tensor_size);
    test_run(alloc_tensor);
    return 0;
}
