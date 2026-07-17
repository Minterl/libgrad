#ifndef LIBGRAD_IMPLEMENTATION
#define LIBGRAD_IMPLEMENTATION
#include "libgrad/internal/alloc.h"
#endif // LIBGRAD_IMPLEMENTATION
#ifndef LG_CPU_IMPLEMENTATION
#define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
 
#include <stdbool.h>
#include <stdlib.h>
#include <libgrad/libgrad.h>
#include <libgrad/cpu.h>
#include "testing.h"

// Using unusual memory alignment of 5 for testing purposes
#define ALLOC_ADDR (void*)(5 * sizeof(lg_scalar))

typedef struct mock_allocator_context {
    size_t bytes_allocated;
} mock_allocator_context;

void* mock_alloc(void *_ctx, size_t size_bytes) {
    mock_allocator_context *ctx = _ctx;
    ctx->bytes_allocated += size_bytes;
    return ALLOC_ADDR;
}

static inline bool increment_coords_rtl(size_t *coords, const size_t *dim, size_t rank) {
    if (rank == 0) return false;

    size_t axis = rank;
    
    while (axis > 0) {
        axis--;
        coords[axis]++;
        if (coords[axis] < dim[axis]) {
            return true; 
        }
        coords[axis] = 0;
    }

    return false;
}

test_status test_tensor_layout() {
    // --- w/o padding ---
    {
        size_t expected_strides[] = {8, 4, 1};
        struct lg_ir_tensor ten = { .desc.dim = {3, 2, 4}, .desc.rank = 3 };
        test_assert(LG_DescComputeLayoutStrides(&ten.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.desc.rank == 3, "got tensor rank %lu", ten.desc.rank);
        test_assert_array_eq(expected_strides, ten.desc.strides, 3, "%lu");
    }

    // --- w/ padding ---
    {
        size_t expected_strides[4] = {224, 32, 8, 1};
        struct lg_ir_tensor ten = { .desc.dim = {2, 7, 4, 3}, .desc.rank = 4 };
        test_assert(LG_DescComputeLayoutStrides(&ten.desc, LG_LAYOUT_ROW_MAJOR, 8) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.desc.rank == 4, "got tensor rank %lu", ten.desc.rank);
        test_assert_array_eq(expected_strides, ten.desc.strides, 3, "%lu");
    }
    
    // --- w/o padding ---
    {
        size_t expected_strides[] = {1, 4, 8};
        struct lg_ir_tensor ten = { .desc.dim = {3, 2, 4}, .desc.rank = 3 };
        test_assert(LG_DescComputeLayoutStrides(&ten.desc, LG_LAYOUT_COL_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.desc.rank == 3, "got tensor rank %lu", ten.desc.rank);
        test_assert_array_eq(expected_strides, ten.desc.strides, 3, "%lu");
    }

    // --- w/ padding ---
    {
        size_t expected_strides[] = {1, 8, 32, 224};
        struct lg_ir_tensor ten = { .desc.dim = {2, 7, 4, 3}, .desc.rank = 4 };
        test_assert(LG_DescComputeLayoutStrides(&ten.desc, LG_LAYOUT_COL_MAJOR, 8) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.desc.rank == 4, "got tensor rank %lu", ten.desc.rank);
        test_assert_array_eq(expected_strides, ten.desc.strides, 3, "%lu");
    }

    return TEST_STATUS_OK;
}

test_status test_tensor_size() {
    struct lg_ir_tensor _36 = {
        .desc.rank = 2, 
        .desc.dim = {6, 6},
        .desc.strides = {6, 1},
        .data = NULL,
    };
    size_t _36_size = LG_DescSizeInBytes(_36.desc);
    test_assert(_36_size == 36 * sizeof(lg_scalar), "tensor size was %lu", _36_size);

    struct lg_ir_tensor also_36 = {
        .desc.rank = 3,
        .desc.dim = {6, 2, 3},
        .desc.strides = {6, 3, 1},
        .data = NULL,
    };
    size_t also_36_size = LG_DescSizeInBytes(also_36.desc);
    test_assert(also_36_size == 36 * sizeof(lg_scalar), "tensor size was %lu", also_36_size);

    struct lg_ir_tensor padded = { .desc.dim = {3, 3, 3}, .desc.rank = 3 };
    test_assert(LG_DescComputeLayoutStrides(&padded.desc, LG_LAYOUT_ROW_MAJOR, 4) == LG_STATUS_OK, "failed to initialize tensor");
    size_t calculated_bytes = LG_DescSizeInBytes(padded.desc);
    // Strides should be (12, 4, 1), meaning the maximum offset at (2, 2, 2) is
    // (12 + 4 + 1) * 2, so the max size is one more than that.
    size_t expected_bytes = ((12 + 4 + 1) * 2 + 1) * sizeof(lg_scalar);
    test_assert(
        calculated_bytes == expected_bytes,
        "tensor size calculated to be %lu bytes, wanted %lu",
        calculated_bytes,
        expected_bytes
    );

    struct lg_ir_tensor zero_ten = {0};
    test_assert(LG_DescComputeLayoutStrides(&zero_ten.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    calculated_bytes = LG_DescSizeInBytes(zero_ten.desc);
    test_assert(calculated_bytes == 0, "tensor size calculated to be %lu bytes", calculated_bytes);

    struct lg_ir_tensor scalar = { .desc.dim = {1}, .desc.rank = 1 };
    test_assert(LG_DescComputeLayoutStrides(&scalar.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    calculated_bytes = LG_DescSizeInBytes(scalar.desc);
    test_assert(calculated_bytes == sizeof(lg_scalar), "tensor size calculated to be %lu bytes", calculated_bytes);

    return TEST_STATUS_OK;
}

test_status test_alloc_tensor() {
    mock_allocator_context ctx = {0};
    struct lg_allocator allocator = {
        .ctx = (void*)&ctx,
        .Alloc = mock_alloc,
        .Free = NULL,
    }; 
    struct lg_ir_tensor ten = {
        .desc.rank = 2, 
        .desc.dim = {6, 6},
        .desc.strides = {6, 1},
        .data = NULL,
    };

    // If we're aligning to a 5 * sizeof(lg_scalar)-byte boundary and starting at 5, then
    // the data allocation should end at 41 * sizeof(lg_scalar),
    // and the gradient allocation should start at the nearest multiple of 5,
    // which is 40 * sizeof(lg_scalar) if lg_scalar is float, yielding 4 * sizeof(lg_scalar) bytes of padding.
    size_t expected_bytes = 36 * sizeof(lg_scalar);

    size_t calculated_one_bytes = LG_DescSizeInBytes(ten.desc);
    test_assert(calculated_one_bytes == expected_bytes, "tensor size calculated to be %lu bytes", calculated_one_bytes);
    test_assert(LG_AllocTensor(&allocator, &ten) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(ten.data == (lg_scalar*)ALLOC_ADDR, "allocated data at address %lu, expected %lu", ten.data, (lg_scalar*)ALLOC_ADDR);
    test_assert(ctx.bytes_allocated == expected_bytes, "allocated %lu bytes, wanted %lu bytes" , ctx.bytes_allocated, expected_bytes);

    return TEST_STATUS_OK;
}

test_status test_tensor_aligned_views_not_compatible() {
    // 4 != 5, so these should clash and not be compatible
    struct lg_ir_tensor x0 = { .desc.dim =  {4, 4}, .desc.rank = 2 };
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    struct lg_ir_tensor x1 = { .desc.dim =  {6, 5, 4}, .desc.rank = 3 };
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");

    enum lg_status status = LG_ComputeBroadcastedAxes((struct lg_desc*[]){&x0.desc, &x1.desc}, 2);
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
    struct lg_ir_tensor x0 = { .desc.dim = {6, 4, 4}, .desc.rank = 3 };
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    // This is a mat44.
    // In memory, with no alignment, this should be a contiguous
    // row-major 2d array (these are (x, y) pairs, not matrix coords):
    // { (0, 0), (0, 1) ... (1, 0), (1, 1) ... (m-1, n-1) }
    struct lg_ir_tensor x1 = { .desc.dim = {4 ,4}, .desc.rank = 2 };
    test_assert(LG_DescComputeLayoutStrides(&x1.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");

    test_assert(x1.desc.strides[0] == 4, "got first stride of %lu", x0.desc.strides[0]);
    test_assert(x1.desc.strides[1] == 1, "got second stride of %lu", x0.desc.strides[1]);

    // Coalesced dimensions
    size_t expected_strides_a[] = {16, 4, 1};
    size_t expected_strides_b[] = {0, 4, 1};

    test_assert(LG_ComputeBroadcastedAxes((struct lg_desc*[]){&x0.desc, &x1.desc}, 2) == LG_STATUS_OK, "failed to broadcast tensors");
    test_assert(LG_SortAxes((struct lg_desc*[]){&x0.desc, &x1.desc}, 2) == LG_STATUS_OK, "failed to sort dmis");
    test_assert_array_eq(expected_strides_a, x0.desc.strides, 3, "%llu");
    test_assert_array_eq(expected_strides_b, x1.desc.strides, 3, "%llu");
    test_assert(LG_CoalesceAxes((struct lg_desc*[]){&x0.desc, &x1.desc}, 2) == LG_STATUS_OK, "failed to coalesce dims");
    test_assert(x0.desc.strides[0] == 1, "%lu");
    
    return TEST_STATUS_OK;
}

void *alloc_libc(void *_, size_t bytes) {
    (void)_;
    return calloc(bytes, 1);
}

void free_libc(void* _, void *ptr) {
    (void)_;
    return free(ptr);
}

test_status test_cpu_add_basic() {
    struct lg_ir_tensor y = { .desc.dim = {4, 4, 12}, .desc.rank = 3 },
              x0 = { .desc.dim = {4, 1, 12}, .desc.rank = 3 },
              x1 = { .desc.dim = {1, 4, 12}, .desc.rank = 3 };
    struct lg_allocator allocator = {
        .Alloc = alloc_libc,
        .Free = free_libc,
    };

    test_assert(LG_DescComputeLayoutStrides(&y.desc, LG_LAYOUT_ROW_MAJOR, 7) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x1.desc, LG_LAYOUT_COL_MAJOR, 2) == LG_STATUS_OK, "failed to lay out tensor");

    test_assert(LG_AllocTensor(&allocator, &y) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(LG_AllocTensor(&allocator, &x0) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(LG_AllocTensor(&allocator, &x1) == LG_STATUS_OK, "failed to allocate tensor");

    test_assert(LG_ComputeBroadcastedAxes(((struct lg_desc*[]){&y.desc, &x0.desc, &x1.desc}), 3) == LG_STATUS_OK, "failed to broadcast tensors");
    test_assert(LG_SortAxes(((struct lg_desc*[]){&y.desc, &x0.desc, &x1.desc}), 3) == LG_STATUS_OK, "failed to sort dims");

    size_t coords[LG_MAX_RANK] = {0};
    do {
        size_t x0_idx = 0;
        size_t x1_idx = 0;
        for (size_t i = 0; i < y.desc.rank; i++) {
            x0_idx += x0.desc.strides[i] * coords[i];
            x1_idx += x1.desc.strides[i] * coords[i];
        }
        x0.data[x0_idx] = 1.0f;
        x1.data[x1_idx] = 2.0f;
    } while (increment_coords_rtl(coords, y.desc.dim, y.desc.rank));

    LG_RT_CPU_Add(y.desc, y.data, x0.desc, x0.data, x1.desc, x1.data);

    for (size_t i = 0; i < LG_MAX_RANK; i++) {
        coords[i] = 0;
    }

    do {
        size_t idx = 0;
        for (size_t i = 0; i < y.desc.rank; i++) {
            idx += y.desc.strides[i] * coords[i];
        }
        test_assert(y.data[idx] == 3.0f, "wanted 3, got %f", y.data[idx]);
    } while (increment_coords_rtl(coords, y.desc.dim, y.desc.rank));;;

    LG_FreeTensor(&allocator, &y);
    LG_FreeTensor(&allocator, &x0);
    LG_FreeTensor(&allocator, &x1);

    return TEST_STATUS_OK;
}

test_status test_cpu_add_vec() {
    struct lg_ir_tensor y = { .desc.dim = {3}, .desc.rank = 1 },
              x0 = { .desc.dim = {3}, .desc.rank = 1 },
              x1 = { .desc.dim = {3}, .desc.rank = 1 };
    struct lg_allocator allocator = {
        .Alloc = alloc_libc,
        .Free = free_libc,
    };

    test_assert(LG_DescComputeLayoutStrides(&y.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x1.desc, LG_LAYOUT_COL_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    test_assert(LG_AllocTensor(&allocator, &y) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(LG_AllocTensor(&allocator, &x0) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(LG_AllocTensor(&allocator, &x1) == LG_STATUS_OK, "failed to allocate tensor");

    test_assert(LG_ComputeBroadcastedAxes(((struct lg_desc*[]){&y.desc, &x0.desc, &x1.desc}), 3) == LG_STATUS_OK, "failed to broadcast tensors");

    size_t coords[LG_MAX_RANK] = {0};
    do {
        size_t x0_idx = 0;
        size_t x1_idx = 0;
        for (size_t i = 0; i < y.desc.rank; i++) {
            x0_idx += x0.desc.strides[i] * coords[i];
            x1_idx += x1.desc.strides[i] * coords[i];
        }
        x0.data[x0_idx] = 1.0f;
        x1.data[x1_idx] = 2.0f;
    } while (increment_coords_rtl(coords, y.desc.dim, y.desc.rank));

    LG_RT_CPU_Add(y.desc, y.data, x0.desc, x0.data, x1.desc, x1.data);

    for (size_t i = 0; i < LG_MAX_RANK; i++) {
        coords[i] = 0;
    }

    do {
        size_t idx = 0;
        for (size_t i = 0; i < y.desc.rank; i++) {
            idx += y.desc.strides[i] * coords[i];
        }
        test_assert(y.data[idx] == 3.0f, "wanted 3, got %f", y.data[idx]);
    } while (increment_coords_rtl(coords, y.desc.dim, y.desc.rank));;;

    LG_FreeTensor(&allocator, &y);
    LG_FreeTensor(&allocator, &x0);
    LG_FreeTensor(&allocator, &x1);

    return TEST_STATUS_OK;
}

test_status test_cpu_matmul() {
    struct lg_ir_tensor y = { .desc.dim = {2, 2}, .desc.rank = 2 },
              x0 = { .desc.dim = {2, 2}, .desc.rank = 2 },
              x1T = { .desc.dim = {2, 2}, .desc.rank = 2 };

    test_assert(LG_DescComputeLayoutStrides(&y.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x1T.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    {
        size_t expected_dim[] = {2, 2};
        size_t expected_strides[] = {2, 1};
        
        test_assert_array_eq(expected_dim, y.desc.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, y.desc.strides, 2, "%lu");
        test_assert_array_eq(expected_dim, x0.desc.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, x0.desc.strides, 2, "%lu");
        test_assert_array_eq(expected_dim, x1T.desc.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, x1T.desc.strides, 2, "%lu");
    }

    lg_scalar x0_data[4] = {1, 2, 3, 4};
    lg_scalar x1_data[4] = {5, 6, 7, 8};
    lg_scalar y_data[4] = {0};

    x0.data = x0_data;
    x1T.data = x1_data;
    y.data = y_data;
    struct lg_ir_tensor y_cpy = y;

    test_assert(LG_ComputeContractedAxes(&y_cpy.desc, &x0.desc, &x1T.desc, 0) == LG_STATUS_OK, "failed to contract output dims");

    {
        size_t expected_dim[] = {2, 2, 2};
        size_t y_expected_strides[] = {2, 1, 0};
        size_t x0_expected_strides[] = {2, 0, 1};
        size_t x1_expected_strides[] = {0, 1, 2};

        test_assert(y_cpy.desc.rank == 3, "y rank: %lu", y_cpy.desc.rank);
        test_assert(x0.desc.rank == 3, "x0 rank: %lu", x0.desc.rank);
        test_assert(x1T.desc.rank == 3, "x1T rank: %lu", x1T.desc.rank);

        test_assert_array_eq(expected_dim, y_cpy.desc.dim, 3, "%lu");
        test_assert_array_eq(y_expected_strides, y_cpy.desc.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, x0.desc.dim, 3, "%lu");
        test_assert_array_eq(x0_expected_strides, x0.desc.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, x1T.desc.dim, 3, "%lu");
        test_assert_array_eq(x1_expected_strides, x1T.desc.strides, 3, "%lu");
    }

    lg_scalar expected_out[] = {19, 22, 43, 50};

    LG_RT_CPU_Contract(y_cpy.desc, y_cpy.data, x0.desc, x0.data, x1T.desc, x1T.data);
    test_assert_array_eq(expected_out, y.data, 4, "%f");

    return TEST_STATUS_OK;
}

test_status test_cpu_matmul_batch() {
    struct lg_ir_tensor y = { .desc.dim = {2, 2, 2}, .desc.rank = 3 },
              x0 = { .desc.dim = {2, 2, 2}, .desc.rank = 3 },
              x1T = { .desc.dim = {2, 2, 2}, .desc.rank = 3 };

    test_assert(LG_DescComputeLayoutStrides(&y.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x1T.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    {
        size_t expected_dim[] = {2, 2, 2};
        size_t expected_strides[] = {4, 2, 1};
        
        test_assert_array_eq(expected_dim, y.desc.dim, 3, "%lu");
        test_assert_array_eq(expected_strides, y.desc.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, x0.desc.dim, 3, "%lu");
        test_assert_array_eq(expected_strides, x0.desc.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, x1T.desc.dim, 3, "%lu");
        test_assert_array_eq(expected_strides, x1T.desc.strides, 3, "%lu");
    }

    lg_scalar x0_data[8] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
    };
    lg_scalar x1_data[8] = {
        5, 6, 7, 8,
        5, 6, 7, 8,
    };
    lg_scalar y_data[8] = {0};

    x0.data = x0_data;
    x1T.data = x1_data;
    y.data = y_data;
    struct lg_ir_tensor y_cpy = y;

    test_assert(LG_ComputeContractedAxes(&y_cpy.desc, &x0.desc, &x1T.desc, 1) == LG_STATUS_OK, "failed to contract output dims");

    {
        size_t expected_dim[] = {2, 2, 2, 2};
        size_t y_expected_strides[] = {4, 2, 1, 0};
        size_t x0_expected_strides[] = {4, 2, 0, 1};
        size_t x1_expected_strides[] = {4, 0, 1, 2};

        test_assert(y_cpy.desc.rank == 4, "y rank: %lu", y_cpy.desc.rank);
        test_assert(x0.desc.rank == 4, "x0 rank: %lu", x0.desc.rank);
        test_assert(x1T.desc.rank == 4, "x1T rank: %lu", x1T.desc.rank);

        test_assert_array_eq(expected_dim, y_cpy.desc.dim, 4, "%lu");
        test_assert_array_eq(y_expected_strides, y_cpy.desc.strides, 4, "%lu");
        test_assert_array_eq(expected_dim, x0.desc.dim, 4, "%lu");
        test_assert_array_eq(x0_expected_strides, x0.desc.strides, 4, "%lu");
        test_assert_array_eq(expected_dim, x1T.desc.dim, 4, "%lu");
        test_assert_array_eq(x1_expected_strides, x1T.desc.strides, 4, "%lu");
    }

    lg_scalar expected_out[] = {
        19, 22, 43, 50,
        19, 22, 43, 50,
    };

    LG_RT_CPU_Contract(y_cpy.desc, y_cpy.data, x0.desc, x0.data, x1T.desc, x1T.data);
    test_assert_array_eq(expected_out, y.data, 8, "%f");

    return TEST_STATUS_OK;
}

test_status test_expr_alloc() {
    struct lg_allocator allocator = {
        .Alloc = alloc_libc,
        .Free = free_libc,
    };

    struct lg_ir_expr expr = {0};
    uint8_t *expr_buf;
    size_t expr_fields_bytes_allocated;
    test_assert(LG_AllocExpr(&allocator, &expr_buf, &expr_fields_bytes_allocated, &expr, 32) == LG_STATUS_OK, "failed to allocate expr");
    test_assert(0 < expr_fields_bytes_allocated, "failed to allocate expr");

    struct lg_ir_tensor x0 = { .desc.rank = 1, .desc.dim = {4} },
              x1 = { .desc.rank = 1, .desc.dim = {4} };
    
    test_assert(LG_DescComputeLayoutStrides(&x0.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(LG_DescComputeLayoutStrides(&x1.desc, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    lg_scalar x0_vals[4] = {1, 2, 3, 4},
              x1_vals[4] = {3, 3, 1, 4};
    test_assert(sizeof(x0_vals) == LG_DescSizeInBytes(x0.desc), "wrong size");
    test_assert(sizeof(x1_vals) == LG_DescSizeInBytes(x1.desc), "wrong size");
    x0.data = x0_vals;
    x1.data = x1_vals;
    
    struct lg_ir_tensor y0;
    test_assert(LG_IR_AppendAdd(&expr, &y0, x0, x1) == LG_STATUS_OK, "failed to append add node");

    struct lg_ir_tensor y1;
    test_assert(LG_IR_AppendAdd(&expr, &y1, y0, x1) == LG_STATUS_OK, "failed to append add node");

    test_assert(LG_IR_CompileExpr(&expr, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to compile expr");

    lg_scalar *data_buf;
    test_assert(LG_AllocExprData(&allocator, &allocator, &data_buf, NULL, &expr) == LG_STATUS_OK, "failed to allocate expr");
    test_assert(data_buf == expr.nodes[0].y.data, "wanted %p; got %p", data_buf, expr.nodes[0].y.data);
    // test_assert(data_buf == expr.y[1].data, "wanted %p; got %p", data_buf, expr.y[1].data); 
    // fuck you, gcc ubsan. corrupts the fucking pointer
    test_assert(LG_RT_CPU_ExecExpr(expr) == LG_STATUS_OK, "failed to exectute expr");

    // The buffers alias, so . . .
    // x0 + x1 = y0 = {4, 5, 4, 8};
    // y0 + x1 = y1 = ...;
    lg_scalar expected_data[4] = {7, 8, 5, 12};
    test_assert_array_eq(expected_data, expr.nodes[0].y.data, 4, "%f");
    // test_assert_array_eq(expected_data, expr.y[1].data, 4, "%f");
    // still corrupts the fucking pointer

    free(expr_buf);
    free(data_buf);

    return TEST_STATUS_OK;
}

int main(void) {
    test_run(tensor_layout);
    test_run(tensor_size);
    test_run(alloc_tensor);
    test_run(tensor_aligned_views_not_compatible);
    test_run(tensor_aligned_views);
    test_run(cpu_add_basic);
    test_run(cpu_add_vec);
    test_run(cpu_matmul);
    test_run(cpu_matmul_batch);
    test_run(expr_alloc);
    return 0;
}
