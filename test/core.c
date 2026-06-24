#ifndef LG_CORE_IMPLEMENTATION
#define LG_CORE_IMPLEMENTATION
#include <asm-generic/errno-base.h>
#endif // LG_CORE_IMPLEMENTATION
#ifndef LG_ALLOC_IMPLEMENTATION
#define LG_ALLOC_IMPLEMENTATION
#endif // LG_ALLOC_IMPLEMENTATION
#ifndef LG_CPU_IMPLEMENTATION
#define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#ifndef LG_INTERNAL_DEBUG_IMPLEMENTATION
#define LG_INTERNAL_DEBUG_IMPLEMENTATION
#endif // LG_INTERNAL_DEBUG_IMPLEMENTATION
 
#include <stdbool.h>
#include <stdlib.h>
#include <libgrad/core.h>
#include <libgrad/alloc.h>
#include <libgrad/cpu.h>
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

static inline bool increment_coords_rtl(lg_size *coords, const lg_size *dim, lg_size rank) {
    if (rank == 0) return false;

    lg_size axis = rank;
    
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
        lg_size expected_strides[] = {8, 4, 1};
        lg_tensor ten = { .dim = {3, 2, 4}, .rank = 3 };
        test_assert(lg_tensor_layout(&ten, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.rank == 3, "got tensor rank %lu", ten.rank);
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }

    // --- w/ padding ---
    {
        lg_size expected_strides[4] = {224, 32, 8, 1};
        lg_tensor ten = { .dim = {2, 7, 4, 3}, .rank = 4 };
        test_assert(lg_tensor_layout(&ten, LG_LAYOUT_ROW_MAJOR, 8) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.rank == 4, "got tensor rank %lu", ten.rank);
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }
    
    // --- w/o padding ---
    {
        lg_size expected_strides[] = {1, 4, 8};
        lg_tensor ten = { .dim = {3, 2, 4}, .rank = 3 };
        test_assert(lg_tensor_layout(&ten, LG_LAYOUT_COL_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.rank == 3, "got tensor rank %lu", ten.rank);
        test_assert_array_eq(expected_strides, ten.strides, 3, "%lu");
    }

    // --- w/ padding ---
    {
        lg_size expected_strides[] = {1, 8, 32, 224};
        lg_tensor ten = { .dim = {2, 7, 4, 3}, .rank = 4 };
        test_assert(lg_tensor_layout(&ten, LG_LAYOUT_COL_MAJOR, 8) == LG_STATUS_OK, "failed to initialize tensor");
        test_assert(ten.rank == 4, "got tensor rank %lu", ten.rank);
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

    lg_tensor padded = { .dim = {3, 3, 3}, .rank = 3 };
    test_assert(lg_tensor_layout(&padded, LG_LAYOUT_ROW_MAJOR, 4) == LG_STATUS_OK, "failed to initialize tensor");
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
    test_assert(lg_tensor_layout(&zero_ten, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    calculated_bytes = lg_tensor_size_bytes(zero_ten);
    test_assert(calculated_bytes == 0, "tensor size calculated to be %lu bytes", calculated_bytes);

    lg_tensor scalar = { .dim = {1}, .rank = 1 };
    test_assert(lg_tensor_layout(&scalar, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
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
    lg_tensor a = { .dim =  {4, 4}, .rank = 2 };
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    lg_tensor b = { .dim =  {6, 5, 4}, .rank = 3 };
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");

    lg_status status = lg_tensor_broadcast((lg_tensor*[]){&a, &b}, 2);
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
    lg_tensor a = { .dim = {6, 4, 4}, .rank = 3 };
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");
    // This is a mat44.
    // In memory, with no alignment, this should be a contiguous
    // row-major 2d array (these are (x, y) pairs, not matrix coords):
    // { (0, 0), (0, 1) ... (1, 0), (1, 1) ... (m-1, n-1) }
    lg_tensor b = { .dim = {4 ,4}, .rank = 2 };
    test_assert(lg_tensor_layout(&b, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to initialize tensor");

    test_assert(b.strides[0] == 4, "got first stride of %lu", a.strides[0]);
    test_assert(b.strides[1] == 1, "got second stride of %lu", a.strides[1]);

    // Coalesced dimensions
    lg_size expected_strides_a[] = {16, 4, 1};
    lg_size expected_strides_b[] = {0, 4, 1};

    test_assert(lg_tensor_broadcast((lg_tensor*[]){&a, &b}, 2) == LG_STATUS_OK, "failed to broadcast tensors");
    test_assert(lg_tensor_sort_dims((lg_tensor*[]){&a, &b}, 2) == LG_STATUS_OK, "failed to sort dmis");
    test_assert_array_eq(expected_strides_a, a.strides, 3, "%llu");
    test_assert_array_eq(expected_strides_b, b.strides, 3, "%llu");
    test_assert(lg_tensor_coalesce_dims((lg_tensor*[]){&a, &b}, 2) == LG_STATUS_OK, "failed to coalesce dims");
    test_assert(a.strides[0] == 1, "%lu");
    
    return TEST_STATUS_OK;
}

void *alloc_libc(void *_, lg_size bytes) {
    (void)_;
    return calloc(bytes, 1);
}

void free_libc(void* _, void *ptr) {
    (void)_;
    return free(ptr);
}

test_status test_cpu_add_basic() {
    lg_tensor out = { .dim = {4, 4, 12}, .rank = 3 },
              a = { .dim = {4, 1, 12}, .rank = 3 },
              b = { .dim = {1, 4, 12}, .rank = 3 };
    lg_allocator allocator = {
        .alloc = alloc_libc,
        .free = free_libc,
    };

    test_assert(lg_tensor_layout(&out, LG_LAYOUT_ROW_MAJOR, 7) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&b, LG_LAYOUT_COL_MAJOR, 2) == LG_STATUS_OK, "failed to lay out tensor");

    test_assert(lg_alloc_tensor(&allocator, &out, false) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &a, false) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &b, false) == LG_STATUS_OK, "failed to allocate tensor");

    test_assert(lg_tensor_broadcast(((lg_tensor*[]){&out, &a, &b}), 3) == LG_STATUS_OK, "failed to broadcast tensors");
    test_assert(lg_tensor_sort_dims(((lg_tensor*[]){&out, &a, &b}), 3) == LG_STATUS_OK, "failed to sort dims");

    lg_size coords[LG_MAX_RANK] = {0};
    do {
        lg_size a_idx = 0;
        lg_size b_idx = 0;
        for (lg_size i = 0; i < out.rank; i++) {
            a_idx += a.strides[i] * coords[i];
            b_idx += b.strides[i] * coords[i];
        }
        a.data[a_idx] = 1.0f;
        b.data[b_idx] = 2.0f;
    } while (increment_coords_rtl(coords, out.dim, out.rank));

    test_assert(lg_cpu_add(out, a, b) == LG_STATUS_OK, "failed to add");

    for (lg_size i = 0; i < LG_MAX_RANK; i++) {
        coords[i] = 0;
    }

    do {
        lg_size idx = 0;
        for (lg_size i = 0; i < out.rank; i++) {
            idx += out.strides[i] * coords[i];
        }
        test_assert(out.data[idx] == 3.0f, "wanted 3, got %f", out.data[idx]);
    } while (increment_coords_rtl(coords, out.dim, out.rank));;;

    lg_free_tensor(&allocator, &out);
    lg_free_tensor(&allocator, &a);
    lg_free_tensor(&allocator, &b);

    return TEST_STATUS_OK;
}

test_status test_cpu_add_vec() {
    lg_tensor out = { .dim = {3}, .rank = 1 },
              a = { .dim = {3}, .rank = 1 },
              b = { .dim = {3}, .rank = 1 };
    lg_allocator allocator = {
        .alloc = alloc_libc,
        .free = free_libc,
    };

    test_assert(lg_tensor_layout(&out, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&b, LG_LAYOUT_COL_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    test_assert(lg_alloc_tensor(&allocator, &out, false) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &a, false) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &b, false) == LG_STATUS_OK, "failed to allocate tensor");

    test_assert(lg_tensor_broadcast(((lg_tensor*[]){&out, &a, &b}), 3) == LG_STATUS_OK, "failed to broadcast tensors");

    lg_size coords[LG_MAX_RANK] = {0};
    do {
        lg_size a_idx = 0;
        lg_size b_idx = 0;
        for (lg_size i = 0; i < out.rank; i++) {
            a_idx += a.strides[i] * coords[i];
            b_idx += b.strides[i] * coords[i];
        }
        a.data[a_idx] = 1.0f;
        b.data[b_idx] = 2.0f;
    } while (increment_coords_rtl(coords, out.dim, out.rank));

    test_assert(lg_cpu_add(out, a, b) == LG_STATUS_OK, "failed to add");

    for (lg_size i = 0; i < LG_MAX_RANK; i++) {
        coords[i] = 0;
    }

    do {
        lg_size idx = 0;
        for (lg_size i = 0; i < out.rank; i++) {
            idx += out.strides[i] * coords[i];
        }
        test_assert(out.data[idx] == 3.0f, "wanted 3, got %f", out.data[idx]);
    } while (increment_coords_rtl(coords, out.dim, out.rank));;;

    lg_free_tensor(&allocator, &out);
    lg_free_tensor(&allocator, &a);
    lg_free_tensor(&allocator, &b);

    return TEST_STATUS_OK;
}

test_status test_cpu_matmul() {
    lg_tensor out = { .dim = {2, 2}, .rank = 2 },
              a = { .dim = {2, 2}, .rank = 2 },
              bT = { .dim = {2, 2}, .rank = 2 };

    test_assert(lg_tensor_layout(&out, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&bT, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    {
        lg_size expected_dim[] = {2, 2};
        lg_size expected_strides[] = {2, 1};
        
        test_assert_array_eq(expected_dim, out.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, out.strides, 2, "%lu");
        test_assert_array_eq(expected_dim, a.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, a.strides, 2, "%lu");
        test_assert_array_eq(expected_dim, bT.dim, 2, "%lu");
        test_assert_array_eq(expected_strides, bT.strides, 2, "%lu");
    }

    lg_dtype a_data[4] = {1, 2, 3, 4};
    lg_dtype b_data[4] = {5, 6, 7, 8};
    lg_dtype out_data[4] = {0};

    a.data = a_data;
    bT.data = b_data;
    out.data = out_data;
    lg_tensor out_cpy = out;

    test_assert(lg_tensor_compute_contracted_dims(&out_cpy, &a, &bT, 0) == LG_STATUS_OK, "failed to contract output dims");

    {
        lg_size expected_dim[] = {2, 2, 2};
        lg_size out_expected_strides[] = {2, 1, 0};
        lg_size a_expected_strides[] = {2, 0, 1};
        lg_size b_expected_strides[] = {0, 1, 2};

        test_assert(out_cpy.rank == 3, "out rank: %lu", out.rank);
        test_assert(a.rank == 3, "a rank: %lu", a.rank);
        test_assert(bT.rank == 3, "b rank: %lu", bT.rank);

        test_assert_array_eq(expected_dim, out_cpy.dim, 3, "%lu");
        test_assert_array_eq(out_expected_strides, out_cpy.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, a.dim, 3, "%lu");
        test_assert_array_eq(a_expected_strides, a.strides, 3, "%lu");
        test_assert_array_eq(expected_dim, bT.dim, 3, "%lu");
        test_assert_array_eq(b_expected_strides, bT.strides, 3, "%lu");
    }

    lg_dtype expected_out[] = {19, 22, 43, 50};

    test_assert(lg_cpu_contract(out_cpy, a, bT) == LG_STATUS_OK, "failed to add");
    test_assert_array_eq(expected_out, out.data, 4, "%f");

    return TEST_STATUS_OK;
}

test_status test_cpu_backward() {
    const lg_size cap = 32;
    lg_tape tape = {
        .cap = cap,
        .inputs_a = malloc(cap * sizeof(lg_tensor)),
        .inputs_b = malloc(cap * sizeof(lg_tensor)),
        .outputs = malloc(cap * sizeof(lg_tensor)),
        .opcodes = malloc(cap * sizeof(lg_opcode)),
    };

    lg_tensor a = { .rank = 1, .dim = {4} },
              b = { .rank = 1, .dim = {4}  },
              out = { .rank = 1, .dim = {4} };

    test_assert(lg_tensor_layout(&a, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&b, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");
    test_assert(lg_tensor_layout(&out, LG_LAYOUT_ROW_MAJOR, 1) == LG_STATUS_OK, "failed to lay out tensor");

    lg_dtype a_vals[4] = {1, 2, 3, 4},
             b_vals[4] = {3, 3, 1, 4},
             out_grad[4] = {1, 1, 1, 1};

    lg_allocator allocator = {
        .alloc = alloc_libc,
        .free = free_libc,
    };

    test_assert(lg_alloc_tensor(&allocator, &out, true) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &a, true) == LG_STATUS_OK, "failed to allocate tensor");
    test_assert(lg_alloc_tensor(&allocator, &b, true) == LG_STATUS_OK, "failed to allocate tensor");

    lg_tensor_copy_vector(a, a_vals, 0, false);
    lg_tensor_copy_vector(b, b_vals, 0, false);
    lg_tensor_copy_vector(out, out_grad, 0, true);

    test_assert(lg_add(&tape, out, a, b) == LG_STATUS_OK, "failed to append add node");

    test_assert(lg_cpu_forward(tape) == LG_STATUS_OK, "failed to do forward pass");
    test_assert(lg_cpu_backward(tape) == LG_STATUS_OK, "failed to do backward pass");

    lg_dtype expected_data[4] = {4, 5, 4, 8};
    lg_dtype expected_grad[4] = {1, 1, 1, 1};
    test_assert_array_eq(expected_data, out.data, 4, "%f");
    test_assert_array_eq(expected_grad, a.grad, 4, "%f");

    lg_free_tensor(&allocator, &a);
    lg_free_tensor(&allocator, &b);
    lg_free_tensor(&allocator, &out);

    free(tape.inputs_a);
    free(tape.inputs_b);
    free(tape.outputs);
    free(tape.opcodes);

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
    test_run(cpu_backward);
    return 0;
}
