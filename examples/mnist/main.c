#define LIBGRAD_IMPLEMENTATION
#include <libgrad/libgrad.h>
#ifndef LG_CPU_IMPLEMENTATION
#   define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#include <libgrad/cpu.h>

#include <assert.h>
#include <common/arena.h>
#include <stdio.h>

#define ARENA_CAP 20 * 1024 * 1024
#define EXPR_CAP 32

int main(void) {
    com_arena alloc = {0};
    assert(com_arena_init(&alloc, ARENA_CAP) == 0);
    lg_allocator libgrad_allocator = com_arena_as_lg_allocator(&alloc);

    lg_expr expr = {
        // TODO: this is also annoying
        .cap = EXPR_CAP,
        .opcodes = (lg_opcode*)arena_alloc(&alloc, EXPR_CAP * sizeof(lg_opcode)),
        .y = (lg_tensor*)arena_alloc(&alloc, EXPR_CAP * sizeof(lg_tensor)),
        .x0 = (lg_tensor*)arena_alloc(&alloc, EXPR_CAP * sizeof(lg_tensor)),
        .x1 = (lg_tensor*)arena_alloc(&alloc, EXPR_CAP * sizeof(lg_tensor)),
    };

    lg_tensor x = {
        .desc.rank = 1,
        .desc.dim = {28 * 28},
    };
    lg_tensor W_0 = {
        .desc.rank = 2,
        .desc.dim = {28 * 28, 128},
    };
    lg_tensor b_0 = {
        .desc.rank = 1,
        .desc.dim = {1, 128},
    };

    lg_status status = LG_STATUS_OK;

    status = lg_alloc_tensor(&libgrad_allocator, &x);
    if (status != LG_STATUS_OK) {
        goto out;
    }

    lg_tensor y_0 = {0};
    status = lg_contract(&expr, &y_0, W_0, x, 0);
    if (status != LG_STATUS_OK) {
        goto out;
    }
    lg_tensor y_1 = {0};
    status = lg_add(&expr, &y_1, y_0, b_0);
    if (status != LG_STATUS_OK) {
        goto out;
    }

    status = lg_expr_compile(&expr);
    if (status != LG_STATUS_OK) {
        goto out;
    }

    status = lg_pin(&expr, &y_1);
    if (status != LG_STATUS_OK) {
        goto out;
    }

    lg_scalar *data = NULL;
    status = lg_alloc_expr(&libgrad_allocator, &libgrad_allocator, &data, &expr);
    if (status != LG_STATUS_OK) {
        goto out;
    }
    
    status = lg_cpu_exec(expr);
    if (status != LG_STATUS_OK) {
        goto out;
    }

    lg_nditer iter = {
        .descs = {y_1.desc},
        .n_tracked_dims = y_1.desc.rank,
    };
    do {
        printf("i %lu v %.2f\n", iter.indices[0], y_1.data[iter.indices[0]]);
    } while (lg_nditer_increment(&iter, y_1.desc.rank - 1));

out:
    com_arena_destroy(&alloc);

    return status;
}
