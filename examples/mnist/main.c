#include "libgrad/internal/alloc.h"
#define LIBGRAD_IMPLEMENTATION
#include <libgrad/libgrad.h>
#ifndef LG_CPU_IMPLEMENTATION
#   define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#include <libgrad/cpu.h>

#include <assert.h>
#include <stdio.h>
#include <common/arena.h>
#include <common/macros.h>

#define ARENA_CAP 100 * 1024 * 1024
#define EXPR_CAP 32

int main(void) {
    struct arena alloc = {0};
    assert(arena_init(&alloc, ARENA_CAP) == 0);
    struct lg_allocator libgrad_allocator = arena_as_lg_allocator(&alloc);

    enum lg_status status = LG_STATUS_OK;

    struct lg_expr expr = {0};
    status = lga_AllocExpr(&libgrad_allocator, NULL, NULL, &expr, EXPR_CAP);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        return status;
    }

    struct lg_tensor x = {
        .desc.rank = 1,
        .desc.dim = {28 * 28},
    };
    struct lg_tensor W_0 = {
        .desc.rank = 2,
        .desc.dim = {28 * 28, 128},
    };
    struct lg_tensor b_0 = {
        .desc.rank = 1,
        .desc.dim = {128},
    };

    status = lga_AllocTensor(&libgrad_allocator, &x);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    struct lg_tensor y_0 = {0};
    status = lgvm_Contract(&expr, &y_0, W_0, x, 0);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }
    struct lg_tensor y_1 = {0};
    status = lgvm_Add(&expr, &y_1, y_0, b_0);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = lgvm_Nop(&expr, y_1);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = lgvm_CompileExpr(&expr, LG_LAYOUT_ROW_MAJOR, 1);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    lg_scalar *data = NULL;
    status = lga_AllocExprData(
        &libgrad_allocator,
        &libgrad_allocator,
        &data,
        NULL,
        &expr
    );
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = lgvm_GetLastLocationOfTensor(&expr, &y_1);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }
    
    status = lgrt_cpu_ExecExpr(expr);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    struct lg_nditer iter = {
        .descs = {y_1.desc},
        .n_tracked_dims = y_1.desc.rank,
    };
    do {
        printf("i %lu v %.2f\n", iter.indices[0], y_1.data[iter.indices[0]]);
    } while (lg_nditerIncrement(&iter, y_1.desc.rank - 1));

out:
    arena_destroy(&alloc);

    return status;
}
