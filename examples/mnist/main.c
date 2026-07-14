#include "libgrad/internal/alloc.h"
#include "libgrad/internal/core.h"
#include <stdlib.h>
#define LIBGRAD_IMPLEMENTATION
#include <libgrad/libgrad.h>
#ifndef LG_CPU_IMPLEMENTATION
#   define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#include <libgrad/cpu.h>

#define EXPR_CAP 32

// TODO: write a small arena allocator common to the examples

// TODO: this should really be available by default
void *libc_alloc(void *ctx, size_t bytes) {
    (void)ctx;
    return (void*)calloc(bytes, 1);
}
void libc_free(void *ctx, void *ptr) {
    (void)ctx;
    free(ptr);
}

int main(void) {
    lg_expr expr = {
        // TODO: this is also annoying
        .cap = EXPR_CAP,
        .opcodes = calloc(EXPR_CAP, sizeof(lg_opcode)),
        .y = calloc(EXPR_CAP, sizeof(lg_tensor)),
        .x0 = calloc(EXPR_CAP, sizeof(lg_tensor)),
        .x1 = calloc(EXPR_CAP, sizeof(lg_tensor)),
    };

    lg_allocator allocator = {
        .alloc = libc_alloc,
        .free = libc_free,
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

    status = lg_alloc_tensor(&allocator, &x);
    if (status != LG_STATUS_OK) {
        goto out_free_expr;
    }

    lg_tensor y_0 = {0};
    status = lg_add(&expr, &y_0, W_0, b_0);
    if (status != LG_STATUS_OK) {
        goto out_free_input;
    }

    status = lg_expr_compile(&expr);
    if (status != LG_STATUS_OK) {
        goto out_free_input;
    }

    lg_scalar *data = NULL;
    status = lg_alloc_expr(&allocator, &allocator, &data, &expr);
    if (status != LG_STATUS_OK) {
        goto out_free_input;
    }

    status = lg_pin(&expr, &y_0);
    if (status != LG_STATUS_OK) {
        goto out_free_data;
    }
    
    status = lg_cpu_exec(expr);
    if (status != LG_STATUS_OK) {
        goto out_free_data;
    }

out_free_data:
    free(data);
out_free_input:
    lg_free_tensor(&allocator, &x);
out_free_expr:
    free(expr.opcodes);
    free(expr.y);
    free(expr.x0);
    free(expr.x1);

    return status;
}
