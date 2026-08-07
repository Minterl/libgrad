#include "libgrad/internal/alloc.h"
#include "libgrad/internal/core.h"
#include "libgrad/internal/vm.h"
#define LIBGRAD_IMPLEMENTATION
#include <libgrad/libgrad.h>
// #ifndef LG_CPU_IMPLEMENTATION
// #   define LG_CPU_IMPLEMENTATION
// #endif // LG_CPU_IMPLEMENTATION
// #include <libgrad/cpu.h>

#include <assert.h>
#include <common/arena.h>
#include <common/macros.h>

#define ARENA_CAP 100 * 1024 * 1024
#define EXPR_CAP 32

int main(void) {
    Arena arena = {0};
    assert(ArenaInit(&arena, ARENA_CAP) == 0);
    LG_Allocator allocator = ArenaAsLgAllocator(&arena);

    LG_StatusKind status = LG_StatusKind_OK;

    LG_Expr expr = {0};
    status = lg_alloc_expr(&allocator, NULL, NULL, &expr, EXPR_CAP, EXPR_CAP);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }

    status = lg_buftab_insert(&expr, 100);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }
    status = lg_buftab_insert(&expr, 101);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }
    status = lg_buftab_insert(&expr, 102);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }

    // TODO: this should be given its own initializer
    LG_CompilationContext ctx = {
        .expr = &expr,
        .scratch = &allocator,
    };
    status = lg_symtab_init(&ctx.symtab, &allocator, EXPR_CAP);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }

    LG_Symbol x = {0};
    status = lg_declare_source(&ctx, &x, (LG_StridedDesc){
        .rank = 1,
        .dim = {28 * 28},
    }, 100);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }
    LG_Symbol W_0 = {0};
    status = lg_declare_source(&ctx, &W_0, (LG_StridedDesc){
        .rank = 2,
        .dim = {128, 28 * 28},
    }, 101);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }
    LG_Symbol b_0 = {0};
    status = lg_declare_source(&ctx, &b_0, (LG_StridedDesc){
        .rank = 1,
        .dim = {128},
    }, 102);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        return status;
    }

    LG_Symbol y_0 = {0};
    status = lg_append_contract(&ctx, &y_0, W_0, x, 1, 0);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        goto out;
    }
    LG_Symbol y_1 = {0};
    status = lg_append_add(&ctx, &y_1, y_0, b_0);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = lg_declare_sink(&ctx,y_1);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = lg_compile_expr(&ctx, 16);
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    uint32_t y1_buf_id = 0;
    size_t y1_offset = 0;
    LG_StridedDesc y1_desc = {0};
    status = lg_get_sink_location(
        &y1_buf_id,
        &y1_offset,
        &y1_desc,
        y_1,
        &expr
    );
    if (status != LG_StatusKind_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    printf("status: %d\n", status);

    for (size_t i = 0; i < expr.nodes_len; i++) {
        printf("y.offset = %lu ", expr.nodes[i].y_offset);
        printf("x0.offset = %lu ", expr.nodes[i].x0_offset);
        printf("x1.offset = %lu\n", expr.nodes[i].x1_offset);
    }

out:
    ArenaDestroy(&arena);

    return status;
}
