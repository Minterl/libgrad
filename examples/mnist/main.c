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
    struct arena arena = {0};
    assert(ArenaInit(&arena, ARENA_CAP) == 0);
    struct lg_allocator allocator = ArenaAsLgAllocator(&arena);

    enum lg_status status = LG_STATUS_OK;

    struct lg_ir_expr expr = {0};
    status = LG_AllocExpr(&allocator, NULL, NULL, &expr, EXPR_CAP, EXPR_CAP);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        return status;
    }

    struct lg_ir_symbol x = {0};
    status = LG_IR_DeclareSource(&x, (struct lg_desc){
        .rank = 1,
        .dim = {28 * 28},
    }, &expr, 100);
    struct lg_ir_symbol W_0 = {0};
    status = LG_IR_DeclareSource(&x, (struct lg_desc){
        .rank = 2,
        .dim = {128, 28 * 28},
    }, &expr, 101);
    struct lg_ir_symbol b_0 = {0};
    status = LG_IR_DeclareSource(&x, (struct lg_desc){
        .rank = 1,
        .dim = {128},
    }, &expr, 101);

    struct lg_ir_symbol y_0 = {0};
    status = LG_IR_AppendContract(&expr, &y_0, W_0, x, 1, 0);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }
    struct lg_ir_symbol y_1 = {0};
    status = LG_IR_AppendAdd(&expr, &y_1, y_0, b_0);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = LG_IR_DeclareSink(y_1, &expr);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    status = LG_IR_CompileExpr(NULL, &allocator, &expr, 16);
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

    uint32_t y1_buf_id = 0;
    size_t y1_offset = 0;
    struct lg_desc y1_desc = {0};
    status = LG_IR_GetSinkLocation(
        &y1_buf_id,
        &y1_offset,
        &y1_desc,
        y_1,
        &expr
    );
    if (status != LG_STATUS_OK) {
        FAILF("status: %d", status);
        goto out;
    }

out:
    ArenaDestroy(&arena);

    return status;
}
