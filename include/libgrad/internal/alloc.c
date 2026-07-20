#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

enum lg_status LG__AllocContiguousBlocks(
    struct lg_allocator *alloc,
    uint8_t **out_ptrs,
    size_t *out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        size += LG__ALIGN_UP(sizes[i], align);
    }

    uint8_t *ptr = alloc->Alloc(alloc->ctx, size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < n; i++) {
        out_ptrs[i] = ptr + current_offset;
        current_offset += LG__ALIGN_UP(sizes[i], align);
    }

    return LG_STATUS_OK;
}

enum lg_status LG_AllocExpr(
    struct lg_allocator *allocator,
    uint8_t **out_ptr,
    size_t *out_bytes_allocated,
    struct lg_ir_expr *expr,
    size_t cap
) {
    const size_t size = cap * sizeof(struct lg_ir_expr_node);
    
    uint8_t *buf = allocator->Alloc(allocator->ctx, size);
    if (buf == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }
    if (out_ptr != NULL) {
        *out_ptr = buf;
    }
    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    expr->nodes = (struct lg_ir_expr_node*)buf;
    expr->cap = cap;
    expr->len = 0;

    return LG_STATUS_OK;
}

void LG_FreeExpr(struct lg_allocator *allocator, struct lg_ir_expr *expr) {
    allocator->Free(allocator->ctx, expr->nodes);
    expr->cap = 0;
    expr->len = 0;
    expr->nodes = NULL;
}

