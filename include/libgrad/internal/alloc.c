#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

enum lg_status LG_AllocTensor(struct lg_allocator *allocator, struct lg_ir_tensor *tensor) {
    size_t one_size = LG_DescSizeInBytes(tensor->desc);
    if (one_size == 0) {
        return LG_STATUS_OK;
    }

    size_t total_size = one_size;

    uint8_t *ptr = allocator->Alloc(allocator->ctx, total_size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    tensor->data = (lg_scalar*)ptr;

    return LG_STATUS_OK;
}

enum lg_status LG_FreeTensor(struct lg_allocator *allocator, struct lg_ir_tensor *tensor) {
#ifdef LG_SAFE
    if (tensor->data == NULL) {
        return LG_STATUS_NULL_POINTER;
    }
#endif // LG_SAFE
    allocator->Free(allocator->ctx, tensor->data);
    tensor->data = NULL;
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

enum lg_status LG_AllocExprData(
    struct lg_allocator *perm,
    struct lg_allocator *scratch,
    lg_scalar **out_ptr,
    size_t *out_bytes_allocated,
    struct lg_ir_expr *expr
) {
    // under the conditions of a cyclomatic complexity
    // of 1 and infinite registers, lsra reaches the global
    // optimum
    // TODO: add alignment
    
    enum lg_status status = LG_STATUS_OK;

    uint8_t *scratch_buf = scratch->Alloc(scratch->ctx, expr->len * 4 * sizeof(size_t));
    if (scratch_buf == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    size_t *sizes = (size_t*)scratch_buf;
    size_t *dead_after = sizes + expr->len;
    size_t *total_freed_after_time = dead_after + expr->len;
    size_t *offsets = total_freed_after_time + expr->len;

    for (size_t i = 0; i < expr->len; i++) {
        sizes[i] = 0;
        dead_after[i] = 0;
        total_freed_after_time[i] = 0;
        offsets[i] = 0;
    }

    for (size_t i = 0; i < expr->len; i++) {
        if (expr->nodes[i].x0.data == NULL) {
            dead_after[expr->nodes[i].x0.born_at] = i;
        }
        if (expr->nodes[i].x1.data == NULL) {
            dead_after[expr->nodes[i].x1.born_at] = i;
        }
        sizes[expr->nodes[i].y.born_at] = LG_DescSizeInBytes(expr->nodes[i].y.desc);
    }
    for (size_t i = 0; i < expr->len; i++) {
        total_freed_after_time[dead_after[i]] += sizes[i];
    }

    size_t current_offset = 0;
    size_t max_offset = 0;
    for (size_t time = 0; time < expr->len; time++) {
        offsets[time] = current_offset;
        current_offset += sizes[expr->nodes[time].y.born_at];
        if (current_offset > max_offset) {
            max_offset = current_offset;
        }
        current_offset -= total_freed_after_time[time];
    }

    uint8_t *_data = perm->Alloc(perm->ctx, max_offset);
    if (_data == NULL) {
        status = LG_STATUS_OUT_OF_MEMORY;
        goto out;
    }
    if (out_ptr != NULL) {
        *out_ptr = (lg_scalar*)_data;
    }
    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = max_offset;
    }

    for (size_t i = 0; i < expr->len; i++) {
        expr->nodes[i].y.data = (lg_scalar*)(_data + offsets[expr->nodes[i].y.born_at]);
        if (expr->nodes[i].x0.data == NULL) {
            expr->nodes[i].x0.data = (lg_scalar*)(_data + offsets[expr->nodes[i].x0.born_at]);
        }
        if (expr->nodes[i].x1.data == NULL) {
            expr->nodes[i].x1.data = (lg_scalar*)(_data + offsets[expr->nodes[i].x0.born_at]);
        }
    }

out:
    scratch->Free(scratch->ctx, scratch_buf);
    return status;
}
