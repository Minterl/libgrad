#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

lg_status lg_alloc_tensor(lg_allocator *allocator, lg_tensor *tensor) {
    lg_size one_size = lg_desc_size_bytes(tensor->desc);
    if (one_size == 0) {
        return LG_STATUS_OK;
    }

    lg_size total_size = one_size;

    uint8_t *ptr = allocator->alloc(allocator->ctx, total_size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    tensor->data = (lg_scalar*)ptr;

    return LG_STATUS_OK;
}

lg_status lg_free_tensor(lg_allocator *allocator, lg_tensor *tensor) {
#ifdef LG_SAFE
    if (tensor->data == NULL) {
        return LG_STATUS_NULL_POINTER;
    }
#endif // LG_SAFE
    allocator->free(allocator->ctx, tensor->data);
    tensor->data = NULL;
    return LG_STATUS_OK;
}

lg_status lg_alloc_expr(
    lg_allocator *allocator,
    uint8_t **out_ptr,
    lg_size *out_bytes_allocated,
    lg_expr *expr,
    lg_size cap
) {
    const lg_size size = cap * sizeof(lg_expr_node);
    
    uint8_t *buf = allocator->alloc(allocator->ctx, size);
    if (buf == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }
    if (out_ptr != NULL) {
        *out_ptr = buf;
    }
    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    expr->nodes = (lg_expr_node*)buf;
    expr->cap = cap;
    expr->len = 0;

    return LG_STATUS_OK;
}

void lg_free_expr(lg_allocator *allocator, lg_expr *expr) {
    allocator->free(allocator->ctx, expr->nodes);
    expr->cap = 0;
    expr->len = 0;
    expr->nodes = NULL;
}

lg_status lg_alloc_expr_data(
    lg_allocator *perm,
    lg_allocator *scratch,
    lg_scalar **out_ptr,
    lg_size *out_bytes_allocated,
    lg_expr *expr
) {
    // under the conditions of a cyclomatic complexity
    // of 1 and infinite registers, lsra reaches the global
    // optimum
    // TODO: add alignment
    
    lg_status status = LG_STATUS_OK;

    uint8_t *scratch_buf = scratch->alloc(scratch->ctx, expr->len * 4 * sizeof(lg_size));
    if (scratch_buf == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    lg_size *sizes = (lg_size*)scratch_buf;
    lg_size *dead_after = sizes + expr->len;
    lg_size *total_freed_after_time = dead_after + expr->len;
    lg_size *offsets = total_freed_after_time + expr->len;

    for (lg_size i = 0; i < expr->len; i++) {
        sizes[i] = 0;
        dead_after[i] = 0;
        total_freed_after_time[i] = 0;
        offsets[i] = 0;
    }

    for (lg_size i = 0; i < expr->len; i++) {
        if (expr->nodes[i].x0.data == NULL) {
            dead_after[expr->nodes[i].x0.born_at] = i;
        }
        if (expr->nodes[i].x1.data == NULL) {
            dead_after[expr->nodes[i].x1.born_at] = i;
        }
        sizes[expr->nodes[i].y.born_at] = lg_desc_size_bytes(expr->nodes[i].y.desc);
    }
    for (lg_size i = 0; i < expr->len; i++) {
        total_freed_after_time[dead_after[i]] += sizes[i];
    }

    lg_size current_offset = 0;
    lg_size max_offset = 0;
    for (lg_size time = 0; time < expr->len; time++) {
        offsets[time] = current_offset;
        current_offset += sizes[expr->nodes[time].y.born_at];
        if (current_offset > max_offset) {
            max_offset = current_offset;
        }
        current_offset -= total_freed_after_time[time];
    }

    uint8_t *_data = perm->alloc(perm->ctx, max_offset);
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

    for (lg_size i = 0; i < expr->len; i++) {
        expr->nodes[i].y.data = (lg_scalar*)(_data + offsets[expr->nodes[i].y.born_at]);
        if (expr->nodes[i].x0.data == NULL) {
            expr->nodes[i].x0.data = (lg_scalar*)(_data + offsets[expr->nodes[i].x0.born_at]);
        }
        if (expr->nodes[i].x1.data == NULL) {
            expr->nodes[i].x1.data = (lg_scalar*)(_data + offsets[expr->nodes[i].x0.born_at]);
        }
    }

out:
    scratch->free(scratch->ctx, scratch_buf);
    return status;
}
