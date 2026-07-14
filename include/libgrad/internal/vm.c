#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h>

lg_status lg_get_last_location(lg_expr *expr, lg_tensor *x) {
    bool found = false;
    for (lg_size i = 0; i < expr->len; i++) {
        if (expr->y[i].born_at == x->born_at) {
            // We only change the data pointer because the expr may have
            // some other `desc` that doesn't match that of the caller.
            // We do not want to mutate the caller's interpretation of the buffer.
            x->data = expr->y[i].data; 
            found = true;
        }
    }
    if (!found) {
        return LG_STATUS_NOT_FOUND;
    }
    return LG_STATUS_OK;
}

lg_status lg_expr_append(
    lg_expr *expr,
    const lg_opcode opcode,
    const lg_tensor y,
    const lg_tensor x0,
    const lg_tensor x1
) {
#ifdef LG_SAFE
    if (expr->len >= expr->cap) {
        return LG_STATUS_EXPR_OVERFLOW;
    }
#endif // LG_SAFE

    lg_size next_idx = expr->len;
    expr->len += 1;
    expr->opcodes[next_idx] = opcode;
    expr->y[next_idx] = y;
    expr->x0[next_idx] = x0;
    expr->x1[next_idx] = x1;

    return LG_STATUS_OK;
}

/// Shallow copies from src to dest withotu copying the data pointer.
/// This exists because simple assignment doesn't behave correctly all of the time.
static inline void __lg_tensor_clone_metadata(lg_tensor *restrict dest, const lg_tensor *restrict src) {
    dest->born_at = src->born_at;
    for (lg_size i = 0; i < src->desc.rank; i++) {
        dest->desc.dim[i] = src->desc.dim[i];
        dest->desc.strides[i] = src->desc.strides[i];
    }
    dest->desc.rank = src->desc.rank;
}

lg_status lg_nop(lg_expr *expr, lg_tensor x) {
    return lg_expr_append(expr, LG_OPCODE_NOP, (lg_tensor){ .born_at = expr->len }, x, (lg_tensor){0});
}

lg_status lg_add(
    lg_expr *expr,
    lg_tensor *y,
    const lg_tensor x0,
    const lg_tensor x1
) {
    const lg_size expr_idx = expr->len;
    lg_status status;

    __lg_tensor_clone_metadata(y, &x0);
    y->born_at = expr_idx;
    // TODO: naive layout
    status = lg_desc_layout(&y->desc, LG_LAYOUT_ROW_MAJOR, 1 /* TODO */);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = lg_expr_append(expr, LG_OPCODE_ADD, *y, x0, x1);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = lg_desc_broadcast((lg_desc*[]){
        &y->desc,
        &expr->x0[expr_idx].desc,
        &expr->x1[expr_idx].desc,
    }, 3);
    if (status != LG_STATUS_OK) {
        return status;
    }

    return LG_STATUS_OK;
}

lg_status lg_contract(
    lg_expr *expr,
    lg_tensor *y,
    lg_tensor x0,
    lg_tensor x1,
    const lg_size n_batch_axes
) {
    const lg_size expr_idx = expr->len;
    lg_status status;

    __lg_tensor_clone_metadata(y, &x0);
    y->born_at = expr_idx;
    status = lg_desc_layout(&y->desc, LG_LAYOUT_ROW_MAJOR, 1 /* TODO */);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = lg_expr_append(expr, LG_OPCODE_CONTRACT, *y, x0, x1);
    if (status != LG_STATUS_OK) {
        return status;
    }

    status = lg_desc_compute_contracted_dims(
        &y->desc,
        &x0.desc,
        &x0.desc,
        n_batch_axes
    );
    if (status != LG_STATUS_OK) {
        return status;
    }

    return LG_STATUS_OK;
}

lg_status __lg_expr_pass_sort_axes(lg_expr *expr) {
    lg_status status;
    for (lg_size i = 0; i < expr->len; i++) {
        status = lg_desc_sort_axes((lg_desc*[]){
            &expr->y[i].desc,
            &expr->x0[i].desc,
            &expr->x1[i].desc,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

lg_status __lg_expr_pass_coalesce_axes(lg_expr *expr) {
    lg_status status;
    for (lg_size i = 0; i < expr->len; i++) {
        status = lg_desc_coalesce_axes((lg_desc*[]){
            &expr->y[i].desc,
            &expr->x0[i].desc,
            &expr->x1[i].desc,
        }, 3);
        if (status != LG_STATUS_OK) {
            return status;
        }
    }
    return LG_STATUS_OK;
}

lg_status lg_expr_compile(lg_expr *expr) {
    lg_status status;
    status = __lg_expr_pass_sort_axes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    status = __lg_expr_pass_coalesce_axes(expr);
    if (status != LG_STATUS_OK) {
        return status;
    }
    return LG_STATUS_OK;
}
