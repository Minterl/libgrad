#include <libgrad/internal/vm.h>

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

lg_status lg_add(
    lg_expr *expr,
    lg_tensor *y,
    const lg_tensor x0,
    const lg_tensor x1
) {
    const lg_size expr_idx = expr->len;
    lg_status status;

    y->born_at = expr_idx;
    for (lg_size i = 0; i < x0.desc.rank; i++) {
        y->desc.dim[i] = x0.desc.dim[i];
    }
    y->desc.rank = x0.desc.rank;
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

lg_status lg_nop(lg_expr *expr, lg_tensor x) {
    return lg_expr_append(expr, LG_OPCODE_NOP, (lg_tensor){ .born_at = expr->len }, x, (lg_tensor){0});
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
