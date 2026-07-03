#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/core.h>

lg_status lg_cpu_backward(lg_expr expr);

/// Tensors must be sorted & broadcasted
lg_status lg_cpu_add(
    const lg_desc y_desc, lg_dtype *restrict y,
    const lg_desc x0_desc, const lg_dtype *restrict x0,
    const lg_desc x1_desc, const lg_dtype *restrict x1
);
lg_status lg_cpu_add_back(
    const lg_desc dy_desc, const lg_dtype *restrict dy,
    const lg_desc x0_desc, lg_dtype *restrict dx0,
    const lg_desc x1_desc, lg_dtype *restrict dx1
);
lg_status lg_cpu_contract(
    const lg_desc y_desc, lg_dtype *restrict y,
    const lg_desc x0_desc, const lg_dtype *restrict x0,
    const lg_desc x1_desc, const lg_dtype *restrict x1
);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

lg_status lg_cpu_forward(lg_expr expr) {
    for (lg_size i = 0; i < expr.len; i++) {
        switch (expr.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add(
                expr.y[i].desc, expr.y[i].data,
                expr.x0[i].desc, expr.x0[i].data,
                expr.x1[i].desc, expr.x1[i].data
            );
            break;
        case LG_OPCODE_SUB:
        case LG_OPCODE_CONTRACT:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        default:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    }

    return LG_STATUS_OK;
}

lg_status lg_cpu_backward(lg_expr expr) {
    if (expr.len == 0) {
        return LG_STATUS_OK;
    }
    
    lg_size i = expr.len - 1;
    do {
        switch (expr.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add_back(
                expr.y[i].desc, expr.y[i].grad,
                expr.x0[i].desc, expr.x0[i].grad,
                expr.x1[i].desc, expr.x1[i].grad
            );
            break;
        case LG_OPCODE_SUB:
        case LG_OPCODE_CONTRACT:
        case LG_OPCODE_HADAMARD:
        case LG_OPCODE_LOSS_MSE:
        case LG_OPCODE_LOSS_CROSS_ENTROPY:
        case LG_OPCODE_RELU:
        case LG_OPCODE_STABLE_SOFTMAX:
        case LG_OPCODE_SIGMOID:
        case LG_OPCODE_LN:
        default:
            return LG_STATUS_UNSUPPORTED_OPCODE;
        }
    } while (i-- > 0);

    return LG_STATUS_OK;
}

lg_status lg_cpu_add(
    const lg_desc y_desc, lg_dtype *restrict y,
    const lg_desc x0_desc, const lg_dtype *restrict x0,
    const lg_desc x1_desc, const lg_dtype *restrict x1
) {
    lg_nditer iter = {
        .descs = {y_desc, x0_desc, x1_desc},
        .n_tracked_dims = y_desc.rank,
    };

    do {
        const lg_size y_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        y[y_idx] = x0[x0_idx] + x1[x1_idx];
   } while (lg_nditer_increment(&iter, y_desc.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_add_back(
    const lg_desc dy_desc, const lg_dtype *restrict dy,
    const lg_desc dx0_desc, lg_dtype *restrict dx0,
    const lg_desc dx1_desc, lg_dtype *restrict dx1
) {
    lg_nditer iter = {
        .descs = {dy_desc, dx0_desc, dx1_desc},
        .n_tracked_dims = dy_desc.rank,
    };

    do {
        const lg_size dy_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        dx0[x0_idx] += dy[dy_idx];
        dx1[x1_idx] += dy[dy_idx];
    } while (lg_nditer_increment(&iter, dy_desc.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_contract(
    const lg_desc y_desc, lg_dtype *restrict y,
    const lg_desc x0_desc, const lg_dtype *restrict x0,
    const lg_desc x1_desc, const lg_dtype *restrict x1
) {
    lg_nditer iter = {
        .descs = {y_desc, x0_desc, x1_desc},
        .n_tracked_dims = y_desc.rank,
    };

    do {
        const lg_size y_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        y[y_idx] += x0[x0_idx] * x1[x1_idx];
    } while (lg_nditer_increment(&iter, y_desc.rank - 1));

    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
