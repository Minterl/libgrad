#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/internal/vm.h>
#include <libgrad/internal/core.h>

lg_status lg_cpu_exec(lg_expr expr);

/// Tensors must be sorted & broadcasted
void lg_cpu_add(
    const lg_desc y_desc, lg_scalar *restrict y,
    const lg_desc x0_desc, const lg_scalar *restrict x0,
    const lg_desc x1_desc, const lg_scalar *restrict x1
);
void lg_cpu_contract(
    const lg_desc y_desc, lg_scalar *restrict y,
    const lg_desc x0_desc, const lg_scalar *restrict x0,
    const lg_desc x1_desc, const lg_scalar *restrict x1
);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

lg_status lg_cpu_exec(lg_expr expr) {
    for (lg_size i = 0; i < expr.len; i++) {
        switch (expr.nodes[i].opcode) {
        case LG_OPCODE_NOP:
            break;
        case LG_OPCODE_ADD:
            lg_cpu_add(
                expr.nodes[i].y.desc, expr.nodes[i].y.data,
                expr.nodes[i].x0.desc, expr.nodes[i].x0.data,
                expr.nodes[i].x1.desc, expr.nodes[i].x1.data
            );
            break;
        case LG_OPCODE_CONTRACT:
            lg_cpu_contract(
                expr.nodes[i].y.desc, expr.nodes[i].y.data,
                expr.nodes[i].x0.desc, expr.nodes[i].x0.data,
                expr.nodes[i].x1.desc, expr.nodes[i].x1.data
            );
            break;
        case LG_OPCODE_SUB:
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

void lg_cpu_add(
    const lg_desc y_desc, lg_scalar *restrict y,
    const lg_desc x0_desc, const lg_scalar *restrict x0,
    const lg_desc x1_desc, const lg_scalar *restrict x1
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
}

void lg_cpu_contract(
    const lg_desc y_desc, lg_scalar *restrict y,
    const lg_desc x0_desc, const lg_scalar *restrict x0,
    const lg_desc x1_desc, const lg_scalar *restrict x1
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
}

#endif // LG_CPU_IMPLEMENTATION
