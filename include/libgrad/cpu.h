#ifndef LG_CPU_H_
#define LG_CPU_H_

#include <libgrad/core.h>

lg_status lg_cpu_backward(lg_tape tape);

/// Tensors must be sorted & broadcasted
lg_status lg_cpu_add(lg_tensor y, const lg_tensor x0, const lg_tensor x1);
lg_status lg_cpu_add_back(const lg_tensor dy, lg_tensor dx0, lg_tensor dx1);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

lg_status lg_cpu_forward(lg_tape tape) {
    for (lg_size i = 0; i < tape.len; i++) {
        switch (tape.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add(
                tape.y[i],
                tape.x0[i],
                tape.x1[i]
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

lg_status lg_cpu_backward(lg_tape tape) {
    if (tape.len == 0) {
        return LG_STATUS_OK;
    }
    
    lg_size i = tape.len - 1;
    do {
        switch (tape.opcodes[i]) {
        case LG_OPCODE_ADD:
            lg_cpu_add_back(tape.y[i], tape.x0[i], tape.x1[i]);
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

lg_status lg_cpu_add(lg_tensor y, const lg_tensor x0, const lg_tensor x1) {
    lg_nditer iter = {
        .tensors = {y, x0, x1},
        .n_tracked_dims = y.desc.rank,
    };

    do {
        const lg_size y_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        y.data[y_idx] = x0.data[x0_idx] + x1.data[x1_idx];
   } while (lg_nditer_increment(&iter, y.desc.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_add_back(const lg_tensor dy, lg_tensor dx0, lg_tensor dx1) {
    lg_nditer iter = {
        .tensors = {dy, dx0, dx1},
        .n_tracked_dims = dy.desc.rank,
    };

    do {
        const lg_size dy_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        dx0.grad[x0_idx] += dy.grad[dy_idx];
        dx1.grad[x1_idx] += dy.grad[dy_idx];
    } while (lg_nditer_increment(&iter, dy.desc.rank - 1));

    return LG_STATUS_OK;
}

lg_status lg_cpu_contract(lg_tensor y, const lg_tensor x0, const lg_tensor x1) {
    lg_nditer iter = {
        .tensors = {y, x0, x1},
        .n_tracked_dims = y.desc.rank,
    };

    do {
        const lg_size y_idx = iter.indices[0];
        const lg_size x0_idx = iter.indices[1];
        const lg_size x1_idx = iter.indices[2];

        y.data[y_idx] += x0.data[x0_idx] * x1.data[x1_idx];
    } while (lg_nditer_increment(&iter, y.desc.rank - 1));

    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
