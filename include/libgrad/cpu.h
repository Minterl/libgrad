#include <libgrad/core.h>

#ifndef LG_CPU_H_
#define LG_CPU_H_

/// Tensors must be sorted & broadcasted
lg_status lg_cpu_add(lg_tensor out, const lg_tensor a, const lg_tensor b);

static inline bool __lg_increment_coords_rtl(lg_size *coord, const lg_size *dim, lg_size rank) {
    if (rank == 0) return false;

    lg_size axis = rank;
    
    while (axis > 0) {
        axis--;
        coord[axis]++;
        if (coord[axis] < dim[axis]) {
            return true; 
        }
        coord[axis] = 0;
    }

    return false;
}

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION


lg_status lg_cpu_add(
    lg_tensor out,
    const lg_tensor a,
    const lg_tensor b
) {
    lg_size coords[LG_MAX_RANK] = {0};

    do {
        lg_size out_idx = 0;
        lg_size a_idx = 0;
        lg_size b_idx = 0;

        for (lg_size i = 0; i < out.rank; i++) {
            out_idx += out.strides[i] * coords[i];
            a_idx += a.strides[i] * coords[i];
            b_idx += b.strides[i] * coords[i];
        }

        out.data[out_idx] = a.data[a_idx] + b.data[b_idx];
    } while (__lg_increment_coords_rtl(coords, out.dim, out.rank));

    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
