#include <libgrad/internal/core.h>
#include <libgrad/internal/debug.h>

lg_size lg_desc_size_bytes(lg_desc desc) {
    if (desc.rank == 0) {
        return 0;
    }

    lg_size max_offset = 0;
    for (lg_size i = 0; i < desc.rank; i++) {
        if (desc.dim[i] > 0) {
            max_offset += (desc.dim[i] - 1) * desc.strides[i];
        }
    }

    return (max_offset + 1) * sizeof(lg_scalar);
}

void lg_copy_vector(lg_desc desc, lg_scalar *restrict dest, const lg_scalar *vector, lg_size copy_to_dim) {
    lg_size dim_offset = 0;
    for (lg_size i = 0; i < copy_to_dim; i++) {
        dim_offset += desc.dim[i];
    }
    const lg_size n_values = desc.dim[copy_to_dim];
    for (lg_size i = 0; i < n_values; i++) {
        dest[dim_offset + i] = vector[i];
    }
}

lg_status lg_desc_layout(lg_desc *desc, lg_layout layout, lg_size unit_align) {
#ifdef LG_SAFE
    if (desc->rank > LG_MAX_RANK) {
        return LG_STATUS_INVALID_RANK;
    }
#endif // LG_SAFE
    lg_size last_stride = 1;
    for (lg_size i = 1; i <= desc->rank; i++) {
        lg_size axis = layout == LG_LAYOUT_ROW_MAJOR ? desc->rank - i : i - 1;
        desc->strides[axis] = last_stride;
        last_stride *= desc->dim[desc->rank - i];
        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (unit_align > 1 && i == 1) {
            last_stride = (last_stride + unit_align - 1) & ~(unit_align - 1);
        }
    }

    return LG_STATUS_OK;
}

bool lg_desc_is_isotropic(lg_desc desc) {
    switch (desc.rank) {
    // All vectors are anisotropic.
    case 2:
        return 0;
    // All scalars and tensors of rank zero are isotropic.
    case 1:
    case 0:
        return 1;
    default: {
        lg_size last_dim = desc.dim[0];
        for (lg_size i = 0; i < desc.rank; i++) {
            if (last_dim != desc.dim[i]) {
                return 0;
            }
        }
        return 1;
    }
    }
}

void lg_desc_tranpose(lg_desc *desc) {
    for (lg_size i = 0; i < desc->rank / 2; i++) {
        const lg_size opp = desc->rank - 1 - i;
        lg_size temp = desc->dim[i];
        desc->dim[i] = desc->dim[opp];
        desc->dim[opp] = temp;
    }
}

/// Returns the maximum rank of all of the descriptors
static inline lg_size __lg_desc_left_pad_dims(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank < max_rank) {
            lg_size shift = max_rank - descs[i]->rank;
            for (lg_size j = descs[i]->rank; j > 0; j--) {
                lg_size src_idx = j - 1;
                descs[i]->dim[src_idx + shift] = descs[i]->dim[src_idx];
                descs[i]->strides[src_idx + shift] = descs[i]->strides[src_idx];
            }
            for (lg_size j = 0; j < shift; j++) {
                descs[i]->dim[j] = 1;
                descs[i]->strides[j] = 0;
            }
            descs[i]->rank = max_rank;
        }
    }
    
    return max_rank;
}
    
lg_status lg_desc_broadcast(lg_desc **descs, lg_size n_descs) {
    const lg_size max_rank = __lg_desc_left_pad_dims(descs, n_descs);

    // --- Validate that all tensors are broadcast-compatible ---
    // Every tensor participating in the tracking must be broadcast-compatible
    // with every other tensor.
    // Naively, we would perform an O(n^2) check across a matrix of all of the participating tensors.
    // The shortcut below is equivalent.
    //
    // Tensors are broadcast-compatible if all of their dimensions are broadcast-compatible.
    // Two dimensions are broadcast-compatible if one of three things is true:
    // 1) The dimensions are the same.
    // 2) One of the dimensions is 1.
    // 3) One of the dimensions does not exist.
    lg_size master_dim[LG_MAX_RANK];
    for (lg_size i = 0; i < max_rank; i++) {
        master_dim[i] = 1;
    }

    for (lg_size i_desc = 0; i_desc < n_descs; i_desc++) {
        for (lg_size i_axis = 0; i_axis < max_rank; i_axis++) {
            lg_size dim_current = (i_axis < descs[i_desc]->rank) ? descs[i_desc]->dim[i_axis] : 1;
            if (dim_current == 1) {
                continue;
            }
            if (master_dim[i_axis] == 1) {
                master_dim[i_axis] = dim_current;
            } else if (master_dim[i_axis] != dim_current) {
                return LG_STATUS_SHAPE_MISMATCH;
            }
        }
    }
    
    // Since we know all of the tensors are broadcast-compatible, and their
    // dims/strides are all in the same order, we can pre-bake broadcasting into the
    // strides in the views.
    //
    // It's important to remember that both the max_rank array and the dims in the views
    // are right-aligned at this point.
    //
    // This is done in two parts for every tensor along each axis:
    // 1) Setting all strides less than the master to zero, causing the actual
    //    offsets to never move.
    // 2) Aligning the logical dimensions of the tensor with the master.
    for (lg_size i_desc = 0; i_desc < n_descs; i_desc++) {
        for (lg_size i_axis = 1; i_axis <= max_rank; i_axis++) {
            if (descs[i_desc]->dim[max_rank - i_axis] < master_dim[max_rank - i_axis]) {
                descs[i_desc]->strides[max_rank - i_axis] = 0;
                descs[i_desc]->dim[max_rank - i_axis] = master_dim[max_rank - i_axis];
            }
        }
    }

    // The resulting state of our tensor views looks like this:
    // - All tensors have the same logical rank
    // - All tensors have the exact same logical dims
    // - The only thing that changes between tensor views is
    //   striding.

    return LG_STATUS_OK;
}

lg_status lg_desc_compute_contracted_dims(lg_desc *y, lg_desc *x0, lg_desc *x1, lg_size n_batch_axes) {
    // The logical tensor axes will be laid out as follows:
    // { [batch], [x0_free], [x1_free], [contracted] }
    //    reg      reg       reg       0          | y strides
    //    reg      reg       0         reg        | x0 strides
    //    reg      0         reg       reg        | x1 strides

    lg_desc y_cpy = *y;
    lg_desc x0_cpy = *x0;
    lg_desc x1_cpy = *x1;
    
    // x0.rank = n_batch + n_contracted + x0_free
    // x1.rank = n_batch + n_contracted + x1_free
    // y.rank = n_batch + x0_free + x1_free
    // ergo ...
    const lg_size n_contracted_axes = (x0->rank + x1->rank - y->rank - n_batch_axes) / 2;
    const lg_size x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const lg_size x1_first_free_axis = n_contracted_axes + n_batch_axes;

    // Batch axes are already in place
    lg_size r = n_batch_axes;

    // Free axes
    for (lg_size i = n_batch_axes; i < x0_first_contracted_axis; i++, r++) {
        x0->dim[r] = x0_cpy.dim[i];
        x0->strides[r] = x0_cpy.strides[i];
        x1->dim[r] = x0_cpy.dim[i];
        x1->strides[r] = 0;
        if (y->dim[r] != x0_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        y->strides[r] = y_cpy.strides[r];
    }
    for (lg_size i = x1_first_free_axis; i < x1_cpy.rank; i++, r++) {
        x0->dim[r] = x1_cpy.dim[i];
        x0->strides[r] = 0;
        x1->dim[r] = x1_cpy.dim[i];
        x1->strides[r] = x1_cpy.strides[i];
        if (y->dim[r] != x1_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        y->strides[r] = y_cpy.strides[r];
    }

    // Contracted axes
    if (n_contracted_axes > 0) {
        for (
            lg_size x0_ax = x0_first_contracted_axis, x1_ax = x1_first_free_axis - 1;
            x0_ax < x0_cpy.rank; // x1_ax > 0
            x0_ax++, x1_ax--, r++
        ) {
            x0->dim[r] = x0_cpy.dim[x0_ax];
            x0->strides[r] = x0_cpy.strides[x0_ax];
            x1->dim[r] = x0_cpy.dim[x0_ax];
            x1->strides[r] = x1_cpy.strides[x1_ax];
            y->dim[r] = x0_cpy.dim[x0_ax];
            y->strides[r] = 0;
        }
    }

    y->rank = r;
    x0->rank = r;
    x1->rank = r;

    return LG_STATUS_OK;
}

lg_status lg_desc_sort_axes(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (lg_size i = 0; i < max_rank; i++) {
        bool swapped = 0;
        for (lg_size j = 1; j < max_rank - i; j++) {
            const lg_size prev_dim = descs[0]->strides[j - 1];
            const lg_size cur_dim = descs[0]->strides[j];
            if (prev_dim < cur_dim) {
                // Swap the dimensions and strides for all of the tensors
                for (lg_size k = 0; k < n_descs; k++) {
                    lg_size temp = descs[k]->dim[j - 1];
                    descs[k]->dim[j - 1] = descs[k]->dim[j];
                    descs[k]->dim[j] = temp;
                    temp = descs[k]->strides[j - 1];
                    descs[k]->strides[j - 1] = descs[k]->strides[j];
                    descs[k]->strides[j] = temp;
                }
                swapped = 1;
            }
        }
        if (!swapped) {
            break;
        }
    }

    return LG_STATUS_OK;
}

lg_status lg_desc_coalesce_axes(lg_desc **descs, lg_size n_descs) {
    lg_size max_rank = 0;
    for (lg_size i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    // --- Coalesce dimensions ---
    // Dimensions can be coalesced under three conditions:
    // 1) One of the dimension is broadcasted
    // 2) One of the dimensions has the unit stride
    //    (is already contiguous in memory)
    // 3) The latter dimensions is the contiguous extension of the 
    //    previous dimenison.
    //
    // Use i + 1 < max_rank to handle underflows.
    for (lg_size i = 0; i + 1 < max_rank; i++) {
        while (i + 1 < max_rank) {
            bool can_coalesce = 1;
            for (lg_size j = 0; j < n_descs; j++) {
                const lg_size d0 = descs[j]->dim[i];
                const lg_size d1 = descs[j]->dim[i + 1];
                const lg_size s0 = descs[j]->strides[i];
                const lg_size s1 = descs[j]->strides[i + 1];

                const bool is_broadcasted = s0 == 0 || s1 == 0;
                const bool has_unit = d0 == 1 || d1 == 1;
                const bool is_contiguous_extension = s0 == s1 * d1;

                if (!is_broadcasted && !has_unit && !is_contiguous_extension) {
                    can_coalesce = 0;
                    break;
                }
            }

            if (!can_coalesce) {
                break;
            }

            const lg_size new_dim = descs[0]->dim[i] * descs[0]->dim[i + 1];
            max_rank--;
            for (lg_size j = 0; j < n_descs; j++) {
                descs[j]->rank = max_rank;
                descs[j]->dim[i] = new_dim;
                if (descs[j]->strides[i + 1] != 0) {
                    descs[j]->strides[i] = descs[j]->strides[i + 1];
                }
                for (lg_size k = i + 1; k < max_rank; k++) {
                    descs[j]->dim[k] = descs[j]->dim[k + 1];
                    descs[j]->strides[k] = descs[j]->strides[k + 1];
                }
            }
        }
    }

    return LG_STATUS_OK;
}

bool lg_nditer_increment(lg_nditer *iter, lg_size axis) {
    const lg_size rank = iter->descs[0].rank;
    const lg_size first_tracked_dim = rank - iter->n_tracked_dims;
    const lg_size *restrict dim = iter->descs[0].dim;

    if (rank == 0) {
        return false;
    }

    axis += 1;
    while (axis > first_tracked_dim) {
        axis--;
        iter->coords[axis]++;
        if (iter->coords[axis] < dim[axis]) {
            for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
                iter->indices[i] += iter->descs[i].strides[axis];
            }
            return true; 
        }
        iter->coords[axis] = 0;
        for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
            iter->indices[i] -= iter->descs[i].strides[axis] * (dim[axis] - 1);
        }
    }

    return false;
}

void lg_nditer_goto(lg_nditer *iter, lg_size *coords) {
    for(lg_size i = 0; i < iter->n_tracked_dims; i++) {
        iter->coords[i] = coords[i];
    }

    for (lg_size i = 0; i < LG_N_TRACKED_TENSORS; i++) {
        iter->indices[i] = 0;
        for (lg_size j = 0; j < iter->n_tracked_dims; j++) {
            iter->indices[i] += iter->descs[i].strides[j] * coords[j];
        }
    }
}
