#include <libgrad/internal/core.h>
#include <libgrad/internal/debug.h>

size_t LG_DescSizeInBytes(struct lg_desc desc) {
    if (desc.rank == 0) {
        return 0;
    }

    size_t max_offset = 0;
    for (size_t i = 0; i < desc.rank; i++) {
        if (desc.dim[i] > 0) {
            max_offset += (desc.dim[i] - 1) * desc.strides[i];
        }
    }

    return (max_offset + 1) * sizeof(lg_scalar);
}

void LG_CopyVectorToAxis(struct lg_desc desc, lg_scalar *restrict dest, const lg_scalar *vector, size_t copy_to_dim) {
    size_t dim_offset = 0;
    for (size_t i = 0; i < copy_to_dim; i++) {
        dim_offset += desc.dim[i];
    }
    const size_t n_values = desc.dim[copy_to_dim];
    for (size_t i = 0; i < n_values; i++) {
        dest[dim_offset + i] = vector[i];
    }
}

enum lg_status LG_DescComputeStrides(struct lg_desc *desc, enum lg_layout layout, size_t unit_align) {
#ifdef LG_SAFE
    if (desc->rank > LG_MAX_RANK) {
        return LG_STATUS_INVALID_RANK;
    }
#endif // LG_SAFE
    size_t last_stride = 1;
    for (size_t i = 1; i <= desc->rank; i++) {
        size_t axis = layout == LG_LAYOUT_ROW_MAJOR ? desc->rank - i : i - 1;
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

bool LG_DescIsIsotropic(struct lg_desc desc) {
    switch (desc.rank) {
    // All vectors are anisotropic.
    case 2:
        return 0;
    // All scalars and tensors of rank zero are isotropic.
    case 1:
    case 0:
        return 1;
    default: {
        size_t last_dim = desc.dim[0];
        for (size_t i = 0; i < desc.rank; i++) {
            if (last_dim != desc.dim[i]) {
                return 0;
            }
        }
        return 1;
    }
    }
}

void LG__DescLeftPadAxes(struct lg_desc **descs, size_t n_descs) {
    size_t max_rank = 0;
    for (size_t i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (size_t i = 0; i < n_descs; i++) {
        if (descs[i]->rank < max_rank) {
            size_t shift = max_rank - descs[i]->rank;
            for (size_t j = descs[i]->rank; j > 0; j--) {
                size_t src_idx = j - 1;
                descs[i]->dim[src_idx + shift] = descs[i]->dim[src_idx];
                descs[i]->strides[src_idx + shift] = descs[i]->strides[src_idx];
            }
            for (size_t j = 0; j < shift; j++) {
                descs[i]->dim[j] = 1;
                descs[i]->strides[j] = 0;
            }
            descs[i]->rank = max_rank;
        }
    }
}

enum lg_status LG_InferBroadcastedDims(
    size_t *out_rank,
    size_t *out_dim,
    const struct lg_desc **descs,
    size_t n_descs
) {
    size_t max_rank = 0;
    for (size_t i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

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
    
    size_t master_dim[LG_MAX_RANK];
    for (size_t i = 0; i < max_rank; i++) {
        master_dim[i] = 1;
    }

    for (size_t i_desc = 0; i_desc < n_descs; i_desc++) {
        for (size_t i_axis = 0; i_axis < descs[i_desc]->rank; i_axis++) {
            const size_t dim_desc = descs[i_desc]->dim[descs[i_desc]->rank - i_axis - 1];
            size_t *const dim_master = &master_dim[max_rank - i_axis - 1];
            if (dim_desc == 1) {
                continue;
            }
            if (*dim_master == 1) {
                *dim_master = dim_desc;
            } else if (*dim_master != dim_desc) {
                return LG_STATUS_SHAPE_MISMATCH;
            }
        }
    }

    if (out_rank != NULL) {
        *out_rank = max_rank;
    }
    if (out_dim != NULL) {
        for (size_t i = 0; i < LG_MAX_RANK; i++) {
            out_dim[i] = master_dim[i];
        }
    }

    return LG_STATUS_OK;
} 

enum lg_status LG_CreateBroadcastSpace(struct lg_desc **descs, size_t n_descs) {
    size_t max_rank = 0;
    size_t master_dim[LG_MAX_RANK] = {0};
    enum lg_status status = LG_InferBroadcastedDims(&max_rank, master_dim, (const struct lg_desc**)descs, n_descs);
    if (status != LG_STATUS_OK) {
        return status;
    }

    LG__DescLeftPadAxes(descs, n_descs);
    
    // Since we know all of the tensors are broadcast-compatible, and their
    // dims/strides are all in the same order, we can pre-bake broadcasting into the
    // strides in the views.
    //
    // This is done in two parts for every tensor along each axis:
    // 1) Setting all strides less than the master to zero, causing the actual
    //    offsets to never move.
    // 2) Aligning the logical dimensions of the tensor with the master.
    for (size_t i_desc = 0; i_desc < n_descs; i_desc++) {
        for (size_t i_axis = 1; i_axis <= max_rank; i_axis++) {
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

void LG_InferContractedDims(
    size_t *out_rank,
    size_t *out_dim,
    const struct lg_desc *x0,
    const struct lg_desc *x1,
    size_t n_contracted_axes,
    size_t n_batch_axes
) {
    // repeated below
    const size_t x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const size_t x1_first_free_axis = n_contracted_axes + n_batch_axes;

    size_t rank = 0;

    if (out_dim != NULL) {
        for (size_t i = n_batch_axes; i < x0_first_contracted_axis; i++, rank++) {
            out_dim[rank] = x0->dim[i];
        }
        for (size_t i = x1->rank - 1; i >= x1_first_free_axis; i--, rank++) {
            out_dim[rank] = x1->dim[i];
        }
    }

    if (out_rank != NULL) {
        *out_rank = rank;
    }
}

enum lg_status LG_CreateContractionSpace(
    struct lg_desc *y,
    struct lg_desc *x0,
    struct lg_desc *x1,
    size_t n_batch_axes
) {
    // The logical tensor axes will be laid out as follows:
    // { [batch], [x0_free], [x1_free], [contracted] }
    //    reg      reg       reg       0          | y strides
    //    reg      reg       0         reg        | x0 strides
    //    reg      0         reg       reg        | x1 strides

    struct lg_desc y_cpy = *y;
    struct lg_desc x0_cpy = *x0;
    struct lg_desc x1_cpy = *x1;
    
    // x0.rank = n_batch + n_contracted + x0_free
    // x1.rank = n_batch + n_contracted + x1_free
    // y.rank = n_batch + x0_free + x1_free
    // ergo ...
    const size_t n_contracted_axes = (x0->rank + x1->rank - y->rank - n_batch_axes) / 2;
    const size_t x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const size_t x1_first_free_axis = n_contracted_axes + n_batch_axes;

    // Batch axes are already in place
    size_t r = n_batch_axes;

    // Free axes
    for (size_t i = n_batch_axes; i < x0_first_contracted_axis; i++, r++) {
        x0->dim[r] = x0_cpy.dim[i];
        x0->strides[r] = x0_cpy.strides[i];
        x1->dim[r] = x0_cpy.dim[i];
        x1->strides[r] = 0;
        if (y->dim[r] != x0_cpy.dim[i]) {
            return LG_STATUS_SHAPE_MISMATCH;
        }
        y->strides[r] = y_cpy.strides[r];
    }
    for (size_t i = x1_first_free_axis; i < x1_cpy.rank; i++, r++) {
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
            size_t x0_ax = x0_first_contracted_axis, x1_ax = x1_first_free_axis - 1;
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

enum lg_status LG_SortAxes(struct lg_desc **descs, size_t n_descs) {
    size_t max_rank = 0;
    for (size_t i = 0; i < n_descs; i++) {
        if (descs[i]->rank > max_rank) {
            max_rank = descs[i]->rank;
        }
    }

    for (size_t i = 0; i < max_rank; i++) {
        bool swapped = 0;
        for (size_t j = 1; j < max_rank - i; j++) {
            const size_t prev_dim = descs[0]->strides[j - 1];
            const size_t cur_dim = descs[0]->strides[j];
            if (prev_dim < cur_dim) {
                // Swap the dimensions and strides for all of the tensors
                for (size_t k = 0; k < n_descs; k++) {
                    size_t temp = descs[k]->dim[j - 1];
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

enum lg_status LG_CoalesceAxes(struct lg_desc **descs, size_t n_descs) {
    size_t max_rank = 0;
    for (size_t i = 0; i < n_descs; i++) {
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
    for (size_t i = 0; i + 1 < max_rank; i++) {
        while (i + 1 < max_rank) {
            bool can_coalesce = 1;
            for (size_t j = 0; j < n_descs; j++) {
                const size_t d0 = descs[j]->dim[i];
                const size_t d1 = descs[j]->dim[i + 1];
                const size_t s0 = descs[j]->strides[i];
                const size_t s1 = descs[j]->strides[i + 1];

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

            const size_t new_dim = descs[0]->dim[i] * descs[0]->dim[i + 1];
            max_rank--;
            for (size_t j = 0; j < n_descs; j++) {
                descs[j]->rank = max_rank;
                descs[j]->dim[i] = new_dim;
                if (descs[j]->strides[i + 1] != 0) {
                    descs[j]->strides[i] = descs[j]->strides[i + 1];
                }
                for (size_t k = i + 1; k < max_rank; k++) {
                    descs[j]->dim[k] = descs[j]->dim[k + 1];
                    descs[j]->strides[k] = descs[j]->strides[k + 1];
                }
            }
        }
    }

    return LG_STATUS_OK;
}

bool LG_NDiterIncrement(struct lg_nditer *iter, size_t axis) {
    const size_t rank = iter->descs[0].rank;
    const size_t first_tracked_dim = rank - iter->n_tracked_dims;
    const size_t *restrict dim = iter->descs[0].dim;

    if (rank == 0) {
        return false;
    }

    axis += 1;
    while (axis > first_tracked_dim) {
        axis--;
        iter->coords[axis]++;
        if (iter->coords[axis] < dim[axis]) {
            for (size_t i = 0; i < LG_N_TRACKED_TENSORS; i++) {
                iter->indices[i] += iter->descs[i].strides[axis];
            }
            return true; 
        }
        iter->coords[axis] = 0;
        for (size_t i = 0; i < LG_N_TRACKED_TENSORS; i++) {
            iter->indices[i] -= iter->descs[i].strides[axis] * (dim[axis] - 1);
        }
    }

    return false;
}

void LG_NDiterGoto(struct lg_nditer *iter, size_t *coords) {
    for(size_t i = 0; i < iter->n_tracked_dims; i++) {
        iter->coords[i] = coords[i];
    }

    for (size_t i = 0; i < LG_N_TRACKED_TENSORS; i++) {
        iter->indices[i] = 0;
        for (size_t j = 0; j < iter->n_tracked_dims; j++) {
            iter->indices[i] += iter->descs[i].strides[j] * coords[j];
        }
    }
}
