#include <libgrad/internal/linalg.h>
#include <libgrad/internal/alloc.h>
#include <libgrad/internal/debug.h>

LG_StatusKind
lg_atran_strided_projection_from_shape(
    LG_Arena *arena,
    const LG_LogicalShape *shape,
    LG_LayoutKind layout,
    uint32_t unit_align,
    LG_AffineTransform *out_atran
) {
    uint64_t *data = lg_arena_alloc_array(arena, uint64_t, shape->rank);
    if (data == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    LG_AffineTransform atran = {0};
    atran.n_rows = 1;
    atran.n_cols = shape->rank;

    int64_t *const A = lg_atran_get_A(&atran);

    uint8_t last_stride = 1;
    for (uint8_t i = 1; i <= shape->rank; i++) {
        uint8_t axis = layout == LG_LayoutKind_RowMajor ? shape->rank - i : i - 1;
        A[axis] = last_stride;

        last_stride *= shape->dim[shape->rank - i];

        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (unit_align > 1 && i == 1) {
            last_stride = (last_stride + unit_align - 1) & ~(unit_align - 1);
        }
    }

    lg_assert(lg_atran_is_valid_address_operator(&atran));
    lg_assert(out_atran != NULL);

    *out_atran = atran;

    return LG_StatusKind_OK;
}

void
lg_atran_apply(const LG_AffineTransform *tran, const int64_t x[static LG_MAX_RANK], int64_t y[static LG_MAX_RANK]) {
    const int64_t *const restrict A = lg_atran_get_A(tran);
    const int64_t *const restrict b = lg_atran_get_b(tran);

    lg_memzero(y, LG_MAX_RANK * sizeof(int64_t));

    for (uint8_t i_rows = 0; i_rows < tran->n_rows; i_rows++) {
        for (uint8_t i_cols = 0; i_cols < tran->n_cols; i_cols++) {
            y[i_rows] += A[tran->n_cols*i_rows + i_cols] * x[i_cols];
        }
    }

    for (uint8_t i_rows = 0; i_rows < tran->n_rows; i_rows++) {
        y[i_rows] += b[i_rows];
    }
}

LG_StatusKind
lg_poly_make_parallelotope(LG_Arena *arena, LG_Polyhedron *out_poly, uint8_t rank, int64_t *lower, int64_t *upper) {
    bool is_canonical_aabb = true;
    for (uint8_t i = 0; i < rank; i++) {
        if (lower[i] != 0 || upper[i] < 0) {
            is_canonical_aabb = false;
            break;
        }
    }

    if (is_canonical_aabb) {
        int64_t *extents = lg_arena_alloc_array(arena, int64_t, rank);
        if (extents == NULL) {
            return LG_StatusKind_OutOfMemory;
        }

        lg_memzero(out_poly, sizeof(LG_Polyhedron));
        out_poly->repr_kind = LG_PolyhedronReprKind_CanonicalAABB;
        out_poly->as.canonical_aabb.rank = rank;
        out_poly->as.canonical_aabb.extents = extents;

        lg_memcpy(extents, upper, rank * sizeof(int64_t));
    } else {
        const uint8_t n_cols = rank;
        const uint8_t n_rows = rank * 2;

        int64_t *data = lg_arena_alloc_array(arena, int64_t, n_rows * n_cols);
        if (data == NULL) {
            return LG_StatusKind_OutOfMemory;
        }

        lg_memzero(out_poly, sizeof(LG_Polyhedron));
        out_poly->repr_kind = LG_PolyhedronReprKind_Hyperplane;
        out_poly->as.hyperplane.n_cols = n_cols;
        out_poly->as.hyperplane.n_rows = n_rows;
        out_poly->as.hyperplane.data = data;

        int64_t *A = lg_hpoly_get_A(&out_poly->as.hyperplane);
        int64_t *b = lg_hpoly_get_b(&out_poly->as.hyperplane);

        for (uint8_t i = 0; i < rank; i++) {
            A[2*i * out_poly->as.hyperplane.n_cols + i] = -1;
            A[(2*i + 1) * out_poly->as.hyperplane.n_cols + i] = 1;
            b[2*i] = -lower[i];
            b[2*i + 1] = upper[i];
        }
    }

    return LG_StatusKind_OK;
}

LG_StatusKind 
lg_infer_broadcasted_dims(
    LG_LogicalShape *lg_nullable out,
    const LG_LogicalShape **shapes,
    size_t n_descs
) {
    size_t max_rank = 0;
    for (size_t i = 0; i < n_descs; i++) {
        if (shapes[i]->rank > max_rank) {
            max_rank = shapes[i]->rank;
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
        for (size_t i_axis = 0; i_axis < shapes[i_desc]->rank; i_axis++) {
            const size_t dim_desc = shapes[i_desc]->dim[shapes[i_desc]->rank - i_axis - 1];
            size_t *const dim_master = &master_dim[max_rank - i_axis - 1];
            if (dim_desc == 1) {
                continue;
            }
            if (*dim_master == 1) {
                *dim_master = dim_desc;
            } else if (*dim_master != dim_desc) {
                return LG_StatusKind_ShapeMismatch;
            }
        }
    }

    if (out != NULL) {
        out->rank = max_rank;
        for (size_t i = 0; i < LG_MAX_RANK; i++) {
            out->dim[i] = master_dim[i];
        }
    }

    return LG_StatusKind_OK;
} 

LG_StatusKind 
lg_infer_contracted_dims(
    LG_LogicalShape *lg_nullable out_y,
    const LG_LogicalShape *x0,
    const LG_LogicalShape *x1,
    size_t n_contracted_axes,
    size_t n_batch_axes
) {
    if (x0->rank < n_contracted_axes || n_contracted_axes + n_batch_axes > x1->rank) {
        return LG_StatusKind_InvalidArgument;
    }

    // repeated below
    const size_t x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const size_t x1_first_free_axis = n_contracted_axes + n_batch_axes;

    size_t rank = 0;

    size_t dim[LG_MAX_RANK] = {0};
    for (size_t i = n_batch_axes; i < x0_first_contracted_axis; i++, rank++) {
        dim[rank] = x0->dim[i];
    }
    for (size_t i = x1->rank; i > x1_first_free_axis; i--, rank++) {
        dim[rank] = x1->dim[i - 1];
    }

    if (out_y != NULL) {
        out_y->rank = rank;
        for (size_t i = 0; i < rank; i++) {
            out_y->dim[i] = dim[i];
        }
    }

    return LG_StatusKind_OK;
}

LG_StatusKind 
lg_create_contracted_iteration_space(
    LG_Arena *arena,
    const LG_LogicalShape *y,
    const LG_LogicalShape *x0,
    const LG_LogicalShape *x1,
    size_t n_batch_axes,
    LG_Polyhedron *out_poly,
    LG_AffineTransform *out_y_atran,
    LG_AffineTransform *out_x0_atran,
    LG_AffineTransform *out_x1_atran
) {
    LG_StatusKind status = LG_StatusKind_OK;

    if (
        n_batch_axes > y->rank ||
        n_batch_axes > x0->rank ||
        n_batch_axes > x1->rank
    ) {
        return LG_StatusKind_InvalidArgument;
    }

    ////////////////////////////////////////////////////////////////////
    /// ~~ Important terms & explanation ~~
    
    // The easiest way to picture how this works is to pretend that
    // these are just strided buffer descriptors.
    //
    // The strides over the iteration space will/would look like this:
    // { [batch], [x0_free], [x1_free], [contracted] }
    //    reg      reg        reg        0           | y strides
    //    reg      reg        0          reg         | x0 strides
    //    reg      0          reg        reg         | x1 strides
    //
    // So, the iteration space itself will be a canonical AABB,
    // and so the affine maps must be projections 
    // C : R^(iteration domain rank) -> R^(index domain rank) s.t some
    // coordinate in the iteration domain may map to the same coordinate
    // in the index domain where the corresponding index domain axis
    // "stride" is zero. That looks like a diagonal matrix, something like
    // this: 
    // { 1, 0, 0 } // a non-zero axis
    // { 0, 1, 0 } // a non-zero axis
    // { 0, 0, 0 } // a zero-strided axis 
    //
    // Combined with accumulation semantics, this can be used to reduce
    // along the zero-strided axes.
    //
    // In English: this is a highly generalized version of matrix 
    // multiplication.

    // x0.rank = n_batch + n_contracted + x0_free
    // x1.rank = n_batch + n_contracted + x1_free
    // y.rank = n_batch + x0_free + x1_free
    // ergo ...
    const size_t n_contracted_axes = (x0->rank + x1->rank - y->rank - n_batch_axes) / 2;
    const size_t iter_domain_rank = y->rank + n_contracted_axes;
    const size_t x0_first_contracted_axis = x0->rank - n_contracted_axes;
    const size_t x1_first_free_axis = n_contracted_axes + n_batch_axes;
    const size_t x0_n_free_axes = x0->rank - n_batch_axes - n_contracted_axes;
    const size_t x1_n_free_axes = x1->rank - n_batch_axes - n_contracted_axes;

    lg_assert(n_contracted_axes < LG_MAX_RANK);
    lg_assert(iter_domain_rank < LG_MAX_RANK);
    lg_assert(x0_first_contracted_axis < LG_MAX_RANK);
    lg_assert(x1_first_free_axis < LG_MAX_RANK);
    lg_assert(x0_n_free_axes < LG_MAX_RANK);
    lg_assert(x1_n_free_axes < LG_MAX_RANK);


    ////////////////////////////////////////////////////////////////////
    /// ~~ Calculate iteration domain extents & map to transforms ~~

    int64_t *iter_domain_extents = lg_arena_alloc_array(arena, int64_t, iter_domain_rank);
    if (iter_domain_extents == NULL) {
        return LG_StatusKind_OutOfMemory;
    }
    int64_t *y_A = lg_arena_alloc_array(arena, int64_t, iter_domain_rank * y->rank);
    if (y_A == NULL) {
        return LG_StatusKind_OutOfMemory;
    }
    int64_t *x0_A = lg_arena_alloc_array(arena, int64_t, iter_domain_rank * x0->rank);
    if (x0_A == NULL) {
        return LG_StatusKind_OutOfMemory;
    }
    int64_t *x1_A = lg_arena_alloc_array(arena, int64_t, iter_domain_rank * x1->rank);
    if (x1_A == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    {
        uint8_t r = 0;
        
        // Batch axes
        for (uint8_t i = 0; i < n_batch_axes; i++, r++) {
            if (
                y->dim[r] != x0->dim[r] ||
                y->dim[r] != x1->dim[r] ||
                x0->dim[r] != x1->dim[r]
            ) {
                return LG_StatusKind_ShapeMismatch;
            }
            y_A[r*y->rank + r] = 1;
            x0_A[r*x0->rank + r] = 1;
            x1_A[r*x1->rank + r] = 1;
            iter_domain_extents[r] = x0->dim[r];
        }

        // Free axes
        for (uint8_t i = n_batch_axes; i < x0_first_contracted_axis; i++, r++) {
            if (y->dim[r] != x0->dim[i]) {
                return LG_StatusKind_ShapeMismatch;
            }
            y_A[r*y->rank + r] = 1;
            x0_A[r*x0->rank + r] = 1;
            x1_A[r*x1->rank + r] = 0;
            iter_domain_extents[r] = x0->dim[i];
        }
        for (uint8_t i = x1_first_free_axis; i < x1->rank; i++, r++) {
            if (y->dim[r] != x1->dim[i]) {
                return LG_StatusKind_ShapeMismatch;
            }
            y_A[r*y->rank + r] = 1;
            x0_A[r*x0->rank + r] = 0;
            x1_A[r*x1->rank + r] = 1;
            iter_domain_extents[r] = x1->dim[i];
        }

        // Contracted axes
        if (n_contracted_axes > 0) {
            for (
                uint8_t x0_ax = x0_first_contracted_axis, x1_ax = x1_first_free_axis - 1;
                x0_ax < x0->rank; // x1_ax > 0
                x0_ax++, x1_ax--, r++
            ) {
                if (x0->dim[x0_ax] != x1->dim[x1_ax]) {
                    y_A[r*y->rank + r] = 0;
                    x0_A[r*x0->rank + r] = 1;
                    x1_A[r*x1->rank + r] = 1;
                    return LG_StatusKind_ShapeMismatch;
                }
                iter_domain_extents[r] = x0->dim[x0_ax];
            }
        }

        lg_assert(r == iter_domain_rank);
    }


    ////////////////
    /// ~~ fin ~~

    LG_Polyhedron iter_domain = {
        .repr_kind = LG_PolyhedronReprKind_CanonicalAABB,
        .as.canonical_aabb.rank = iter_domain_rank,
        .as.canonical_aabb.extents = iter_domain_extents,
    };

    out_y_atran->n_rows = y->rank;
    out_y_atran->n_cols = iter_domain_rank;
    out_y_atran->data = y_A;

    out_x0_atran->n_rows = x0->rank;
    out_x0_atran->n_cols = iter_domain_rank;
    out_x0_atran->data = x0_A;

    out_x1_atran->n_rows = x1->rank;
    out_x1_atran->n_cols = iter_domain_rank;
    out_x1_atran->data = x1_A;

    *out_poly = iter_domain;

    lg_assert(status == LG_StatusKind_OK);
    return status;
}
