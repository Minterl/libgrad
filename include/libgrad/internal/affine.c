#include <libgrad/internal/affine.h>
#include <libgrad/internal/alloc.h>

LG_AffineTransform
lg_atran_strided_from_shape(const LG_LogicalShape *shape, LG_LayoutKind layout, uint32_t unit_align) {
    LG_AffineTransform tran = {0};
    tran.n_rows = 1;
    tran.n_cols = shape->rank;

    uint8_t last_stride = 1;
    for (uint8_t i = 1; i <= shape->rank; i++) {
        uint8_t axis = layout == LG_LayoutKind_RowMajor ? shape->rank - i : i - 1;
        tran.A[axis] = last_stride;

        last_stride *= shape->dim[shape->rank - i];

        // Conceptually, we only pad the rightmost dimension.
        // However, this affects the stride of the second-rightmost dimension first
        // (and then all subsequent dimensions).
        if (unit_align > 1 && i == 1) {
            last_stride = (last_stride + unit_align - 1) & ~(unit_align - 1);
        }
    }

    return tran;
}

void
lg_atran_apply(const LG_AffineTransform *tran, const int64_t x[static LG_MAX_RANK], int64_t y[static LG_MAX_RANK]) {
    lg_memzero(y, LG_MAX_RANK * sizeof(int64_t));

    for (uint8_t i_rows = 0; i_rows < tran->n_rows; i_rows++) {
        for (uint8_t i_cols = 0; i_cols < tran->n_cols; i_cols++) {
            y[i_rows] += tran->A[tran->n_cols*i_rows + i_cols] * x[i_cols];
        }
    }

    for (uint8_t i_rows = 0; i_rows < tran->n_rows; i_rows++) {
        y[i_rows] += tran->b[i_rows];
    }
}
