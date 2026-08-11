#ifndef LG_AFFINE_H_
#define LG_AFFINE_H_

#include <libgrad/internal/core.h>
#include <stdint.h>

/// Affine map representing the equation y = Ax + b
/// stored row-major
typedef struct
LG_AffineTransform {
    uint8_t  n_rows, n_cols;
    int64_t  A[LG_MAX_RANK * LG_MAX_RANK]; // R^(rows, cols)
    int64_t  b[LG_MAX_RANK]; // R^(rows)
} LG_AffineTransform;

LG_AffineTransform
lg_atran_strided_from_shape(const LG_LogicalShape *shape, LG_LayoutKind layout, uint32_t unit_align);

void
lg_atran_apply(const LG_AffineTransform *tran, const int64_t x[static LG_MAX_RANK], int64_t y[static LG_MAX_RANK]);

#endif //LG_AFFINE_H_c
